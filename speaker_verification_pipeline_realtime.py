'''!pip install nemo_toolkit[asr]
!pip install megatron-core
!pip install torch torchaudio torchvision
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Rust to PATH for this shell
import os
os.environ["PATH"] += ":/root/.cargo/bin"

# Install build tools
!apt-get update
!apt-get install -y build-essential python3-dev
!pip install --upgrade pip setuptools wheel
!pip install deepfilternet'''

# Speaker Verification Pipeline using TitaNet-L and Silero VAD v6
# Integrates with voice-to-text systems for enrollment and verification
import os
import torch
import librosa
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from scipy.io import wavfile
from df import init_df, enhance
import pandas as pd
import torchaudio
import sys
import queue
import threading
import time
import sounddevice as sd
import wave

from nemo.collections.asr.models import EncDecSpeakerLabelModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerificationConfig:
    """Configuration for speaker verification pipeline"""
    # Model paths
    titanet_model_path: str = "/content/drive/MyDrive/titanet-l.nemo"

    # VAD settings
    use_vad: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_duration: int = 250  # ms
    vad_max_speech_duration: int = np.inf  # ms

    # Verification settings
    similarity_threshold: float = 0.6  # adjustable based on your EER tests
    min_audio_duration: float = 1.0  # seconds
    max_audio_duration: float = np.inf  # seconds

    verify_audio_duration: int = 5
    enroll_audio_duration: int = 20

    # Processing settings
    target_sample_rate: int = 16000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4

    use_denoise: bool = True

    # Storage
    temp_dir: str = "/content/drive/MyDrive/integration/temp"
    enrollment_db_path: str = "./speaker_enrollments.json"

    # Output settings
    output_format: str = "json"  # json, csv, or both
    save_embeddings: bool = True
    save_vad_segments: bool = False

class SpeakerVerificationPipeline:
    """Production speaker verification pipeline"""

    def __init__(self, config: VerificationConfig, speaker_id, mode):
        self.config = config
        self.device = torch.device(config.device)
        self.mode = mode
        self.speaker_id = speaker_id
        # Initialize models
        self._load_models()

        # Create directories
        os.makedirs(config.temp_dir, exist_ok=True)

        # Load enrollment database
        self.enrollment_db = self._load_enrollment_db()

        logger.info(f"Speaker verification pipeline initialized on {self.device}")

    def _load_models(self):
        """Load TitaNet-L and Silero VAD models"""
        try:
            # Load TitaNet-L
            logger.info("Loading TitaNet-L model...")
            self.speaker_model = EncDecSpeakerLabelModel.restore_from(
                self.config.titanet_model_path
            )
            self.speaker_model = self.speaker_model.eval().to(self.device)
            logger.info("TitaNet-L loaded successfully")

            # Load Silero VAD
            if self.config.use_vad:
                logger.info("Loading Silero VAD v6...")
                self.vad_model, self.vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
                self.vad_model = self.vad_model.to(self.device)
                (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, _) = self.vad_utils
                logger.info("Silero VAD v6 loaded successfully")
            else:
                self.vad_model = None
                self.vad_utils = None

            if self.config.use_denoise:
                logger.info("Loading DeepFilterNet...")
                self.denoise_model, self.df_state, _ = init_df()
            else:
                self.denoise_model = None

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _load_enrollment_db(self) -> Dict:
        """Load speaker enrollment database"""
        if os.path.exists(self.config.enrollment_db_path):
            with open(self.config.enrollment_db_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_enrollment_db(self):
        """Save speaker enrollment database"""
        with open(self.config.enrollment_db_path, 'w') as f:
            json.dump(self.enrollment_db, f, indent=2)

    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Preprocess audio: load, resample, convert to mono"""
        # Load audio with librosa (automatically converts to mono and resamples)
        waveform, sample_rate = librosa.load(
            audio_path,
            sr=self.config.target_sample_rate,  # Automatically resample to target rate
            mono=True,  # Convert to mono
            dtype=np.float32
        )

        # Ensure 1D array (librosa.load already returns 1D for mono=True)
        waveform = waveform.flatten()

        return waveform, sample_rate
    def _seconds_to_samples_tss(tss: List[dict], sampling_rate: int) -> List[dict]:
        """Convert coordinates expressed in seconds to integer sample coordinates."""
        return [
            {
                'start': int(round(crd['start'] * sampling_rate)),
                'end': int(round(crd['end'] * sampling_rate))
            }
            for crd in tss
        ]
    def collect_chunks(self, tss: List[dict],
                   wav: torch.Tensor,
                   seconds: bool = False,
                   sampling_rate: int = None) -> torch.Tensor:
        if seconds and not sampling_rate:
            raise ValueError('sampling_rate must be provided when seconds is True')

        chunks = list()
        _tss = self._seconds_to_samples_tss(tss, sampling_rate) if seconds else tss

        for i in _tss:
            chunks.append(wav[i['start']:i['end']])

        return torch.cat(chunks)


    def apply_vad(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply VAD to extract speech segments"""
        if not self.config.use_vad or self.vad_model is None:
            return waveform

        try:
            # Convert numpy to torch tensor for VAD
            audio_tensor = torch.from_numpy(waveform).float()
            audio_tensor = audio_tensor.to(self.device)
          
            # Get speech timestamps
            timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=sample_rate,
                threshold=self.config.vad_threshold,
                min_speech_duration_ms=self.config.vad_min_speech_duration,
                max_speech_duration_s=self.config.vad_max_speech_duration // 1000,
            )
          

            if not timestamps:
                logger.warning("No speech detected by VAD")
                return waveform

            # Collect speech chunks
            speech_audio = self.collect_chunks(timestamps, audio_tensor)
          
            if speech_audio is None or speech_audio.numel() == 0:
                logger.warning("VAD returned empty audio")
                return waveform

            segment_duration = (self.config.enroll_audio_duration if self.mode == 1 else self.config.verify_audio_duration)

            section_samples = segment_duration * sample_rate
            sections = []

            start = 0
            while start < len(speech_audio):
                end = start + section_samples
                sections.append(speech_audio[start:end])
                start = end
        
         
            return sections

        except Exception as e:
            logger.warning(f"VAD failed, using original audio: {e}")
            return waveform


    def denoise_and_save(self, speech_audio: list, sample_rate: int, choice: bool):
        """
        Args:
            speech_audio: list of 1D torch tensors [samples]
            sample_rate: sample rate of the audio
            choice: 1 = enroll, 0 = verify
        """

        timestamp = int(datetime.now().timestamp())
        sub_dir = "enrollment" if choice == 1 else "verification"
        save_dir = os.path.join(self.config.temp_dir, f"{sub_dir}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        Upsampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
        Downsampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=sample_rate)

        # Save to WAV files
        for i, clip in enumerate(speech_audio):
            clip = torch.as_tensor(clip, dtype=torch.float32, device='cpu')
            Upsampler = Upsampler.to('cpu')
            clip = Upsampler(clip)
            clip = clip.unsqueeze(0)
            enhanced_clip = enhance(self.denoise_model, self.df_state, clip)
            enhanced_clip = Downsampler(enhanced_clip)
            enhanced_clip = enhanced_clip.squeeze(0)
            temp_path = os.path.join(
                save_dir,
                f"temp_{timestamp}_{i}.wav"
            )
            waveform_int16 = (enhanced_clip.numpy() * 32767).astype(np.int16)
            wavfile.write(temp_path, sample_rate, waveform_int16)
            print(f"Saved {sub_dir} clip {i+1} -> {temp_path}")

        return save_dir

    def extract_embedding(self, file) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio file"""
        try:
            # Extract embedding using NeMo
            with torch.no_grad():
                embedding = self.speaker_model.get_embedding(file)
                embedding = embedding.cpu().numpy().squeeze()

            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed for {file}: {e}")
            return None

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(emb1, emb2)
        norms = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norms == 0:
            return 0.0
        return float(dot_product / norms)
    def enroll_speaker_from_file (self, speaker_id: str, audio_path: str) -> bool:
        waveform, sample_rate = self.preprocess_audio(audio_path)
        return self.enroll_speaker(speaker_id, waveform, sample_rate)

    def enroll_speaker(self, speaker_id: str, waveform, sample_rate = None) -> bool:
        """Enroll a speaker with multiple audio samples"""
        embeddings = []
        if sample_rate == None: 
            sample_rate = self.config.target_sample_rate
        # Check duration
        duration = len(waveform) / sample_rate
        if duration < self.config.min_audio_duration:
            logger.warning(f"Audio too short: {duration:.2f}s < {self.config.min_audio_duration}s")
            return None

        if duration > self.config.max_audio_duration:
            logger.warning(f"Audio too long: {duration:.2f}s, truncating to {self.config.max_audio_duration}s")
            max_samples = int(self.config.max_audio_duration * sample_rate)
            waveform = waveform[:max_samples]
        

        # Apply VAD
        speech_audio = self.apply_vad(waveform, sample_rate)
        save_dir = self.denoise_and_save(speech_audio, sample_rate, 1)  # returns path to enrollment folder
        wav_files = [f for f in os.listdir(save_dir) if f.endswith(".wav")]

        for file_name in wav_files:
            file_path = os.path.join(save_dir, file_name)
            embedding = self.extract_embedding(file_path)
            embeddings.append(embedding.tolist())

        logger.info(f"Enrolling speaker {speaker_id} with {len(wav_files)} audio files")

        if not embeddings:
            logger.error(f"No valid embeddings extracted for speaker {speaker_id}")
            return False

        # Store enrollment data
        self.enrollment_db[speaker_id] = {
            "embeddings": embeddings,
            "enrollment_date": datetime.now().isoformat(),
            "num_samples": len(embeddings),
            "audio_path": save_dir
        }

        self._save_enrollment_db()
        logger.info(f"Speaker {speaker_id} enrolled successfully with {len(embeddings)} embeddings")
        return True

    def verify_speaker(self, audio_path: str) -> Dict:
        """Verify speaker identity against enrollment database"""
        # Extract embedding from test audio
        test_embedding = self.extract_embedding(audio_path)

        if test_embedding is None:
            return {
                "success": False,
                "error": "Failed to extract embedding from test audio",
                "audio_path": audio_path
            }

        results = {
            "success": True,
            "audio_path": audio_path,
            '''"test_embedding": test_embedding.tolist() if self.config.save_embeddings else None,'''
            "verification_time": datetime.now().isoformat(),
            "speakers": {}
        }

        # Compare with all enrolled speakers (or specific speaker if claimed)
        speakers_to_check = list(self.enrollment_db.keys())

        for speaker_id in speakers_to_check:
            if speaker_id not in self.enrollment_db:
                continue

            enrolled_embeddings = self.enrollment_db[speaker_id]["embeddings"]
            similarities = []

            # Compare with all enrollment samples
            for enrolled_emb in enrolled_embeddings:
                enrolled_emb = np.array(enrolled_emb)
                similarity = self.cosine_similarity(test_embedding, enrolled_emb)
                similarities.append(similarity)

            # Calculate statistics
            max_similarity = max(similarities)
            avg_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)

            # Verification decision
            is_verified = max_similarity >= self.config.similarity_threshold

            results["speakers"][speaker_id] = {
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "std_similarity": std_similarity,
                "num_comparisons": len(similarities),
                "is_verified": is_verified,
                "threshold_used": self.config.similarity_threshold
            }

            # Find best match
            best_speaker = None
            best_similarity = 0.0

            for speaker_id, data in results["speakers"].items():
                if data["max_similarity"] > best_similarity:
                    best_similarity = data["max_similarity"]
                    best_speaker = speaker_id

            results["best_match"] = {
                "speaker_id": best_speaker,
                "similarity": best_similarity,
                "verified": best_similarity >= self.config.similarity_threshold
            }
        if results["best_match"]["verified"]:
          print(f"✅ Verified speaker: {best_speaker} "
                f"(similarity={best_similarity:.3f})")
        else:
            print(f"❌ Guest")

        return results

    def batch_verify_from_file(self, audio_path: str = None):   
        waveform, sample_rate = self.preprocess_audio(audio_path)
        return self.batch_verify(waveform, sample_rate)

    def batch_verify(self, waveform, sample_rate = None):
        """Batch verification of one audio file but can contain multiple sections with different voices"""
        all_results = []

        if sample_rate == None: 
            sample_rate = self.config.target_sample_rate
        # Check duration
        duration = len(waveform) / sample_rate
        if duration < self.config.min_audio_duration:
            logger.warning(f"Audio too short: {duration:.2f}s < {self.config.min_audio_duration}s")
            return None

        if duration > self.config.max_audio_duration:
            logger.warning(f"Audio too long: {duration:.2f}s, truncating to {self.config.max_audio_duration}s")
            max_samples = int(self.config.max_audio_duration * sample_rate)
            waveform = waveform[:max_samples]

        # Apply VAD
        speech_audio = self.apply_vad(waveform, sample_rate)
        save_dir = self.denoise_and_save(speech_audio, sample_rate, 0)  # returns path to enrollment folder
        wav_files = [f for f in os.listdir(save_dir) if f.endswith(".wav")]

        for file_name in wav_files:
            file_path = os.path.join(save_dir, file_name)
            results = self.verify_speaker(file_path)
            all_results.append(results)
            df = pd.DataFrame(all_results)
            df.to_json(os.path.join(save_dir, "results.json"))
        return all_results

    def get_enrollment_stats(self) -> Dict:
        """Get statistics about enrolled speakers"""
        total_speakers = len(self.enrollment_db)
        total_samples = sum(data["num_samples"] for data in self.enrollment_db.values())

        return {
            "total_enrolled_speakers": total_speakers,
            "total_enrollment_samples": total_samples,
            "speakers": {
                speaker_id: {
                    "num_samples": data["num_samples"],
                    "enrollment_date": data["enrollment_date"]
                }
                for speaker_id, data in self.enrollment_db.items()
            }
        }
    def enroll_or_verify(self, waveform): 
        if self.mode == 1:
            return self.enroll_speaker(waveform, self.speaker_id)
        else:
            return self.batch_verify(waveform)

# Convenience functions for integration with voice-to-text systems

def create_pipeline(config_path: str = None) -> SpeakerVerificationPipeline:
    """Create speaker verification pipeline with optional config file"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = VerificationConfig(**config_dict)
    else:
        config = VerificationConfig()

    return SpeakerVerificationPipeline(config)

def quick_verify(audio_path: str, speaker_id: str = None,
                pipeline: SpeakerVerificationPipeline = None) -> Dict:
    """Quick verification function for integration"""
    if pipeline is None:
        pipeline = create_pipeline()

    return pipeline.verify_speaker(audio_path, speaker_id)

class RealtimeRecorder:
    def __init__(self, samplerate=44100, channels=1, mode, device_name="UM02", pipeline: SpeakerVerificationPipeline):
        self.samplerate = samplerate
        self.channels = channels
        self.device_name = device_name
        self.pipeline = pipeline
        self.mode = mode
        self.q_audio_1 = queue.Queue()
        self.q_audio_2 = queue.Queue()
        self.q_process = queue.Queue()
        self.current_segment_buffer = []
        self.stop_event = threading.Event()

        self.stream1 = None
        self.stream2 = None
        self.writer_thread_obj = None
        self.processor_thread_obj = None
        self.verify_audio_duration: int = 5
        self.enroll_audio_duration: int = 20

    # === Mic detection ===
    def is_mic_connected(self):
        devices = sd.query_devices()
        indices = [
            i for i, dev in enumerate(devices)
            if dev['max_input_channels'] > 0 and self.device_name.lower() in dev['name'].lower()
        ]
        if len(indices) == 0:
            return False, 0, None, None
        elif len(indices) == 1:
            return True, 1, indices[0], None
        else:
            return True, 2, indices[0], indices[1]

    # === Audio callbacks ===
    def audio_callback_1(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.q_audio_1.put(indata.copy().astype(np.int16))

    def audio_callback_2(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.q_audio_2.put(indata.copy().astype(np.int16))

    def flush_queue(self, q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    # === Writer thread ===
    def writer_thread(self, two_mic=False):
        self.current_segment_buffer = []
        segment_start_time = time.time()

        while not self.stop_event.is_set():
            try:
                if two_mic:
                    chunk1 = self.q_audio_1.get(timeout=0.5)
                    chunk2 = self.q_audio_2.get(timeout=0.5)
                    chunk = np.column_stack((chunk1, chunk2))
                else:
                    chunk = self.q_audio_1.get(timeout=0.5)
                self.current_segment_buffer.append(chunk)

                segment_duration = (self.enroll_audio_duration if self.mode == 1 else self.verify_audio_duration)

                if time.time() - segment_start_time >= self.segment_duration:
                    print("🕔 5-minute chunk reached — saving segment.")
                    segment_data = np.concatenate(self.current_segment_buffer, axis=0)
                    self.current_segment_buffer = []
                    segment_start_time = time.time()
                    self.q_process.put(segment_data)

            except queue.Empty:
                continue

    def final_flush(self, two_mic=False):
        if two_mic:
            while not self.q_audio_1.empty() and not self.q_audio_2.empty():
                try:
                    c1 = self.q_audio_1.get_nowait()
                    c2 = self.q_audio_2.get_nowait()
                    chunk = np.column_stack((c1, c2))
                    self.current_segment_buffer.append(chunk)
                except queue.Empty:
                    break
        else:
            while not self.q_audio_1.empty():
                try:
                    c1 = self.q_audio_1.get_nowait()
                    self.current_segment_buffer.append(c1)
                except queue.Empty:
                    break

        if self.stop_event.is_set() and self.current_segment_buffer:
            print("💾 Saving final buffer...")
            segment_data = np.concatenate(self.current_segment_buffer, axis=0)
            self.current_segment_buffer = []
            self.process_segment(segment_data)
            print("✅ Final flush done.")


    # === Recording controls ===
    def start_recording(self, two_mic=False):
        self.stop_event.clear()
        _, _, idx1, idx2 = self.is_mic_connected()
        d1 = sd.query_devices(idx1)
        print(f"🎙 Recording from: {d1['name']}")

        self.writer_thread_obj = threading.Thread(target=self.writer_thread, args=(two_mic,), daemon=True)
        self.writer_thread_obj.start()
      
        self.processor_thread_obj = threading.Thread(target=self.processing_thread, daemon=True)
        self.processor_thread_obj.start()


        self.stream1 = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=idx1,
            callback=self.audio_callback_1,
            dtype='int16',
            blocksize=8192
        )
        self.stream1.start()

        self.stream2 = None
        if two_mic and idx2 is not None:
            d2 = sd.query_devices(idx2)
            print(f"🎙 Second mic: {d2['name']}")
            self.stream2 = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                device=idx2,
                callback=self.audio_callback_2,
                dtype='int16',
                blocksize=8192
            )
            self.stream2.start()

        return self.stream1, self.stream2, self.writer_thread_obj

    def stop_recording(self, two_mic=False):
        self.stop_event.set()

        for s in [self.stream1, self.stream2]:
            if s:
                try:
                    s.stop()
                    s.close()
                except Exception as e:
                    print(f"⚠️ Error stopping stream: {e}")

        if self.writer_thread_obj:
            self.writer_thread_obj.join()
        if self.processor_thread_obj:
            self.processor_thread_obj.join()

        self.final_flush(two_mic)
        self.flush_queue(self.q_audio_1)
        self.flush_queue(self.q_audio_2)

        print("🛑 Recording stopped cleanly.")
    def process_segment(self, data: np.ndarray):
        """Send the chunk directly to your VAD / verification pipeline."""
        waveform = data.astype(np.float32) / 32768.0  # convert int16 to float32
        if waveform.ndim == 2:  # stereo → mono
            waveform = waveform.mean(axis=1)
        waveform = torch.from_numpy(waveform)  # convert to tensor
        waveform = waveform.unsqueeze(0)
        downsampler = torchaudio.transforms.Resample(orig_freq=self.samplerate, new_freq=16000)
        waveform = downsampler(waveform)
        waveform = waveform.squeeze(0)
        # Send to pipeline
        self.pipeline.enroll_or_verify(waveform)

    def processing_thread(self):
        while not self.stop_event.is_set() or not self.q_process.empty():
            try:
                segment = self.q_process.get(timeout=0.5)
                self.process_segment(segment)
            except queue.Empty:
                continue

def choose_mode():
    print("Welcome! Please choose an option:")
    print("1. Enroll a new speaker")
    print("0. Verify a speaker")
    
    while True:
        choice = input("Enter 1 or 0: ").strip()
        if choice == "1":
            mode = 1
            speaker_id = input("Enter speaker ID for enrollment: ").strip()
            break
        elif choice == "0":
            mode = 0
            speaker_id = None
            break
        else:
            print("Invalid choice, try again.")
    
    return mode, speaker_id

if __name__ == "__main__":
    # 1️⃣ Let user choose mode first

    mode, speaker_id = choose_mode()

    # 2️⃣ Configure your verification pipeline
    config = VerificationConfig(
        similarity_threshold=0.65,
        use_vad=True,
        vad_threshold=0.5,
    )

    pipeline = SpeakerVerificationPipeline(
        mode=mode,
        speaker_id=speaker_id,
        config=config
    )

    # 3️⃣ Create and start the realtime recorder
    realtime = RealtimeRecorder(mode=mode, pipeline=pipeline)

    mic_connected, mic_count, idx1, idx2 = realtime.is_mic_connected()
    if not mic_connected:
      print("❌ No UM02 microphone detected. Exiting.")
      sys.exit(1)
    if mic_count == 2: 
      two_mic = True
    else: 
      two_mic = False

    print("Starting recording... Press Ctrl+C to stop.")
    try:
        s1, s2, writer_thread = realtime.start_recording(two_mic=two_mic)
        while True:
            time.sleep(0.1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("Stopping recording...")
        realtime.stop_recording(two_mic=two_mic)
        sys.exit(1)
