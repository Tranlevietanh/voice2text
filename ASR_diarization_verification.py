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
!pip install deepfilternet

Khi cài deepfilternet mà lỗi liên quan đến torchaudio.backend thì vào python3.10/site-packages/df/io.py và xóa những cái liên quan đến AudioMetaData nhé'''

# Speaker Verification Pipeline using TitaNet-L and Silero VAD v6
# Integrates with voice-to-text systems for enrollment and verification
import os
import re
import shutil
import torch
import librosa
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
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
from onnx_infer import ONNXZipformerASR
import shutil
import argparse
import psutil
from collections import defaultdict
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from diarization_separation import SpeakerDiarizationSeparation
from collections import defaultdict
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Configuration for speaker verification pipeline"""
    # Model paths
    titanet_model_path: str = "/home/vietanh/Visual_Studio_for_Python/Zipformer_app/titanet-l.nemo"

    # Verification settings
    similarity_threshold: float = 0.5  # adjustable based on your EER tests
    min_audio_duration: float = 1.0  # seconds

    vad_threshold: float = 0.5
    vad_min_silence_duration: int = 500  # ms

    # Processing settings
    target_sample_rate: int = 16000
    device: str = "cpu"  # or "cuda" if GPU is available
    batch_size: int = 4

    use_vad: bool = True
    use_diarization: bool = True
    use_denoise: bool = True
    use_verification: bool = True

    # Storage
    temp_dir: str = "/home/vietanh/Visual_Studio_for_Python/Zipformer_app/temp"
    output_json_txt_dir: str = "/home/vietanh/Visual_Studio_for_Python/Zipformer_app/output"
    enrollment_db_path: str = "/home/vietanh/Visual_Studio_for_Python/Zipformer_app/speaker_enrollments.json"

    save_embeddings: bool = True  # whether to save extracted embeddings in results

    asr_model: bool = True  # whether to use ASR model for transcription
    asr_encoder_path: str = "/home/vietanh/Visual_Studio_for_Python/Zipformer_app/zipformer_encoder.onnx"
    asr_decoder_path: str = "/home/vietanh/Visual_Studio_for_Python/Zipformer_app/zipformer_decoder.onnx"
    asr_joiner_path: str = "/home/vietanh/Visual_Studio_for_Python/Zipformer_app/zipformer_joiner.onnx"
    asr_tokens_file: str = "/home/vietanh/Visual_Studio_for_Python/Zipformer_app/exp/lang_bpe_2000/tokens.txt"
    asr_beam_size: int = 4
    temperature = 1
    blank_penalty = 0
class DialogueSaver:
    def __init__(self, txt_path):
        self.txt_path = txt_path
        self.last_speaker = None
        self.current_text = []
        self.current_timestamp = None

    def save_segment(self, result):
        # Determine speaker name
        if result["best_match"].get("verified", False):
            speaker = result["best_match"]["speaker_id"]
        else:
            speaker = "Guest"

        # Timestamp: (start, end) in seconds
        timestamp = result.get("timestamp", (None, None))

        # Extract text
        text = result.get("asr_text", "").strip()
        if not text:
            return

        # Same speaker → append text and extend timestamp end time
        if speaker == self.last_speaker:
            self.current_text.append(text)

            # extend end timestamp
            if self.current_timestamp:
                self.current_timestamp = (
                    self.current_timestamp[0],
                    timestamp[1]
                )

        else:
            # New speaker → flush old line
            if self.last_speaker:
                start, end = self.current_timestamp
                with open(self.txt_path, "a", encoding="utf-8") as f:
                    f.write(f"[{start:.2f}–{end:.2f}] {self.last_speaker}: {' '.join(self.current_text)}\n")

            # Reset buffer
            self.last_speaker = speaker
            self.current_text = [text]
            self.current_timestamp = timestamp

    def flush(self):
        if self.last_speaker and self.current_text:
            start, end = self.current_timestamp
            with open(self.txt_path, "a", encoding="utf-8") as f:
                f.write(f"[{start:.2f}–{end:.2f}] {self.last_speaker}: {' '.join(self.current_text)}\n")


class SpeakerVerificationPipeline:
    """Production speaker verification pipeline"""

    def __init__(self, config: VerificationConfig, speaker_id, num_speakers, mode, hf_token: Optional[str] = None):
        self.config = config
        self.device = torch.device(config.device)
        self.mode = mode
        self.speaker_id = speaker_id
        self.segment_timestamp_offset = (0.0, 0.0)
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.num_speakers = num_speakers

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

            import torch
            torch.cuda.is_available = lambda : False

            start_time = time.time()

            if self.config.use_verification:
                logger.info("Loading TitaNet-L model on CPU...")
                self.speaker_model = EncDecSpeakerLabelModel.restore_from(
                    restore_path=self.config.titanet_model_path,
                    map_location="cpu"
                ).eval().to("cpu")
                logger.info("TitaNet-L loaded successfully")

            
            if self.config.use_vad:
                logger.info("Loading Silero VAD v6...")
                self.vad_model, self.vad_utils = torch.hub.load(
                    'snakers4/silero-vad', 'silero_vad', force_reload=False
                )
                self.vad_model = self.vad_model.to(self.device)
                (self.get_speech_timestamps, self.save_audio,
                self.read_audio, self.VADIterator, _) = self.vad_utils

            if self.config.use_denoise:
                logger.info("Loading DeepFilterNet...")
                self.denoise_model, self.df_state, _ = init_df()

            if self.config.use_diarization:
                if self.hf_token:
                    logger.info("Initializing Diarization Pipeline...")
                    self.separation_model = SpeakerDiarizationSeparation(
                        hf_token=self.hf_token, 
                        device="cpu"
                    )
                    # Pre-load the pipeline to avoid loading it during the first inference
                    self.separation_model.load_separation_pipeline()
                else:
                    logger.warning("No HF Token provided. Diarization features will be disabled.")
                    self.separation_model = None

            if self.mode == 0 and self.config.asr_model:
                logger.info("Loading Zipformer ONNX ASR...")
                self.asr_model = ONNXZipformerASR(
                    self.config.asr_encoder_path,
                    self.config.asr_decoder_path,
                    self.config.asr_joiner_path,
                    self.config.asr_tokens_file
                )
            elapsed = time.time() - start_time
            logger.info(f"Pipeline loading: {elapsed:.2f} seconds")
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

    def resampler (self, waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample waveform to target sample rate"""
        if orig_sr == target_sr:
            return waveform
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        return resampler(waveform)
    
    def _seconds_to_samples_tss(self, tss: List[dict], sampling_rate: int) -> List[dict]:
        """Convert coordinates expressed in seconds to sample coordinates."""
        return [{
            'start': round(crd['start']) * sampling_rate,
            'end': round(crd['end']) * sampling_rate
        } for crd in tss]
    
    def collect_chunks(self, tss: List[dict],
                   wav: torch.Tensor,
                   seconds: bool = False,
                   sampling_rate: int = None):
        if seconds and not sampling_rate:
            raise ValueError('sampling_rate must be provided when seconds is True')

        chunks = list()
        _tss = self._seconds_to_samples_tss(tss, sampling_rate) if seconds else tss

        for i in _tss:
            chunks.append(wav[i['start']:i['end']])

        return chunks
    
    def apply_vad(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ):
        """Apply VAD and return raw speech segments (no accumulation)"""

        if not self.config.use_vad or self.vad_model is None:
            return [waveform], [(0.0, waveform.shape[-1] / sample_rate)]

        try:
            # waveform: (1, T) → (T,)
            audio = waveform.squeeze(0).to(self.device).float()

            vad_timestamps = self.get_speech_timestamps(
                audio,
                self.vad_model,
                sampling_rate=sample_rate,
                threshold=self.config.vad_threshold,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
                return_seconds=True
            )

            if not vad_timestamps:
                logger.warning("No speech detected by VAD")
                return [], []

            sections = []
            timestamps = []

            for t in vad_timestamps:
                start_s = int(t["start"] * sample_rate)
                end_s   = int(t["end"] * sample_rate)

                # hard safety check
                if end_s <= start_s:
                    continue

                # slice TIME dimension correctly
                chunk = audio[start_s:end_s]   # (samples,)

                if chunk.numel() == 0:
                    continue

                # restore channel dimension → (1, samples)
                sections.append(chunk.unsqueeze(0))
                timestamps.append((t["start"], t["end"]))

            return sections, timestamps

        except Exception as e:
            logger.error(f"VAD processing failed: {e}")
            return [], []


    def denoise_and_save(self, audio_segments: list, choice: bool):

        timestamp = int(datetime.now().timestamp())
        sub_dir = "enrollment" if choice == 1 else "verification"
        save_dir = os.path.join(self.config.temp_dir, f"{sub_dir}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        timestamps = []

        for i, (segment, audio_timestamp) in enumerate(audio_segments):

            enhanced_clip = enhance(self.denoise_model, self.df_state, segment)

            enhanced_clip = self.resampler(enhanced_clip, 48000, 8000)
            sample_rate = 8000

            temp_path = os.path.join(
                save_dir,
                f"temp_{timestamp}_{i}.wav"
            )

            torchaudio.save(
                temp_path,
                enhanced_clip.cpu(),
                sample_rate
            )

            timestamps.append(audio_timestamp)

        return save_dir, timestamps     

    def process_with_diarization(
        self,
        audio_path: Optional[str] = None,
        timestamp: Optional[tuple] = None,
        num_speakers: Optional[int] = None,
        sample_rate: int = 16000,
    ):
        # 1. Safety Check: If we don't have data AND don't have a model to get it, exit.
        if self.separation_model is None:
            logger.error("No separation model initialized.")
            return []
        separated_data = None

        # 2. Inference: If no data was passed in, use the model to create it.
        if audio_path is not None:
            separated_data = self.separation_model.separate(audio_path, num_speakers)

        # 3. Validation: If the model failed to produce data, exit.
        if separated_data is None:
            return []

        audio_segments = []

        # --------------------------------------------------
        # CASE 1: separation output already provided
        # --------------------------------------------------
        if separated_data is not None:
            for speaker, data in separated_data.items():
                audio = data["audio"]
                sr = data.get("sample_rate", sample_rate)

                for segment in data["segments"]:
                    start_sample = int(segment["start"] * sr)
                    end_sample = int(segment["end"] * sr)

                    if end_sample <= start_sample:
                        continue

                    audio_segments.append({
                        "speaker": speaker,
                        "waveform": audio[start_sample:end_sample],
                        "sample_rate": sr,
                        "timestamp": (
                            segment["start"], segment["end"]
                        ) if timestamp is None else (
                            timestamp[0] + segment["start"],
                            timestamp[0] + segment["end"],
                        ),
                        "duration": segment["end"] - segment["start"],
                    })

            return audio_segments



    def save_dominant_speaker_audio(self,
        audio_segments: List[Dict],
        output_path: str,
        filename: str = "dominant_speaker.wav",
    ):
        """
        Select dominant speaker and save all their audio segments as one WAV file.

        Args:
            audio_segments: output of process_with_diarization()
            output_path: directory to save audio
            filename: output wav filename

        Returns:
            dominant_speaker label
        """
        if not audio_segments:
            raise ValueError("audio_segments is empty")

        # 1️⃣ Find dominant speaker
        durations = defaultdict(float)
        for seg in audio_segments:
            durations[seg["speaker"]] += seg["duration"]

        dominant_speaker = max(durations, key=durations.get)

        # 2️⃣ Collect segments belonging to dominant speaker
        dominant_segments = [
            seg for seg in audio_segments
            if seg["speaker"] == dominant_speaker
        ]

        # 3️⃣ Sort by timestamp to preserve speech order
        dominant_segments.sort(key=lambda s: s["timestamp"][0])

        # 4️⃣ Concatenate waveforms
        waveforms = [seg["waveform"] for seg in dominant_segments]
        full_waveform = torch.cat(waveforms, dim=-1)

        sample_rate = dominant_segments[0]["sample_rate"]

        full_waveform = self.resampler(full_waveform, sample_rate, 16000)
        sample_rate = 16000

        # 5️⃣ Save to disk
        timestamp = int(datetime.now().timestamp())

        save_dir = os.path.join(self.config.temp_dir, f"{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        out_file = os.path.join(save_dir, filename)

        torchaudio.save(
            str(out_file),
            full_waveform.unsqueeze(0),  # (1, T)
            sample_rate
        )

        return dominant_speaker, str(out_file)

    def save_diarized_segments(self,
        audio_segments,
        save_dir,
        base_name,
    ):
        """
        Save each diarized audio segment as an individual WAV file.

        Args:
            audio_segments: list returned by process_with_diarization()
            save_dir: directory to save wav files
            base_name: base name of the source audio (e.g. chunk_001)

        Returns:
            list of saved file paths
        """
        os.makedirs(save_dir, exist_ok=True)
        saved_files = []
        timestamps = []

        for i, seg in enumerate(audio_segments):
            speaker = seg["speaker"]
            sr = seg["sample_rate"]
            timestamp = seg["timestamp"]
            waveform = self.resampler(seg["waveform"], seg["sample_rate"], 16000)
            sr = 16000

            filename = (
                f"{base_name}_seg{i:03d}_"
                f"{speaker}.wav"
            )
            out_path = os.path.join(save_dir, filename)

            # torchaudio expects (channels, time)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            torchaudio.save(out_path, waveform, sr)
            saved_files.append(out_path)
            timestamps.append(timestamp)

        return saved_files, timestamps



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

    def enroll_speaker(self, audio_path: str, speaker_id: str) -> bool:
        embeddings = []
        waveform, sample_rate = torchaudio.load(audio_path)

        # Duration check
        duration = waveform.shape[-1] / sample_rate
        if duration < self.config.min_audio_duration:
            return None
        
        waveform = self.resampler(waveform, sample_rate, 16000)
        sample_rate = 16000
        
        audio_segments, timestamps = self.apply_vad(waveform, sample_rate)

        if not audio_segments:
            logger.info("Skipping verification because no speech detected.")
            return []

        resampled_audio_segments = []
        for seg, timestamp in zip(audio_segments, timestamps):
            resampled_seg = self.resampler(seg, sample_rate, 48000)
            resampled_audio_segments.append((resampled_seg, timestamp))
        sample_rate = 48000

        save_dir, timestamps = self.denoise_and_save(resampled_audio_segments, 1)
        wav_files = [f for f in os.listdir(save_dir) if f.endswith(".wav")]

        def numeric_key(f):
            m = re.search(r"_(\d+)\.wav$", f)
            return int(m.group(1)) if m else 0

        wav_files = sorted(wav_files, key=numeric_key)

        for file_name, timestamp in zip(wav_files, timestamps):
            file_path = os.path.join(save_dir, file_name)
            diarized_audio_segments = self.process_with_diarization(file_path, timestamp)
            dominant_speaker, dominant_file = self.save_dominant_speaker_audio(
                diarized_audio_segments,
                output_path=save_dir,
                filename=f"dominant_speaker_{file_name}"
            )
            embedding = self.extract_embedding(dominant_file)
            embeddings.append(embedding.tolist())

        logger.info(f"Enrolling speaker {speaker_id} with {len(wav_files)} audio files")

        if not embeddings:
            logger.error(f"No valid embeddings extracted for speaker {speaker_id}")
            return False

        # Store enrollment data
        # Append to existing speaker or create new entry
        if speaker_id in self.enrollment_db:
            existing_data = self.enrollment_db[speaker_id]

            # Append new embeddings
            existing_data["embeddings"].extend(embeddings)
            existing_data["num_samples"] = len(existing_data["embeddings"])

        else:
            # Create new entry for new speaker
            self.enrollment_db[speaker_id] = {
                "embeddings": embeddings,
                "enrollment_date": datetime.now().isoformat(),
                "num_samples": len(embeddings),
            }

        self._save_enrollment_db()
        logger.info(f"Speaker {speaker_id} enrolled successfully with {len(embeddings)} embeddings")
        shutil.rmtree(save_dir)
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
            "test_embedding": test_embedding.tolist() if self.config.save_embeddings else None,
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
        best_speaker = max(
            results["speakers"].keys(),
            key=lambda s: results["speakers"][s]["max_similarity"]
        )
        best_similarity = results["speakers"][best_speaker]["max_similarity"]

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


    def batch_verify(self, audio_path, sample_rate=None, num_speakers: Optional[int] = None, jsonl_path=None, txt_path=None, dialogue_saver=None):
        start_time = time.time()
        all_results = []
        waveform, sample_rate = torchaudio.load(audio_path)
        # Duration check
        duration = waveform.shape[-1] / sample_rate
        if duration < self.config.min_audio_duration:
            return None
        
        waveform = self.resampler(waveform, sample_rate, 16000)
        sample_rate = 16000
        
        audio_segments, timestamps = self.apply_vad(waveform, sample_rate)

        if not audio_segments:
            logger.info("Skipping verification because no speech detected.")
            return []

        resampled_audio_segments = []
        for seg, timestamp in zip(audio_segments, timestamps):
            resampled_seg = self.resampler(seg, sample_rate, 48000)
            resampled_audio_segments.append((resampled_seg, timestamp))
        sample_rate = 48000

        save_dir, timestamps = self.denoise_and_save(resampled_audio_segments, 0)
        wav_files = [f for f in os.listdir(save_dir) if f.endswith(".wav")]

        def numeric_key(f):
            m = re.search(r"_(\d+)\.wav$", f)
            return int(m.group(1)) if m else 0

        wav_files = sorted(wav_files, key=numeric_key)

        for file_name, timestamp in zip(wav_files, timestamps):
            file_path = os.path.join(save_dir, file_name)
            diarized_audio_segments = self.process_with_diarization(file_path, timestamp, None)
            base_name = os.path.splitext(file_name)[0]
            segment_files, segment_timestamps = self.save_diarized_segments(
                diarized_audio_segments,
                save_dir=save_dir,
                base_name=base_name
            )

            for seg_file, seg_ts in zip(segment_files, segment_timestamps):
                verify_result = self.verify_speaker(seg_file)
                asr_result = {}
                if self.config.asr_model:
                    asr_result = self.asr_model.transcribe_audio(
                        seg_file,
                        beam_size=self.config.asr_beam_size,
                        temperature=self.config.temperature,
                        blank_penalty=self.config.blank_penalty
                    )

                verify_result.update({f"asr_{k}": v for k, v in asr_result.items()})
                verify_result['timestamp'] = seg_ts
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(verify_result) + "\n")

                dialogue_saver.save_segment(verify_result)
                all_results.append(verify_result)

        '''shutil.rmtree(save_dir)'''
        elapsed = time.time() - start_time
        print (elapsed)
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

    def enroll_or_verify(self, audio_path, jsonl_path=None, txt_path=None, dialogue_saver=None): 
        if self.mode == 1:
            return self.enroll_speaker(audio_path, self.speaker_id)
        else:
            return self.batch_verify(
                audio_path,
                num_speakers=self.num_speakers,
                jsonl_path=jsonl_path,
                txt_path=txt_path,
                dialogue_saver=dialogue_saver
            )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Speaker Verification Pipeline using TitaNet-L, Zipformer ASR and Pyannote Diarization",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Mode Group: mutually exclusive (enrollment vs verification)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--enroll', action='store_true', help="Run in enrollment mode.")
    mode_group.add_argument('--verify', action='store_true', help="Run in verification mode.")
    
    # File input (required)
    parser.add_argument(
        '--file', 
        type=str, 
        nargs='+',  
        metavar='PATH', 
        required=True,
        help="One or more audio files to process (space-separated)."
    )

    # Required arguments for Enrollment mode
    parser.add_argument('--speaker_id', type=str, default=None, help="Required when --enroll is used. The ID of the speaker to enroll.")

    # Optional arguments for verification mode
    parser.add_argument('--num_speakers', type=int, default=1, help="Number of speakers expected in verification (default: 1).")

    parser.add_argument('--sim_threshold', type=float, default=0.5, help="Cosine similarity threshold for verification (default: 0.65).")

    args = parser.parse_args()
    
    # Validation
    if args.enroll and args.speaker_id is None:
        parser.error("--speaker_id is required when --enroll is specified.")
    
    return args

def run_file_mode(args, pipeline):
    """Runs the pipeline on one or more specified audio files."""
    audio_paths = args.file
    all_results_successful = True

    for test_audio_path in audio_paths:
        if not os.path.exists(test_audio_path):
            print(f"❌ Input file not found: {test_audio_path}. Skipping.")
            all_results_successful = False
            continue
        
        base_name = os.path.splitext(os.path.basename(test_audio_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_name_prefix = f"{base_name}_{timestamp}"
        json_path = os.path.join(pipeline.config.output_json_txt_dir, f"{output_name_prefix}.jsonl")
        txt_path = os.path.join(pipeline.config.output_json_txt_dir, f"{output_name_prefix}.txt")
        
        dialogue_saver = DialogueSaver(txt_path) 

        if args.enroll:
            print(f"\n🎬 Starting Enrollment from file: {test_audio_path} for speaker: {args.speaker_id}")
            success = pipeline.enroll_speaker(test_audio_path, args.speaker_id)
            if not success:
                print(f"❌ Enrollment failed for {test_audio_path}.")
                all_results_successful = False

        elif args.verify:
            print(f"\n🎬 Starting Verification from file: {test_audio_path}")
            results = pipeline.batch_verify(
                test_audio_path,
                jsonl_path=json_path,
                txt_path=txt_path,
                dialogue_saver=dialogue_saver
            )
            if results:
                print(f"✅ Verification completed for {test_audio_path}. Results written to {txt_path}")
            else:
                print(f"❌ Verification failed or no speech segments found in {test_audio_path}.")
                all_results_successful = False
        
        dialogue_saver.flush()

    if all_results_successful:
        print("\nPipeline finished. All files processed successfully.")
    else:
        print("\nPipeline finished with some failures.")

if __name__ == "__main__":
    args = parse_args()
    
    mode = 1 if args.enroll else 0
    speaker_id = args.speaker_id

    config = VerificationConfig(
        similarity_threshold=args.sim_threshold,
    )

    pipeline = SpeakerVerificationPipeline(
        mode=mode,
        speaker_id=speaker_id,
        num_speakers=args.num_speakers,
        config=config
    )

    run_file_mode(args, pipeline)
    sys.exit(0)
