import librosa
from pipeline.pyannote_wrapper import Diarizer, Separater
from pipeline.pyannote_onnx_wrapper import ONNXSpeechSeparationPipeline
from pipeline.titanet_wrapper import Verifier
from utils.utils import load_embedding_database, write_verification_results
from pathlib import Path
import soundfile as sf
import numpy as np

class VoiceIDPipeline:
    def __init__(self, diarizer_config, verifier_model, embedding_database_path, hf_token=None):
        self.diarizer = Diarizer(diarizer_config, hf_token=hf_token)
        self.verifier = Verifier(verifier_model)
        self.embedding_database = load_embedding_database(embedding_database_path)


    def extract_audio_segment(self, waveform, sample_rate, start, end):

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        segment = waveform[start_sample:end_sample]

        return segment

    def process_verify(self, audio_path, threshold, output_txt_path=None, output_wav_path=None, min_duration=1.0):
        waveform, sample_rate = sf.read(audio_path)  # [1, T]
        diarized_segments = self.diarizer.diarize(audio_path)
        if not diarized_segments:
            print("No diarized segments found.")
            return []
        
        verified_segments = []

        if output_wav_path is not None:
            output_wav_path = Path(output_wav_path)

            if output_wav_path.exists() and not output_wav_path.is_dir():
                raise ValueError(f"{output_wav_path} exists but is not a directory")


            output_wav_path.mkdir(parents=True, exist_ok=True)

            segment_counter = 0

        for segment in diarized_segments:
            start = segment["start"]
            end = segment["end"]
            speaker = segment["speaker"]

            audio_segment = self.extract_audio_segment(waveform, sample_rate, start, end)

            if output_wav_path is not None:
                filename = f"segment_{segment_counter:04d}_{start:.2f}_{end:.2f}.wav"
                segment_path = output_wav_path / filename

                sf.write(str(segment_path), audio_segment, self.verifier.model._cfg.train_ds.sample_rate)
                segment_counter += 1 

            duration = end - start
            if duration < min_duration:
                continue    

            if sample_rate != self.verifier.model._cfg.train_ds.sample_rate:
                audio_segment = librosa.core.resample(audio_segment, orig_sr=sample_rate, target_sr=self.verifier.model._cfg.train_ds.sample_rate)


            best_speaker = None
            best_score = 0.0


            for speaker_id, speaker_data in self.embedding_database.items():
                for speaker_embedding in speaker_data:
                    score = self.verifier.verify(
                        audio_segment,
                        speaker_embedding
                    )

                    if score > best_score:
                        best_score = score
                        best_speaker = speaker_id

            verified_segments.append({
                "start": start,
                "end": end,
                "speaker": best_speaker if best_score >= threshold else "Guest",
                "score": round(best_score, 3),
            })
        if output_txt_path is not None:
            write_verification_results(verified_segments, output_txt_path)

        return verified_segments
    
    def process_enroll(self, audio_path, speaker_id, min_duration=1.0):

        diarized_segments = self.diarizer.diarize(audio_path)
        if not diarized_segments:
            print("No diarized segments found.")
            return

        speaker_durations = {}

        # 1️⃣ Compute total duration per speaker
        for segment in diarized_segments:
            spk = segment["speaker"]
            start = segment["start"]
            end = segment["end"]

            duration = end - start
            speaker_durations[spk] = speaker_durations.get(spk, 0) + duration

        if not speaker_durations:
            print("No speaker durations found.")
            return

        dominant_speaker = max(speaker_durations, key=speaker_durations.get)

        print(f"Dominant speaker: {dominant_speaker}")
        print(f"Total duration: {speaker_durations[dominant_speaker]:.2f}s")

        # 2️⃣ Collect all segments of dominant speaker
        active_chunks = []

        for segment in diarized_segments:
            if segment["speaker"] != dominant_speaker:
                continue

            start = segment["start"]
            end = segment["end"]

            audio_segment, sample_rate = self.extract_audio_segment(
                audio_path, start, end
            )

            active_chunks.append(audio_segment)

        if not active_chunks:
            print("No speech segments found for dominant speaker.")
            return

        # 3️⃣ Concatenate into full speech
        full_speech = np.concatenate(active_chunks)

        total_duration = len(full_speech) / sample_rate
        if total_duration < min_duration:
            print("Not enough total speech for enrollment.")
            return

        # 4️⃣ Resample if needed
        target_sr = self.verifier.model._cfg.train_ds.sample_rate
        if sample_rate != target_sr:
            full_speech = librosa.resample(
                full_speech,
                orig_sr=sample_rate,
                target_sr=target_sr
            )

        # 5️⃣ Extract ONE embedding
        embedding, _ = self.verifier.model.infer_segment(full_speech)
        embedding = embedding.squeeze(0).detach().cpu().numpy()

        # 7️⃣ Store
        if speaker_id not in self.embedding_database:
            self.embedding_database[speaker_id] = []

        self.embedding_database[speaker_id].append(embedding.tolist())

        print(f"Enrolled speaker '{speaker_id}' successfully.")

class VoiceIDWithSeparationPipeline:
    def __init__(self, separater_config, verifier_model, embedding_database_path, hf_token=None):
        self.separater = Separater(separater_config, hf_token=hf_token)
        self.verifier = Verifier(verifier_model)
        self.embedding_database = load_embedding_database(embedding_database_path)
    
    def merge_segments(self, segments, max_gap=1.0):
        if not segments:
            return []

        merged = []
        current_start = segments[0].start
        current_end = segments[0].end

        for seg in segments[1:]:
            gap = seg.start - current_end

            if gap <= max_gap:
                current_end = seg.end
            else:
                merged.append((current_start, current_end))
                current_start = seg.start
                current_end = seg.end

        merged.append((current_start, current_end))
        return merged

    def process_verify(self, audio_path, threshold, output_txt_path=None, output_wav_path=None, min_duration=1.0):

        separated_segments, diarization_info, sample_rate = self.separater.separate(audio_path)

        if not separated_segments:
            print("No separated segments found.")
            return []
        
        labels = diarization_info.labels()

        verified_speakers = []
        target_sr = self.verifier.model._cfg.train_ds.sample_rate

        for speaker in labels:
            timeline = diarization_info.label_timeline(speaker)
            segments = list(timeline)

            if not segments:
                continue

            merged_segments = self.merge_segments(segments, max_gap=1.0)

            speaker_waveform = separated_segments[speaker]
            active_chunks = []

            for start, end in merged_segments:
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                active_chunks.append(speaker_waveform[start_sample:end_sample])

            if not active_chunks:
                continue

            trimmed_waveform = np.concatenate(active_chunks)

            total_duration = len(trimmed_waveform) / sample_rate
            if total_duration < min_duration:
                continue

            if sample_rate != target_sr:
                trimmed_waveform = librosa.resample(
                    trimmed_waveform,
                    orig_sr=sample_rate,
                    target_sr=target_sr
                )

            if output_wav_path is not None:
                output_wav_path = Path(output_wav_path)
                output_wav_path.mkdir(parents=True, exist_ok=True)
                filename = f"{speaker}.wav"
                sf.write(str(output_wav_path / filename), trimmed_waveform, target_sr)

            best_speaker = None
            best_score = 0.0

            for speaker_id, speaker_data in self.embedding_database.items():
                for speaker_embedding in speaker_data:
                    score = self.verifier.verify(trimmed_waveform, speaker_embedding)

                    if score > best_score:
                        best_score = score
                        best_speaker = speaker_id

            identified_speaker = best_speaker if best_score >= threshold else "Guest"

            for start, end in merged_segments:
                verified_speakers.append({
                    "speaker_label": speaker,
                    "start": start,
                    "end": end,
                    "speaker": identified_speaker,
                    "score": round(best_score, 3),
                })
        if output_txt_path is not None:
            write_verification_results(verified_speakers, output_txt_path)

        return verified_speakers
    
    def process_enroll(self, audio_path, speaker_id, min_duration=1.0, max_gap=1.0):

        separated_segments, diarization_info, sample_rate = self.separater.separate(audio_path)

        if not separated_segments:
            print("No separated segments found.")
            return

        labels = diarization_info.labels()

        speaker_durations = {}
        speaker_merged = {}

        for speaker in labels:
            timeline = diarization_info.label_timeline(speaker)
            segments = list(timeline)

            if not segments:
                continue

            merged_segments = self.merge_segments(segments, max_gap=max_gap)

            total_duration = sum(end - start for start, end in merged_segments)

            speaker_durations[speaker] = total_duration
            speaker_merged[speaker] = merged_segments

        if not speaker_durations:
            print("No valid speaker durations found.")
            return

        dominant_speaker = max(speaker_durations, key=speaker_durations.get)

        print(f"Dominant speaker: {dominant_speaker}")
        print(f"Merged duration: {speaker_durations[dominant_speaker]:.2f}s")

        embeddings = []
        target_sr = self.verifier.model._cfg.train_ds.sample_rate

        speaker_waveform = separated_segments[dominant_speaker]
        active_chunks = []

        for start, end in speaker_merged[dominant_speaker]:

            duration = end - start
            if duration < min_duration:
                continue

            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            active_chunks.append(
                speaker_waveform[start_sample:end_sample]
            )

        if not active_chunks:
            print("No valid speech chunks found.")
            return

        full_speech = np.concatenate(active_chunks)

        target_sr = self.verifier.model._cfg.train_ds.sample_rate

        if sample_rate != target_sr:
            full_speech = librosa.resample(
                full_speech,
                orig_sr=sample_rate,
                target_sr=target_sr
            )

        embedding, _ = self.verifier.model.infer_segment(full_speech)
        embedding = embedding.squeeze(0).detach().cpu().numpy()

        if speaker_id not in self.embedding_database:
            self.embedding_database[speaker_id] = []

        self.embedding_database[speaker_id].append(embedding.tolist())

        print(f"Enrolled speaker '{speaker_id}' successfully.")

class VoiceIDWithONNXSeparationPipeline:
    def __init__(self, separater_config, onnx_model_name, verifier_model, embedding_database_path, hf_token=None):
        """
        Uses ONNXSpeechSeparationPipeline to extract isolated speaker waveforms,
        then verifies identities using Titanet.
        """
        # 1. Initialize your custom ONNX-injected separation pipeline
        # This uses your existing ONNXSpeechSeparationPipeline class
        print(f"Loading ONNX Separation model...")
        self.separater_onnx = ONNXSpeechSeparationPipeline(separater_config, onnx_model_name, hf_token=hf_token)
        
        # 2. Standard Verifier (Titanet)
        self.verifier = Verifier(verifier_model)
        
        # 3. Embedding Database
        self.embedding_database = load_embedding_database(embedding_database_path)
        self.target_sr = self.verifier.model._cfg.train_ds.sample_rate

    def process_verify(self, audio_path, threshold, output_txt_path=None, output_wav_path=None, min_duration=1.0):
        separated_segments, diarization_info, sample_rate = self.separater_onnx(audio_path)

        if not separated_segments:
            print("No separated segments found.")
            return []
        
        labels = diarization_info.labels()

        verified_speakers = []
        target_sr = self.verifier.model._cfg.train_ds.sample_rate

        for speaker in labels:
            timeline = diarization_info.label_timeline(speaker)
            segments = list(timeline)

            if not segments:
                continue

            merged_segments = self.merge_segments(segments, max_gap=1.0)

            speaker_waveform = separated_segments[speaker]
            active_chunks = []

            for start, end in merged_segments:
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                active_chunks.append(speaker_waveform[start_sample:end_sample])

            if not active_chunks:
                continue

            trimmed_waveform = np.concatenate(active_chunks)

            total_duration = len(trimmed_waveform) / sample_rate
            if total_duration < min_duration:
                continue

            if sample_rate != target_sr:
                trimmed_waveform = librosa.resample(
                    trimmed_waveform,
                    orig_sr=sample_rate,
                    target_sr=target_sr
                )

            if output_wav_path is not None:
                output_wav_path = Path(output_wav_path)
                output_wav_path.mkdir(parents=True, exist_ok=True)
                filename = f"{speaker}.wav"
                sf.write(str(output_wav_path / filename), trimmed_waveform, target_sr)

            best_speaker = None
            best_score = 0.0

            for speaker_id, speaker_data in self.embedding_database.items():
                for speaker_embedding in speaker_data:
                    score = self.verifier.verify(trimmed_waveform, speaker_embedding)

                    if score > best_score:
                        best_score = score
                        best_speaker = speaker_id

            identified_speaker = best_speaker if best_score >= threshold else "Guest"

            for start, end in merged_segments:
                verified_speakers.append({
                    "speaker_label": speaker,
                    "start": start,
                    "end": end,
                    "speaker": identified_speaker,
                    "score": round(best_score, 3),
                })
        if output_txt_path is not None:
            write_verification_results(verified_speakers, output_txt_path)

        return verified_speakers

    def process_enroll(self, audio_path, speaker_id, min_duration=1.0, max_gap=1.0):

        separated_segments, diarization_info, sample_rate = self.separater_onnx(audio_path)

        if not separated_segments:
            print("No separated segments found.")
            return

        labels = diarization_info.labels()

        speaker_durations = {}
        speaker_merged = {}

        for speaker in labels:
            timeline = diarization_info.label_timeline(speaker)
            segments = list(timeline)

            if not segments:
                continue

            merged_segments = self.merge_segments(segments, max_gap=max_gap)

            total_duration = sum(end - start for start, end in merged_segments)

            speaker_durations[speaker] = total_duration
            speaker_merged[speaker] = merged_segments

        if not speaker_durations:
            print("No valid speaker durations found.")
            return

        dominant_speaker = max(speaker_durations, key=speaker_durations.get)

        print(f"Dominant speaker: {dominant_speaker}")
        print(f"Merged duration: {speaker_durations[dominant_speaker]:.2f}s")

        embeddings = []
        target_sr = self.verifier.model._cfg.train_ds.sample_rate

        speaker_waveform = separated_segments[dominant_speaker]
        active_chunks = []

        for start, end in speaker_merged[dominant_speaker]:

            duration = end - start
            if duration < min_duration:
                continue

            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            active_chunks.append(
                speaker_waveform[start_sample:end_sample]
            )

        if not active_chunks:
            print("No valid speech chunks found.")
            return

        full_speech = np.concatenate(active_chunks)

        target_sr = self.verifier.model._cfg.train_ds.sample_rate

        if sample_rate != target_sr:
            full_speech = librosa.resample(
                full_speech,
                orig_sr=sample_rate,
                target_sr=target_sr
            )

        embedding, _ = self.verifier.model.infer_segment(full_speech)
        embedding = embedding.squeeze(0).detach().cpu().numpy()

        if speaker_id not in self.embedding_database:
            self.embedding_database[speaker_id] = []

        self.embedding_database[speaker_id].append(embedding.tolist())

        print(f"Enrolled speaker '{speaker_id}' successfully.")
