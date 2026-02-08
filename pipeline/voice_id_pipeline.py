from pipeline.pyannote_wrapper import Diarizer
from pipeline.titanet_wrapper import Verifier
from utils.utils import load_embedding_database, write_verification_results
import torchaudio

class VoiceIDPipeline:
    def __init__(self, diarizer_config, verifier_model, embedding_database_path):
        self.diarizer = Diarizer(diarizer_config)
        self.verifier = Verifier(verifier_model)
        self.embedding_database = load_embedding_database(embedding_database_path)

    
    def extract_audio_segment(self, audio_path, start, end):
        waveform, sample_rate = torchaudio.load(audio_path)  # [1, T]

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        segment = waveform[:, start_sample:end_sample]

        return segment, sample_rate

    def process(self, audio_path, threshold=0.65, output_path=None):
        diarized_segments = self.diarizer.diarize(audio_path)

        verified_segments = []
        for segment in diarized_segments:
            start = segment["start"]
            end = segment["end"]
            speaker = segment["speaker"]

            audio_segment, _ = self.extract_audio_segment(audio_path, start, end)

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
                "score": round(best_score.item(), 3),
            })
        if output_path is not None:
            write_verification_results(verified_segments, output_path)

        return verified_segments