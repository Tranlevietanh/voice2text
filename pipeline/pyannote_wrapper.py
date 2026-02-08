import torch
from pyannote.audio import Pipeline
from pathlib import Path

class Diarizer:
    def __init__(self, file_name):
        root = Path(__file__).resolve().parents[1]
        diarizer_config = root / "models" / "Pyannote" / file_name

        self.pipeline = Pipeline.from_pretrained(
            checkpoint = diarizer_config
        ).to(torch.device("cpu"))

    def diarize(self, audio_path):
        diarized_output = self.pipeline(audio_path)

        serialized_diarized_output = diarized_output.serialize()

        diarization_only_output = serialized_diarized_output.get("diarization", [])

        print (diarization_only_output)

        return diarization_only_output