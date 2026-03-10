import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speech_separation import SpeechSeparation
from pathlib import Path


class Diarizer:
    def __init__(self, file_name, hf_token=None):
        root = Path(__file__).resolve().parents[1]
        diarizer_config = root / "models" / "Pyannote" / file_name

        if diarizer_config.exists():
            print(f"Loading diarization pipeline from local path: {diarizer_config}")
            self.pipeline = Pipeline.from_pretrained(
                checkpoint=str(diarizer_config)
            ).to(torch.device("cpu"))

        else:
            print("Local diarizer not found. Loading from HuggingFace...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=hf_token
            ).to(torch.device("cpu"))

    def diarize(self, audio_path):
        diarized_output = self.pipeline(audio_path)

        serialized_diarized_output = diarized_output.serialize()

        diarization_only_output = serialized_diarized_output.get("diarization", [])

        print (diarization_only_output)

        return diarization_only_output
    
class Separater: 
    def __init__(self, file_name, hf_token=None):
        root = Path(__file__).resolve().parents[1]
        separater_config = root / "models" / "Pyannote" / file_name

        if separater_config.exists():
            print(f"Loading separation pipeline from local path: {separater_config}")
            self.pipeline = Pipeline.from_pretrained(
                checkpoint=str(separater_config)
            ).to(torch.device("cpu"))

        else:
            print("Local separater not found. Loading from HuggingFace...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/separation-ami-1.0",
                token=hf_token
            ).to(torch.device("cpu"))


    def separate(self, audio_path):
        diarized_info, separated_output = self.pipeline(audio_path)
        model = self.pipeline._segmentation.model
        sample_rate = model.hparams.get("sample_rate", 16000)

        sources = separated_output.data  # shape: (num_samples, num_speakers)

        # Get speaker labels (ordered consistently with columns)
        speaker_labels = diarized_info.labels()

        speaker_audios = {}

        for idx, speaker in enumerate(speaker_labels):
            speaker_waveform = sources[:, idx]
            speaker_audios[speaker] = speaker_waveform
        
        print (diarized_info)

        return speaker_audios, diarized_info, sample_rate
