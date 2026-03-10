from pyannote.audio import Pipeline
from .separater_onnx import ToTaToNetONNXProxy
import torch
from pathlib import Path

class ONNXSpeechSeparationPipeline:
    def __init__(self, file_name: str, onnx_model_name: str, hf_token=None):
        """
        Loads the Pyannote pipeline structure and replaces the PyTorch 
        segmentation model with an ONNX proxy.
        """
        root = Path(__file__).resolve().parents[1]
        separater_config = root / "models" / "Pyannote" / file_name
        device = torch.device("cpu")

        # 1. Load the Pipeline structure
        if separater_config.exists():
            print(f"Loading pipeline structure from local path: {separater_config}")
            self.pipeline = Pipeline.from_pretrained(checkpoint=str(separater_config))
        else:
            print(f"Local config not found at {separater_config}. Loading from HuggingFace...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/separation-ami-1.0", 
                use_auth_token=hf_token
            )
        onnx_model_path = root / "models" / "Pyannote" / "separation" / onnx_model_name
        # 2. Inject ONNX Model
        print(f"Injecting ONNX model from {onnx_model_path}...")
        self.onnx_proxy = ToTaToNetONNXProxy(onnx_model_path, device="cpu")
        
        # Replace the segmentation model with our ONNX proxy
        self.pipeline._segmentation.model = self.onnx_proxy
        self.pipeline.to(device)

    def __call__(self, audio_file: str):
        diarized_info, separated_output = self.pipeline(audio_file)
        model = self.pipeline._segmentation.model
        sample_rate = model.hparams.get("sample_rate", 16000)

        sources = separated_output.data  # shape: (num_samples, num_speakers)

        # Get speaker labels (ordered consistently with columns)
        speaker_labels = diarized_info.labels()

        speaker_audios = {}

        for idx, speaker in enumerate(speaker_labels):
            speaker_waveform = sources[:, idx]
            speaker_audios[speaker] = speaker_waveform
        

        return speaker_audios, diarized_info, sample_rate