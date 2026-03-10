import onnxruntime as ort
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from pyannote.audio.core.io import Audio
from pyannote.core import SlidingWindow
from pyannote.audio.utils.receptive_field import conv1d_num_frames

class ToTaToNetONNXProxy(nn.Module):
    """
    A Proxy class that wraps an ONNX session to look like a PyTorch Model.
    This allows pyannote.audio Pipelines to use ONNX for inference.
    """
    def __init__(self, onnx_path: str, device: str = "cpu"):
        super().__init__()
        
        # 1. Initialize ONNX Session
        providers = ['CPUExecutionProvider']
            
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        # 2. Mock attributes required by pyannote.audio.Model
        # ToTaToNet expects 16kHz audio
        self.audio = Audio(sample_rate=16000, mono="downmix")
        self.hparams = nn.ParameterDict({
            "sample_rate": 16000, 
            "num_channels": 1
        })
        
        # This is used by the pipeline to determine sliding window steps
        # These values must match your exported ToTaToNet config
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        spec_diarization = Specifications(
            duration=5.0,
            resolution=Resolution.FRAME,
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            classes=[f"speaker_{i}" for i in range(3)],
            permutation_invariant=True
        )

        spec_separation = Specifications(
            duration=5.0,
            resolution=Resolution.FRAME,
            problem=Problem.MULTI_LABEL_CLASSIFICATION, # Or appropriate problem type
            classes=[f"source_{i}" for i in range(3)],
            permutation_invariant=True
        )

        # You MUST provide a tuple of specifications for a multi-output model
        self.specifications = (spec_diarization, spec_separation)
        self.diarization_scaling = 8 
        self.encoder_stride = 16
        self.encoder_kernel_size = 32

    def forward(self, waveforms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass that the pyannote Inference engine calls.
       
        """
        # waveforms shape: (batch, channel, sample)
        
        # Prepare input for ONNX
        onnx_input = {
            self.input_name: waveforms.cpu().numpy().astype(np.float32)
        }
        
        # Run ONNX inference
        # Returns: [scores, sources]
        onnx_outputs = self.session.run(None, onnx_input)
        
        # Convert back to torch Tensors for the Pipeline's aggregation logic
        scores = torch.from_numpy(onnx_outputs[0])
        sources = torch.from_numpy(onnx_outputs[1])
        
        return scores, sources
    
    def num_frames(self, num_samples: int) -> int:
        """
        Replicates the ToTaToNet.num_frames logic exactly.
        """
        equivalent_stride = self.diarization_scaling * self.encoder_stride
        equivalent_kernel_size = self.diarization_scaling * self.encoder_kernel_size

        return conv1d_num_frames(
            num_samples, 
            kernel_size=equivalent_kernel_size, 
            stride=equivalent_stride
        )
    
    @property
    def receptive_field(self) -> SlidingWindow:
        # duration = equivalent_kernel_size / sample_rate = 256 / 16000
        # step = equivalent_stride / sample_rate = 128 / 16000
        return SlidingWindow(
            start=0.0,
            duration=0.016, # Exact value for 256 samples
            step=0.008      # Exact value for 128 samples (125 FPS)
        )