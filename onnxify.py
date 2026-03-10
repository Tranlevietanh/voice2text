from pyannote.audio import Model
import torch
import onnxruntime as ort


def main(checkpoint: str, onnx_model: str):
    model = Model.from_pretrained(checkpoint)
    model.eval()

    dummy_input = torch.randn(1, 1, 80000)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model,
            do_constant_folding=False,
            export_params=True,
            input_names=["waveforms"],
            output_names=["scores", "sources"],
            dynamic_axes={
                "waveforms": {0: "batch_size", 1: "num_channels", 2: "num_samples"},
                "scores": {0: "batch_size", 1: "num_frames", 2: "num_sources"},
                "sources": {0: "batch_size", 1: "num_samples", 2: "num_sources"},
            }
        )

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort.InferenceSession(onnx_model, sess_options=opts)
    print("ONNX export successful.")


if __name__ == "__main__":
    main(
        "/home/vietanh/Visual_Studio_for_Python/V2T/models/Pyannote/separation/pytorch_model.bin",
        "/home/vietanh/Visual_Studio_for_Python/V2T/models/Pyannote/separation/new_separation_80k.onnx",
    )