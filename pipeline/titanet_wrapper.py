from pathlib import Path
import torch
import nemo.collections.asr as nemo_asr
import numpy as np


class Verifier:
    def __init__(self, file_name):
        root = Path(__file__).resolve().parents[1]
        restore_path = root / "models" / file_name

        if restore_path.exists() and restore_path.is_file():
            print(f"Loading model from local file: {restore_path}")
            self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
                restore_path=str(restore_path),
                map_location=torch.device("cpu"),
            )
        else:
            print("Local model not found. Loading pretrained 'titanet-large'...")
            self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name="titanet-large",
                map_location=torch.device("cpu"),
            )

    def verify(self, audio_segment, database_embedding):

        embedding, _ = self.model.infer_segment(audio_segment)

        embedding = embedding.squeeze().detach().cpu().numpy().reshape(-1)

        database_embedding = database_embedding.squeeze().detach().cpu().numpy().reshape(-1)

        dot_product = np.dot(embedding, database_embedding)
        norms = np.linalg.norm(embedding) * np.linalg.norm(database_embedding)

        if norms == 0:
            return 0.0

        return float(dot_product / norms)
    

    
    