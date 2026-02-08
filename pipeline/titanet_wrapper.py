import torch
import nemo.collections.asr as nemo_asr
from pathlib import Path


class Verifier: 
    def __init__(self, file_name):
        root = Path(__file__).resolve().parents[1]
        restore_path = root / "models" / file_name
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(restore_path=str(restore_path), map_location = torch.device("cpu"))

    def verify(self, audio_segment, database_embedding): 
        audio_segment = audio_segment.squeeze(0)  # Remove batch dimension if present
        embedding, _ = self.model.infer_segment(audio_segment)
        embedding = embedding.squeeze(0)  # Remove batch dimension
        embedding = embedding / torch.linalg.norm(embedding)
        database_embedding = database_embedding / torch.linalg.norm(database_embedding)

        similarity_score = torch.dot(embedding, database_embedding) / ((torch.dot(embedding, embedding) * torch.dot(database_embedding, database_embedding)) ** 0.5)
        similarity_score = (similarity_score + 1) / 2

        return similarity_score
    
    
