import json
import torch

def load_embedding_database(database_path):
    with open(database_path, "r") as f:
        database = json.load(f)
    
    database_embeddings = {}
    for speaker_id, speaker_data in database.items():
        database_embeddings[speaker_id] = [
            torch.tensor(e, dtype=torch.float32)
            for e in speaker_data["embeddings"]
        ]
    return database_embeddings

from pathlib import Path

def write_verification_results(results, output_path):
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        for r in results:
            start = r["start"]
            end = r["end"]
            speaker = r["speaker"]
            score = r["score"]

            f.write(
                f"[{start:07.3f} - {end:07.3f}] "
                f"speaker={speaker} score={score:.3f}\n"
            )

