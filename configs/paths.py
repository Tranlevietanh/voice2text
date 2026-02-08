from pathlib import Path
from datetime import datetime
DIARIZER_CONFIG_FILENAME = "pyannote_config.yaml" #Filename only, filepath is constructed in Diarizer/Verifier class
VERIFIER_MODEL_FILENAME = "titanet-l.nemo"

ROOT_DIR = Path(__file__).resolve().parents[1]  # V2T/

EMBEDDING_DATABASE_PATH = ROOT_DIR / "utils" / "speaker_enrollments.json"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_TXT_PATH = ROOT_DIR / "experiments" / "outputs" / f"verification_results_{TIMESTAMP}.txt"

THRESHOLD = 0.65