from pathlib import Path
from datetime import datetime
import os

DIARIZER_CONFIG_FILENAME = "pyannote_diarizer_config.yaml" #Filename only, filepath is constructed in Diarizer/Verifier class
SEPARATER_CONFIG_FILENAME = "pyannote_separater_config.yaml"
ONNX_MODEL_FILENAME = "new_separation_80k.onnx"
VERIFIER_MODEL_FILENAME = "titanet-l.nemo"

ROOT_DIR = Path(__file__).resolve().parents[1]  # V2T/

EMBEDDING_DATABASE_PATH = ROOT_DIR / "utils" / "speaker_enrollments.json"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_TXT_PATH = ROOT_DIR / "experiments" / "outputs" / f"verification_results_{TIMESTAMP}.txt"
OUTPUT_WAV_PATH = ROOT_DIR / "experiments" / "outputs" / f"{TIMESTAMP}"
THRESHOLD = 0.65
MIN_SEGMENT_DURATION = 1.0
HF_TOKEN = os.getenv("HF_TOKEN")