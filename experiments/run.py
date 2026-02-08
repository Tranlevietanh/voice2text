from pathlib import Path
import argparse
from pipeline.voice_id_pipeline import VoiceIDPipeline
from configs.paths import (
    DIARIZER_CONFIG_FILENAME,
    VERIFIER_MODEL_FILENAME,
    EMBEDDING_DATABASE_PATH,
    THRESHOLD,
    OUTPUT_TXT_PATH,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run speaker diarization + speaker verification"
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Verification threshold (default: from config)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = args.audio
    threshold = args.threshold

    pipeline = VoiceIDPipeline(
        diarizer_config=DIARIZER_CONFIG_FILENAME,
        verifier_model=VERIFIER_MODEL_FILENAME,
        embedding_database_path=EMBEDDING_DATABASE_PATH,
    )
    results = pipeline.process(
        audio_path=audio_path,
        threshold=THRESHOLD,
        output_path=OUTPUT_TXT_PATH,
    )

    print(f"Results saved to: {OUTPUT_TXT_PATH}")


if __name__ == "__main__":
    main()
