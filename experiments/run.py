from pathlib import Path
import argparse
from pipeline.voice_id_pipeline import VoiceIDPipeline, VoiceIDWithONNXSeparationPipeline, VoiceIDWithSeparationPipeline
from configs.paths import (
    DIARIZER_CONFIG_FILENAME,
    SEPARATER_CONFIG_FILENAME,
    ONNX_MODEL_FILENAME,
    VERIFIER_MODEL_FILENAME,
    EMBEDDING_DATABASE_PATH,
    THRESHOLD,
    OUTPUT_TXT_PATH,
    OUTPUT_WAV_PATH,
    HF_TOKEN,
    MIN_SEGMENT_DURATION,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run speaker diarization + speaker verification or enrollment"
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["verify", "enroll"],
        help="Select mode: verify or enroll",
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["Pytorch", "ONNX"],
        help="Model version to use for inference",
    )

    parser.add_argument(
        "--speaker_id",
        type=str,
        help="Speaker ID (required for enroll mode)",
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

    if args.version == "Pytorch":
        pipeline = VoiceIDWithSeparationPipeline(
            separater_config=SEPARATER_CONFIG_FILENAME,
            verifier_model=VERIFIER_MODEL_FILENAME,
            embedding_database_path=EMBEDDING_DATABASE_PATH,
        )
    elif args.version == "ONNX":
        pipeline = VoiceIDWithONNXSeparationPipeline(
            separater_config=SEPARATER_CONFIG_FILENAME,
            onnx_model_name=ONNX_MODEL_FILENAME,
            verifier_model=VERIFIER_MODEL_FILENAME,
            embedding_database_path=EMBEDDING_DATABASE_PATH,
        hf_token=HF_TOKEN
    )

    if args.mode == "verify":
        results = pipeline.process_verify(
            audio_path=args.audio,
            threshold=args.threshold,
            output_txt_path=OUTPUT_TXT_PATH,
            output_wav_path=OUTPUT_WAV_PATH,
            min_duration=MIN_SEGMENT_DURATION
        )

        print(f"Verification results saved to: {OUTPUT_TXT_PATH}")

    elif args.mode == "enroll":
        if not args.speaker_id:
            raise ValueError("speaker_id is required when mode=enroll")

        pipeline.process_enroll(
            audio_path=args.audio,
            speaker_id=args.speaker_id,
            min_duration=MIN_SEGMENT_DURATION
        )

        print(f"Enrollment completed for speaker: {args.speaker_id}")


if __name__ == "__main__":
    main()
