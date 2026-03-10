#!/bin/bash

# Exit immediately if something fails
set -e

# --------- CONFIG ---------
AUDIO_PATH="data/audio_file_2.wav"
THRESHOLD=0.65
OUTPUT_DIR="experiments/outputs"
# --------------------------

echo "Running VoiceID pipeline..."
echo "Audio: $AUDIO_PATH"
echo "Threshold: $THRESHOLD"
echo "Output dir: $OUTPUT_DIR"
echo

python3 -m experiments.run \
  --audio "$AUDIO_PATH" \
  --threshold "$THRESHOLD" 
