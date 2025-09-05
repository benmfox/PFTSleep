#!/bin/bash

# This script runs the PFTSleep inference process.
# It uses the pftsleep_inference.py script and a YAML configuration file.

# --- Configuration ---
# Path to the Python inference script
# Ensure this script is in the same directory or provide the full path.
INFERENCE_SCRIPT="pftsleep_inference.py"

# Path to the YAML configuration file
# Ensure this file is in the same directory or provide the full path.
CONFIG_FILE="pftsleep_inference_config.yaml"

# --- Script Logic ---

echo "--- PFTSleep Inference Script ---"
echo "Starting inference process..."
echo ""

# Check if the inference script exists
if [ ! -f "$INFERENCE_SCRIPT" ]; then
    echo "Error: The inference script '$INFERENCE_SCRIPT' was not found."
    echo "Please ensure '$INFERENCE_SCRIPT' is in the current directory or update the INFERENCE_SCRIPT variable."
    exit 1
fi

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: The configuration file '$CONFIG_FILE' was not found."
    echo "Please ensure '$CONFIG_FILE' is in the current directory or update the CONFIG_FILE variable."
    exit 1
fi

echo "Running inference with script: $INFERENCE_SCRIPT"
echo "Using configuration file: $CONFIG_FILE"
echo ""

# Execute the Python inference script
# The Python script will handle model download if necessary.
# If models are not found, it will prompt for a Hugging Face token to download them.
python "$INFERENCE_SCRIPT" "$CONFIG_FILE"

# Check the exit status of the Python script
if [ $? -eq 0 ]; then
    echo ""
    echo "PFTSleep inference completed successfully."
    echo "Predictions should be saved to the path specified in '$CONFIG_FILE' (e.g., 'preds.pt')."
else
    echo ""
    echo "PFTSleep inference failed. Please check the error messages above."
    exit 1
fi

exit 0
