#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ORIGINAL_SCRIPT="$SCRIPT_DIR/run_skyrl_agent_oh7b_s1.sh"

# Make sure the original script is executable
chmod +x "$ORIGINAL_SCRIPT"

# Function to run the script
run_script() {
    echo "Starting SkyRL agent training..."
    "$ORIGINAL_SCRIPT"
    return $?
}

# Main loop
while true; do
    run_script
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully!"
        break
    else
        echo "Training failed with exit code $EXIT_CODE. Restarting in 5 seconds..."
        sleep 5
    fi
done 