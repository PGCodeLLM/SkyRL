#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ORIGINAL_SCRIPT="$SCRIPT_DIR/run_skyrl_agent_oh7b_s2.sh"
CLEAR_SCRIPT="./clear.sh"

# Make sure the original script is executable
chmod +x "$ORIGINAL_SCRIPT"
chmod +x "$CLEAR_SCRIPT"

# Function to run the script
run_script() {
    echo "Clearing undead images..."
    "$CLEAR_SCRIPT"
    echo "Starting SkyRL agent training..."
    "$ORIGINAL_SCRIPT"
    return $?
}

# Main loop
while true; do
    # Create a temp file for capturing output
    OUTPUT_FILE=$(mktemp)

    # Start run_script in the background, redirecting stdout and stderr
    run_script > "$OUTPUT_FILE" 2>&1 &
    SCRIPT_PID=$!

    # Tail the output in the background for real-time display
    tail -f "$OUTPUT_FILE" &
    TAIL_PID=$!

    # Monitor the output file for updates
    LAST_MODIFIED=$(stat -c %Y "$OUTPUT_FILE")
    while kill -0 $SCRIPT_PID 2>/dev/null; do
        sleep 60
        NEW_MODIFIED=$(stat -c %Y "$OUTPUT_FILE")
        if [ "$NEW_MODIFIED" -ne "$LAST_MODIFIED" ]; then
            LAST_MODIFIED=$NEW_MODIFIED
        else
            # Check if 10 minutes (600 seconds) have passed without update
            NOW=$(date +%s)
            if [ $((NOW - LAST_MODIFIED)) -ge 600 ]; then
                echo "No output for 10 minutes. Sending SIGINT to the process and restarting..."
                kill -SIGINT $SCRIPT_PID
                break
            fi
        fi
    done

    # Wait for the script to finish if not already
    wait $SCRIPT_PID
    EXIT_CODE=$?

    # Kill the tail process
    kill $TAIL_PID 2>/dev/null
    wait $TAIL_PID 2>/dev/null

    # Remove the temp file
    rm -f "$OUTPUT_FILE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully!"
        break
    else
        echo "Training failed with exit code $EXIT_CODE. Restarting in 5 seconds..."
        sleep 5
    fi

done 