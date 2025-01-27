#!/bin/bash
YOUR_COMMAND="sh scripts/train_mix3_compare.sh 0,1,2,3,4,5,6,7"

# Define the interval (in seconds) at which to check GPU utilization
CHECK_INTERVAL=100

# Number of consecutive checks with 0 utilization required to trigger your command
CONSECUTIVE_CHECKS_REQUIRED=2
CONSECUTIVE_CHECKS=0

# Main loop
while true; do
    # Get the GPU utilization for all GPUs using nvidia-smi
    GPU_UTILIZATION=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    # Check if all GPUs have 0 utilization
    if echo $GPU_UTILIZATION | grep -q "0 0 0 0 0 0 0 0"; then
        CONSECUTIVE_CHECKS=$((CONSECUTIVE_CHECKS + 1))
    else
        CONSECUTIVE_CHECKS=0
    fi

    if [ "$CONSECUTIVE_CHECKS" -ge "$CONSECUTIVE_CHECKS_REQUIRED" ]; then
        echo "All GPUs have 0 utilization for at least 200 seconds. Running your command..."
    # Run your command here
        $YOUR_COMMAND
        break
    else
        echo "Not all GPUs have 0 utilization for 200 seconds. Waiting..."
    fi

    sleep $CHECK_INTERVAL
done

echo "Script finished."
