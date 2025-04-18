#!/bin/bash

interval_between_trials=30

# Prompt the user for the path to the script
read -p "Enter the path to the script (e.g., ./standalone.sh): " SCRIPT_PATH

# Check if the script exists and is executable
if [ ! -x "$SCRIPT_PATH" ]; then
  echo "The specified script does not exist or is not executable."
  exit 1
fi

# Prompt the user for the number of parallel trials
read -p "Enter the number of parallel trials (1 to n): " num_trials

# Trim whitespace from input
num_trials=$(echo "$num_trials" | xargs) # Trim leading/trailing whitespace

# Debug output to check the value of num_trials and its length
echo "num_trials is '$num_trials'"

# Convert to integer and validate
if [[ "$num_trials" =~ ^[0-9]+$ ]]; then
  if [ "$num_trials" -lt 1 ]; then
    echo "Invalid input. Please enter a positive integer greater than 0."
    exit 1
  fi
else
  echo "Invalid input. Please enter a positive integer greater than 0."
  exit 1
fi

# Prompt for Docker image and GPU IDs
read -p "Enter the Docker image (e.g., smallworld2020/lang_classifier:v3): " docker_image
read -p "Enter the GPU IDs (e.g., 0,1 for multiple GPUs or 'all' for all GPUs): " gpu_ids

# Validate GPU IDs (ensure it's a valid format)
if [[ "$gpu_ids" != "all" && ! "$gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Invalid GPU IDs. Please provide a comma-separated list (e.g., 0,1 or 'all')."
  exit 1
fi

# Generate a run timestamp in seconds with float precision
run_timestamp=$(date +%s.%N)

# Run the specified number of parallel trials with the same timestamp in the background
for i in $(seq 1 "$num_trials"); do
  sleep "$interval_between_trials"
  
  # Execute the script in the background with the necessary arguments
  "$SCRIPT_PATH" --docker-image="$docker_image" --gpu-ids="$gpu_ids" --run_timestamp="$run_timestamp" &
  echo "Started trial $i with timestamp $run_timestamp. Background PID: $!"

done

# Exit the script while letting the background processes continue running
echo "All trials started in the background with timestamp $run_timestamp."
exit 0

