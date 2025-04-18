#!/bin/bash

set -e

# Default values
gpu_ids="all"
docker_image=""
run_timestamp=""

# Parse arguments
for arg in "$@"; do
  case $arg in
    --docker-image=*)
      docker_image="${arg#*=}"
      shift
      ;;
    --gpu-ids=*)
      gpu_ids="${arg#*=}"
      shift
      ;;
    --run_timestamp=*)
      run_timestamp="${arg#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $0 --docker-image=\"<image>\" [--gpu-ids=\"<gpu_ids>\"] [--run_timestamp=\"<timestamp>\"]"
      exit 1
      ;;
  esac
done

if [ -z "$docker_image" ]; then
  echo "Error: --docker-image is required."
  exit 1
fi

if [ -z "$run_timestamp" ]; then
  echo "Error: --run_timestamp is required."
  exit 1
fi

# CPU thread settings
total_cores=$(nproc)
num_threads=$(echo "($total_cores * 0.2)/1" | bc)
if [ "$num_threads" -lt 1 ]; then num_threads=1; fi
export OMP_NUM_THREADS=$num_threads

# Paths
WORKDIR=$(pwd)
JOB_NAME=lang_ident_classifier_optim_test
JOB_LOG_DIR="$WORKDIR/standalone_job_log/$JOB_NAME"
mkdir -p "$JOB_LOG_DIR"

# Find free MASTER_PORT
MASTER_PORT=$(for port in $(seq 8000 35000); do nc -z localhost "$port" 2>/dev/null || { echo "$port"; break; }; done)

# Derive PPN
if [[ "$gpu_ids" == "all" ]]; then
  gpu_flag="all"
  PPN=$(nvidia-smi -L | wc -l)
else
  gpu_flag="\"device=$gpu_ids\""
  PPN=$(echo "$gpu_ids" | tr ',' '\n' | wc -l)
fi

# Run docker
docker run --rm --runtime=nvidia --gpus "$gpu_flag" \
  -v "$WORKDIR":"$WORKDIR" \
  -w "/app" \
  --ipc=host \
  "$docker_image" \
  run-hyperparam \
    --config=app_config_optim_test.yaml \
    --backend=nccl \
    --run_timestamp "$run_timestamp" \
    >> "$JOB_LOG_DIR/RUN_$run_timestamp.out" 2>&1


