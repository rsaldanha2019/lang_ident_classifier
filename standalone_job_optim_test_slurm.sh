#!/bin/bash

# ---------------- DEFAULTS ----------------
ENV_TYPE=""
ENV_VALUE=""
CONFIG_FILE=""
RESUME_STUDY_FROM_TRIAL_NUMBER=""
BACKEND="nccl"
CPU_CORES=""

# ---------------- DEVICE COUNT ----------------
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES not set, defaulting PPN=1"
    PPN=1
else
    IFS=',' read -ra gpu_array <<< "$CUDA_VISIBLE_DEVICES"
    PPN=${#gpu_array[@]}
fi

echo "Setting PPN=$PPN based on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---------------- ARG PARSING ----------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_TYPE_VALUE="$2"
            if [[ "$ENV_TYPE_VALUE" == "none" ]]; then
                ENV_TYPE="none"
                ENV_VALUE=""
            else
                IFS=':' read -r ENV_TYPE ENV_VALUE <<< "$ENV_TYPE_VALUE"
            fi
            shift 2
            ;;
        --config_file_path)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --resume_study_from_trial_number)
            RESUME_STUDY_FROM_TRIAL_NUMBER="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --cpu_cores)
            CPU_CORES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$ENV_TYPE" || -z "$CONFIG_FILE" ]]; then
    echo "Error: --env and --config_file_path are required"
    exit 1
fi

# ---------------- SETUP ----------------
JOB_NAME=$(basename "$CONFIG_FILE" .yaml)
WORKDIR=$(pwd)
JOB_LOG_DIR="$WORKDIR/standalone_job_log/$JOB_NAME"
mkdir -p "$JOB_LOG_DIR"

RUN_TIMESTAMP=$(date +"%Y%m%d%H%M%S")
LOG_FILE="$JOB_LOG_DIR/RUN_$RUN_TIMESTAMP.out"

MASTER_PORT=$(for port in $(seq 8000 35000); do
    nc -z localhost "$port" 2>/dev/null || { echo "$port"; break; }
done)

RESUME_ARG=""
if [[ -n "$RESUME_STUDY_FROM_TRIAL_NUMBER" ]]; then
    RESUME_ARG="--resume_study_from_trial_number=$RESUME_STUDY_FROM_TRIAL_NUMBER"
fi

CPU_ARG=""
if [[ -n "$CPU_CORES" ]]; then
    export OMP_NUM_THREADS=$CPU_CORES
    export MKL_NUM_THREADS=$CPU_CORES
    export OPENBLAS_NUM_THREADS=$CPU_CORES
    export NUMEXPR_NUM_THREADS=$CPU_CORES
    CPU_ARG="--cpu_cores=$CPU_CORES"
fi

# ---------------- CUDA PINNING WRAPPER ----------------
PIN_CUDA_CMD='
if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ -n "$LOCAL_RANK" ]; then
    IFS=\",\" read -ra DEV <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES="${DEV[$LOCAL_RANK]}"
fi
'

# ---------------- RUN ----------------
if [ "$ENV_TYPE" == "conda" ]; then
    echo "Running inside conda env: $ENV_VALUE"

    conda run -n "$ENV_VALUE" bash -c "
    export MASTER_PORT=$MASTER_PORT
    $PIN_CUDA_CMD
    python -u -m torch.distributed.run \
        --nproc-per-node=$PPN \
        --master-port=$MASTER_PORT \
        -m lang_ident_classifier.cli.lang_ident_classifier_api \
        --config=$CONFIG_FILE \
        $CPU_ARG \
        --backend=$BACKEND \
        --run_timestamp=$RUN_TIMESTAMP \
        $RESUME_ARG \
    >> \"$LOG_FILE\" 2>&1
    "

elif [ "$ENV_TYPE" == "docker" ]; then
    echo "Running inside Docker image: $ENV_VALUE"

    docker run --rm --runtime=nvidia \
        --gpus "device=$CUDA_VISIBLE_DEVICES" \
        -v "$WORKDIR:/app" \
        --ipc=host \
        -w /app \
        "$ENV_VALUE" \
        bash -c "
        export MASTER_PORT=$MASTER_PORT
        $PIN_CUDA_CMD
        python -u -m torch.distributed.run \
            --nproc-per-node=$PPN \
            --master-port=$MASTER_PORT \
            -m lang_ident_classifier.cli.lang_ident_classifier_api \
            --config=/app/$CONFIG_FILE \
            $CPU_ARG \
            --backend=$BACKEND \
            --run_timestamp=$RUN_TIMESTAMP \
            $RESUME_ARG
        " >> "$LOG_FILE" 2>&1

elif [ "$ENV_TYPE" == "none" ]; then
    echo "Running directly on host"

    bash -c "
    export MASTER_PORT=$MASTER_PORT
    $PIN_CUDA_CMD
    python -u -m torch.distributed.run \
        --nproc-per-node=$PPN \
        --master-port=$MASTER_PORT \
        -m lang_ident_classifier.cli.lang_ident_classifier_api \
        --config=$CONFIG_FILE \
        $CPU_ARG \
        --backend=$BACKEND \
        --run_timestamp=$RUN_TIMESTAMP \
        $RESUME_ARG
    " >> "$LOG_FILE" 2>&1

else
    echo "Unknown environment type: $ENV_TYPE"
    exit 1
fi

