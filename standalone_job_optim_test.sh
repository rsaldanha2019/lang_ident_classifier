#!/usr/bin/env bash
set -euo pipefail

# --- DEFAULTS ---
ENV_TYPE=""
ENV_VALUE=""
CONFIG_FILE=""
BACKEND="nccl"
CPU_CORES=""

usage() {
    echo "Usage: $0 --env <conda:env|docker:image|none> --config_file_path <config.yaml> [--backend nccl|gloo] [--cpu_cores N]"
    exit 1
}

if [ $# -eq 0 ]; then usage; fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_TYPE_VALUE="$2"
            if [[ "$ENV_TYPE_VALUE" == "none" ]]; then
                ENV_TYPE="none"
                ENV_VALUE=""
            else
                IFS=':' read -r ENV_TYPE ENV_VALUE <<< "$ENV_TYPE_VALUE"
                if [[ "$ENV_TYPE" != "conda" && "$ENV_TYPE" != "docker" ]]; then
                    echo "Error: --env must be one of: conda:<env>, docker:<image>, none"
                    exit 1
                fi
            fi
            shift 2
            ;;
        --config_file_path)
            CONFIG_FILE="$2"
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
            usage
            ;;
    esac
done

if [[ -z "$ENV_TYPE" ]]; then
    echo "Error: --env is required"
    exit 1
fi
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config_file_path <file.yaml> is required"
    exit 1
fi

JOB_NAME=$(basename "$CONFIG_FILE" .yaml)
WORKDIR=$(pwd)
JOB_LOG_DIR=$WORKDIR/standalone_job_log/$JOB_NAME
mkdir -p "$JOB_LOG_DIR"

# PPN
if [ -z "${CUDA_VISIBLE_DEVICES-}" ]; then
    echo "CUDA_VISIBLE_DEVICES not set, defaulting PPN=1"
    PPN=1
else
    IFS=',' read -ra gpu_array <<< "$CUDA_VISIBLE_DEVICES"
    PPN=${#gpu_array[@]}
fi
echo "PPN=$PPN based on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"

# Port + timestamp
MASTER_PORT=$(for port in $(seq 8000 35000); do nc -z localhost "$port" 2>/dev/null || { echo "$port"; break; }; done)
RUN_TIMESTAMP=$(date +"%Y%m%d%H%M%S")

CPU_ARG=""
if [[ -n "${CPU_CORES}" ]]; then
    export OMP_NUM_THREADS=$CPU_CORES
    export MKL_NUM_THREADS=$CPU_CORES
    export OPENBLAS_NUM_THREADS=$CPU_CORES
    export NUMEXPR_NUM_THREADS=$CPU_CORES
    CPU_ARG="--cpu_cores=$CPU_CORES"
    echo "[INFO] Threads set to $CPU_CORES"
fi

PY_CMD="python -u -m torch.distributed.run \
    --nproc-per-node $PPN \
    --master-port $MASTER_PORT \
    -m lang_ident_classifier.cli.lang_ident_classifier_api \
    --config_file_path=$CONFIG_FILE \
    $CPU_ARG \
    --backend=$BACKEND \
    --run_timestamp=$RUN_TIMESTAMP

# --- RUN ---
if [ "$ENV_TYPE" == "conda" ]; then
    echo "Running in conda env: $ENV_VALUE"
    setsid bash -c "conda run -n \"$ENV_VALUE\" bash -c '
      export MASTER_PORT=$MASTER_PORT
      $PY_CMD >> \"$JOB_LOG_DIR/RUN_$RUN_TIMESTAMP.out\" 2>&1
    '" </dev/null &
    disown
    echo "Detached conda job. Log: $JOB_LOG_DIR/RUN_$RUN_TIMESTAMP.out"

elif [ "$ENV_TYPE" == "docker" ]; then
    echo "Running in Docker image: $ENV_VALUE"
    MY_UID=$(id -u)
    MY_GID=$(id -g)
    MY_UNAME=$(whoami)
    GPU_FLAG=""
    if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
        GPU_FLAG="--gpus device=$CUDA_VISIBLE_DEVICES"
    fi
    CONTAINER_NAME="run_${RUN_TIMESTAMP}"
    docker run -d --name "$CONTAINER_NAME" --restart unless-stopped \
        $GPU_FLAG \
        -v "$WORKDIR:/app" \
        --ipc=host \
        -w /app \
        -e UID="$MY_UID" \
        -e GID="$MY_GID" \
        -e USERNAME="$MY_UNAME" \
        -e HF_CACHE=/app/.cache \
        "$ENV_VALUE" \
        /bin/bash -c "export MASTER_PORT=$MASTER_PORT && $PY_CMD"
    echo "Detached docker container $CONTAINER_NAME. Logs: docker logs -f $CONTAINER_NAME"

elif [ "$ENV_TYPE" == "none" ]; then
    echo "Running directly on host"
    setsid bash -c "
      export MASTER_PORT=$MASTER_PORT
      $PY_CMD >> \"$JOB_LOG_DIR/RUN_$RUN_TIMESTAMP.out\" 2>&1
    " </dev/null &
    disown
    echo "Detached host job. Log: $JOB_LOG_DIR/RUN_$RUN_TIMESTAMP.out"

else
    echo "Unknown env: $ENV_TYPE"
    exit 1
fi

cat <<EOF
Job started
Timestamp: $RUN_TIMESTAMP
Job name: $JOB_NAME
Log dir: $JOB_LOG_DIR
Master port: $MASTER_PORT

Check progress:
- Host/Conda: tail -f "$JOB_LOG_DIR/RUN_$RUN_TIMESTAMP.out"
- Docker: docker logs -f run_${RUN_TIMESTAMP}
EOF

