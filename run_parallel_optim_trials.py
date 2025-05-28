import argparse
import shutil
import subprocess
import time
import os
import socket
import random
from datetime import datetime

def is_docker_available():
    """Check if Docker is available on the system"""
    try:
        subprocess.run(["docker", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def is_conda_available():
    """Check if Conda is available on the system"""
    try:
        subprocess.run(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def find_free_port(start_port=8000, end_port=35000):
    """Find a free port between the specified range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))
            if result != 0:
                return port
    raise Exception("Could not find a free MASTER_PORT.")

def get_job_name_from_yaml(config_file_path):
    """Get the job name from the YAML file or use the filename if not available."""
    return os.path.splitext(os.path.basename(config_file_path))[0]

def get_current_timestamp():
    """Generate a timestamp in yyyymmddhhmmsss format"""
    return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

def run_job(config_file_path, gpu_ids='all', run_timestamp=''):
    if not run_timestamp:
        raise ValueError("Error: --run_timestamp is required.")

    if run_timestamp == 'now':
        run_timestamp = get_current_timestamp()

    job_name = get_job_name_from_yaml(config_file_path)

    total_cores = os.cpu_count()
    num_threads = max(1, int(total_cores * 0.2))
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    workdir = os.getcwd()
    job_log_dir = os.path.join(workdir, 'standalone_job_log', job_name)
    os.makedirs(job_log_dir, exist_ok=True)

    master_port = find_free_port()

    if gpu_ids == 'all':
        try:
            result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            ppn = len(result.stdout.decode().splitlines())
        except subprocess.CalledProcessError:
            ppn = 1
    else:
        ppn = len(gpu_ids.split(','))

    # Main execution inside Docker or Conda
    print(f"Running distributed training for job '{job_name}' with {ppn} process(es) on GPUs: {gpu_ids}")
    
    torchrun_command = [
        'python', '-m', 'torch.distributed.run',
        '--nproc-per-node', str(ppn),
        '--master-port', str(master_port),
        '-m', 'lang_ident_classifier.cli.hyperparam_selection_model_optim',
        '--config', config_file_path,
        '--backend', 'nccl',
        '--run_timestamp', run_timestamp,
    ]

    log_path = os.path.join(job_log_dir, f"RUN_{run_timestamp}.out")
    with open(log_path, 'w') as log_file:
        subprocess.run(torchrun_command, stdout=log_file, stderr=log_file)

def run_trial(trial_num, config_file_path, gpu_ids, run_timestamp):
    print(f"Running trial {trial_num} with GPU(s) {gpu_ids} and timestamp {run_timestamp}...")
    try:
        run_job(config_file_path=config_file_path, gpu_ids=gpu_ids, run_timestamp=run_timestamp)
    except subprocess.CalledProcessError as e:
        print(f"Error during trial {trial_num} execution: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run parallel optimization trials.")
    parser.add_argument('--config_file_path', type=str, required=True, help='Path to the config yaml to run (e.g., ./app_config.yaml)')
    parser.add_argument('--num_trials', type=int, required=True, help='Number of trials to run')
    parser.add_argument('--gpu_ids', type=str, required=True, help="GPU IDs to use (e.g., '0,1' or 'all')")

    args = parser.parse_args()

    if args.num_trials < 1:
        print("Error: The number of trials must be at least 1.")
        return

    if args.gpu_ids != 'all' and not all(gpu.strip().isdigit() for gpu in args.gpu_ids.split(',')):
        print("Error: Invalid GPU IDs. Please provide a comma-separated list (e.g., 0,1 or 'all').")
        return

    if not os.path.exists(args.config_file_path):
        print(f"Error: The specified file does not exist: {args.config_file_path}")
        return

    for i in range(1, args.num_trials + 1):
        run_timestamp = get_current_timestamp()
        time.sleep(5)
        run_trial(i, args.config_file_path, args.gpu_ids, run_timestamp)

if __name__ == "__main__":
    main()
