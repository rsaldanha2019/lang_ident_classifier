import argparse
import shutil
import subprocess
import time
import os
import socket
import yaml
import random
from datetime import datetime

def is_docker_available():
    """Check if Docker is available on the system"""
    try:
        subprocess.run(["docker", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False

def is_conda_available():
    """Check if Conda is available on the system"""
    try:
        subprocess.run(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False

def find_free_port(start_port=8000, end_port=35000):
    """Find a free port between the specified range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))
            if result != 0:  # Port is free
                return port
    raise Exception("Could not find a free MASTER_PORT.")

def get_job_name_from_yaml(config_file_path):
    """Get the job name from the YAML file or use the filename if not available."""
    job_name = os.path.splitext(os.path.basename(config_file_path))[0]
    return job_name

def get_current_timestamp():
    """Generate a timestamp in yyyymmddhhmmsss format (compact format)"""
    return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # Get the timestamp down to milliseconds (excluding last 3 digits of microseconds)

def run_job(config_file_path, docker_image='', gpu_ids='all', run_timestamp=''):
    # Check if run_timestamp is provided
    if not run_timestamp:
        raise ValueError("Error: --run_timestamp is required.")
    
    # Use the provided run_timestamp or generate a new one
    if run_timestamp == 'now':
        run_timestamp = get_current_timestamp()

    # Get the job name from the config file
    job_name = get_job_name_from_yaml(config_file_path)

    # CPU thread settings
    total_cores = os.cpu_count()
    num_threads = max(1, int(total_cores * 0.2))  # Calculate number of threads as 20% of total cores
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    # Paths
    workdir = os.getcwd()
    job_log_dir = os.path.join(workdir, 'standalone_job_log', job_name)
    os.makedirs(job_log_dir, exist_ok=True)

    # Find free MASTER_PORT using Python's socket library
    master_port = find_free_port()

    # Derive PPN and GPU flag
    if gpu_ids == 'all':
        gpu_flag = 'all'
        ppn = len(subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.splitlines())
    else:
        gpu_flag = f"\"device={gpu_ids}\""
        ppn = len(gpu_ids.split(','))


    # Running with Docker
    if docker_image:
        print(f"Running with Docker for job '{job_name}' using GPUs {gpu_ids}...")
        docker_command = [
            'docker', 'run', '--rm', '--runtime=nvidia', '--gpus', gpu_flag,
            '-v', f"{workdir}:/app",
            '--privileged',
            '--ipc=host',
            '--shm-size=16g',
            '--ulimit', 'nofile=65536:65536',
            '-w', '/app',
            docker_image,
            'python', '-m', 'torch.distributed.run',
            '--nproc-per-node', str(ppn),
            '--master-port', str(master_port),
            '-m', 'lang_ident_classifier.cli.hyperparam_selection_model_optim',
            '--config', f"/app/{os.path.relpath(config_file_path, workdir)}",
            '--backend', 'nccl',
            '--run_timestamp', run_timestamp,
        ]


        # First torchrun execution
        with open(os.path.join(job_log_dir, f"RUN_{run_timestamp}.out"), 'w') as log_file:
            subprocess.run(docker_command, stdout=log_file, stderr=log_file)
        
        # Wait for 5-10 seconds before the second run
        # time.sleep(random.randint(5, 10))
        
        # Second torchrun execution
        # with open(os.path.join(job_log_dir, f"RUN_{run_timestamp}_2.out"), 'w') as log_file:
        #     subprocess.run(docker_command, stdout=log_file, stderr=log_file)

    # Mimicking Docker's behavior with Conda
    elif shutil.which('conda'):
        print(f"Docker not found, mimicking Docker behavior with Conda for job '{job_name}' using GPUs {gpu_ids}...")
        
        # Ensure Conda environment is activated, then run the job
        conda_command = [
            'conda', 'run', '-n', 'lang_ident_classifier',
            'bash', '-c', f'''
                export PYTHONPATH=$(pwd)/lib/python3.10/site-packages:$PYTHONPATH && \
                export MASTER_PORT={master_port} && \
                python -m torch.distributed.run \
                --nproc-per-node={ppn} \
                --master-port={master_port} \
                -m lang_ident_classifier.cli.hyperparam_selection_model_optim \
                --config={config_file_path} \
                --backend=nccl \
                --run_timestamp={run_timestamp} >> {os.path.join(job_log_dir, f"RUN_{run_timestamp}.out")} 2>&1
            '''
        ]

        # Ensure the command is printed for debugging
        print(f"Running command: {' '.join(conda_command)}")

        # Execute the command and write output to the log file
        with open(os.path.join(job_log_dir, f"RUN_{run_timestamp}.out"), 'w') as log_file:
            subprocess.run(conda_command, stdout=log_file, stderr=log_file)
        
        # # Wait for 5-10 seconds before the second run
        # time.sleep(random.randint(5, 10))
        
        # # Second torchrun execution
        # with open(os.path.join(job_log_dir, f"RUN_{run_timestamp}_2.out"), 'w') as log_file:
        #     subprocess.run(conda_command, stdout=log_file, stderr=log_file)
    
    # If neither Docker nor Conda are available
    else:
        raise EnvironmentError("Error: Docker and Conda are both unavailable. Cannot proceed.")

def run_trial(trial_num, config_file_path, gpu_ids, run_timestamp, docker_image=None):
    """Function to run a trial"""
    print(f"Running trial {trial_num} with GPU(s) {gpu_ids} and timestamp {run_timestamp}...")

    # Check if Docker is available and run accordingly
    if docker_image and is_docker_available():
        print(f"Running trial {trial_num} with Docker...")
        try:
            run_job(config_file_path=config_file_path, docker_image=docker_image, gpu_ids=gpu_ids, run_timestamp=run_timestamp)
        except subprocess.CalledProcessError as e:
            print(f"Error during trial {trial_num} execution: {e}")
    elif is_conda_available():
        print(f"Running trial {trial_num} with Conda...")
        try:
            run_job(config_file_path=config_file_path, docker_image=None, gpu_ids=gpu_ids, run_timestamp=run_timestamp)
        except subprocess.CalledProcessError as e:
            print(f"Error during trial {trial_num} execution: {e}")
    else:
        print("Error: Neither Docker nor Conda found on your system.")

def main():
    parser = argparse.ArgumentParser(description="Run parallel optimization trials.")
    
    # Adding arguments
    parser.add_argument('--config_file_path', type=str, help='Path to the config yaml to run (e.g., ./app_config.yaml)')
    parser.add_argument('--num_trials', type=int, help='Number of trials to run')
    parser.add_argument('--gpu_ids', type=str, help="GPU IDs to use (e.g., '0,1' or 'all')")
    if is_docker_available():
        parser.add_argument('--docker_image', type=str, help="Docker image to use (optional)", default=None)
    
    # Parse the arguments
    args = parser.parse_args()

    # Validate the number of trials
    if args.num_trials < 1:
        print("Error: The number of trials must be at least 1.")
        return

    # Ensure GPU IDs format is correct
    if args.gpu_ids != 'all' and not all(gpu.isdigit() for gpu in args.gpu_ids.split(',')):
        print("Error: Invalid GPU IDs. Please provide a comma-separated list (e.g., 0,1 or 'all').")
        return

    # Check if the script exists and is executable
    if not os.path.exists(args.config_file_path):
        print(f"Error: The specified file does not exist: {args.config_file_path}")
        return

    # Run the trials
    for i in range(1, args.num_trials + 1):
        run_timestamp = get_current_timestamp()  # Get the current timestamp in compact format
        time.sleep(5)  # Interval between trials
        if is_docker_available():
            run_trial(i, args.config_file_path, args.gpu_ids, run_timestamp, args.docker_image)
        else:
            run_trial(i, args.config_file_path, args.gpu_ids, run_timestamp)

if __name__ == "__main__":
    main()
