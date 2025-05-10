import argparse
import subprocess
import time
import os

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

def run_trial(trial_num, script_path, gpu_ids, run_timestamp, docker_image=None):
    """Function to run a trial"""
    print(f"Running trial {trial_num} with GPU(s) {gpu_ids} and timestamp {run_timestamp}...")

    # Check if Docker is available and run accordingly
    if docker_image and is_docker_available():
        print(f"Running trial {trial_num} with Docker...")
        try:
            subprocess.run([
                "docker", "run", "--rm", "--runtime=nvidia", "--gpus", gpu_ids,
                "-v", f"{script_path}:{script_path}",
                "-w", "/app", "--ipc=host", docker_image,
                "bash", script_path, "--gpu_ids", gpu_ids, "--run_timestamp", run_timestamp
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during trial {trial_num} execution: {e}")
    elif is_conda_available():
        print(f"Running trial {trial_num} with Conda...")
        try:
            subprocess.run([
                "conda", "run", "-n", "lang_ident_classifier", "bash", script_path,
                "--gpu_ids", gpu_ids, "--run_timestamp", run_timestamp
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during trial {trial_num} execution: {e}")
    else:
        print("Error: Neither Docker nor Conda found on your system.")

def main():
    parser = argparse.ArgumentParser(description="Run parallel optimization trials.")
    
    # Adding arguments
    parser.add_argument('--script_path', type=str, help='Path to the script to run (e.g., ./standalone_job_optim_test.sh)')
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
    if not os.path.exists(args.script_path):
        print(f"Error: The specified script does not exist: {args.script_path}")
        return

    # Generate a run timestamp for trial
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())

    # If Docker is available, and user hasn't passed docker_image, prompt for it
    if is_docker_available() and not args.docker_image:
        args.docker_image = input("Enter the Docker image (e.g., smallworld2020/lang_classifier:v3): ")

    # Run the trials
    for i in range(1, args.num_trials + 1):
        time.sleep(5)  # Interval between trials
        if is_docker_available():
            run_trial(i, args.script_path, args.gpu_ids, run_timestamp, args.docker_image)
        else:
            run_trial(i, args.script_path, args.gpu_ids, run_timestamp, None)
    print("All trials completed!")

if __name__ == "__main__":
    main()

