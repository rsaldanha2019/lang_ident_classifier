import subprocess
import time
import argparse
import sys

def is_docker_available():
    """Check if Docker is available on the system"""
    try:
        subprocess.run(["docker", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False

def is_script_executable(script_path):
    """Check if the provided script exists and is executable"""
    try:
        subprocess.run(["chmod", "+x", script_path], check=True)  # Make it executable if not
        return True
    except subprocess.CalledProcessError:
        print(f"The specified script {script_path} does not exist or is not executable.")
        return False

def validate_gpu_ids(gpu_ids):
    """Validate GPU IDs format"""
    if gpu_ids == "all":
        return True
    elif all(gpu_id.isdigit() for gpu_id in gpu_ids.split(",")):
        return True
    return False

def run_trial(script_path, gpu_ids, run_timestamp, trial_num, docker_image=None):
    """Run a trial in the background with or without Docker"""
    print(f"Starting trial {trial_num} with timestamp {run_timestamp}...")

    # Prepare the command based on whether Docker is available
    command = [script_path, "--gpu-ids", gpu_ids, "--run_timestamp", run_timestamp]
    
    if docker_image:
        # Run the trial with Docker
        print(f"Running with Docker image: {docker_image}")
        command = ["docker", "run", "--rm", "--runtime=nvidia", "--gpus", gpu_ids,
                   "-v", f"{script_path}:{script_path}", "-w", "/app", docker_image] + command

    # Run the script (either with Docker or not)
    process = subprocess.Popen(command)
    print(f"Started trial {trial_num}. Background PID: {process.pid}")

    return process

def parse_args():
    """Parse command-line arguments using argparse"""
    parser = argparse.ArgumentParser(description="Run parallel trials for script execution.")
    
    # Command-line arguments (all are required)
    parser.add_argument("script_path", help="Path to the script (e.g., ./standalone.sh)", type=str)
    parser.add_argument("num_trials", type=int, help="Number of parallel trials to run (1 to n)")
    parser.add_argument("gpu_ids", help="Comma-separated list of GPU IDs (e.g., 0,1 or 'all')", type=str)
    
    # Conditional argument for Docker image
    if is_docker_available():
        parser.add_argument("docker_image", help="Docker image (e.g., smallworld2020/lang_classifier:v3)", type=str)

    return parser.parse_args()

def main():
    """Main function to handle the user input and execute trials"""
    
    # Parse command-line arguments
    args = parse_args()

    # Validate the script
    if not is_script_executable(args.script_path):
        sys.exit(1)
    
    # Validate the number of trials
    if args.num_trials < 1:
        print("Invalid number of trials. Please enter a positive integer greater than 0.")
        sys.exit(1)

    # Validate GPU IDs
    if not validate_gpu_ids(args.gpu_ids):
        print("Invalid GPU IDs. Please provide a comma-separated list (e.g., 0,1 or 'all').")
        sys.exit(1)

    # Generate a run timestamp in seconds with float precision
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    print(f"Run Timestamp: {run_timestamp}")

    # Start trials with the given number of parallel executions
    processes = []
    for i in range(1, args.num_trials + 1):
        time.sleep(30)  # Interval between trials
        docker_image = args.docker_image if is_docker_available() else None
        process = run_trial(args.script_path, args.gpu_ids, run_timestamp, i, docker_image)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()

    print("All trials have been started successfully in the background.")

if __name__ == "__main__":
    main()

