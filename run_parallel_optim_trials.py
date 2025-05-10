import os
import subprocess
import time

def get_user_input(prompt, default=None):
    """Function to get user input with an optional default value."""
    if default:
        user_input = input(f"{prompt} (default: {default}): ")
        return user_input if user_input else default
    return input(f"{prompt}: ")

def is_docker_available():
    """Check if Docker is available on the system."""
    return subprocess.run(["command", "-v", "docker"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0

def is_conda_available():
    """Check if Conda is available on the system."""
    return subprocess.run(["command", "-v", "conda"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0

def run_trial(script_path, gpu_ids, run_timestamp, docker_image=None):
    """Run a single trial using Docker or Conda."""
    if docker_image:
        print(f"Running trial with Docker...")
        # Running the script with Docker
        subprocess.Popen([
            "docker", "run", "--rm", "--runtime=nvidia", "--gpus", gpu_ids,
            "-v", os.getcwd() + ":" + os.getcwd(),
            "-w", "/app", "--ipc=host", docker_image,
            script_path, "--gpu-ids", gpu_ids, "--run_timestamp", run_timestamp
        ])
    elif is_conda_available():
        print(f"Running trial with Conda...")
        # Running the script with Conda
        subprocess.Popen([
            "conda", "run", "-n", "lang_ident_classifier", "python", script_path,
            "--gpu-ids", gpu_ids, "--run_timestamp", run_timestamp
        ])
    else:
        print("Error: Neither Docker nor Conda was found on your system.")
        exit(1)

def main():
    # Interval between trials in seconds
    interval_between_trials = 30

    # Interactive user inputs
    script_path = get_user_input("Enter the path to the script (e.g., ./standalone_job_optim_test.sh)", "./standalone_job_optim_test.sh")
    
    # Check if the script exists
    if not os.path.isfile(script_path):
        print(f"Error: The specified script '{script_path}' does not exist.")
        exit(1)

    num_trials = int(get_user_input("Enter the number of parallel trials (1 to n)", "1"))
    
    if num_trials < 1:
        print("Invalid number of trials. It should be greater than 0.")
        exit(1)

    gpu_ids = get_user_input("Enter the GPU IDs (e.g., 0,1 for multiple GPUs or 'all' for all GPUs)", "all")
    
    # Validate GPU IDs (ensure it's a valid format)
    if gpu_ids != "all" and not all(x.isdigit() for x in gpu_ids.split(',')):
        print("Invalid GPU IDs format. Please provide a comma-separated list (e.g., 0,1 or 'all').")
        exit(1)

    # Generate a run timestamp in seconds with float precision
    run_timestamp = str(time.time())

    # Ask for Docker image if Docker is available
    docker_image = None
    if is_docker_available():
        docker_image = get_user_input("Enter the Docker image (e.g., smallworld2020/lang_classifier:v3)", "smallworld2020/lang_classifier:v3")

    # Run the specified number of parallel trials with the same timestamp in the background
    for i in range(1, num_trials + 1):
        print(f"Starting trial {i} with timestamp {run_timestamp}.")
        
        run_trial(script_path, gpu_ids, run_timestamp, docker_image)
        time.sleep(interval_between_trials)

    print("All trials have started in the background.")
    print("Waiting for trials to finish...")

    # We use wait for all subprocesses to complete (since we're using subprocess.Popen)
    while len(subprocess._active) > 0:
        time.sleep(1)

    print("All trials have finished.")

if __name__ == "__main__":
    main()

