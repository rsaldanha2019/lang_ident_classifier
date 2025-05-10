import sys
import time
import subprocess
import ipywidgets as widgets
from IPython.display import display

def get_user_input(prompt, default=None):
    """Prompt the user for input, using different methods based on the environment."""
    if not sys.stdin.isatty():  # Checking if we are in a non-interactive environment (like Kaggle)
        print(f"{prompt} (default: {default})")
        if default:
            return default  # If no user input stream is available, return default
        return ""  # Empty input when no user input is available
    # For terminal or interactive environments, use regular input
    user_input = input(f"{prompt} (default: {default}): ")
    return user_input if user_input else default

def create_input_widget(prompt, default_value):
    """Create an interactive input widget using ipywidgets for Jupyter environments."""
    widget = widgets.Text(
        value=default_value,
        description=prompt,
        disabled=False
    )
    display(widget)
    return widget

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
    # Interactive prompts (use `input()` for non-Kaggle and widgets for Kaggle)
    if not sys.stdin.isatty():  # Running in non-interactive environment like Kaggle
        script_path = create_input_widget("Enter the path to the script (e.g., ./standalone_job_optim_test.sh)", "./standalone_job_optim_test.sh").value
        gpu_ids = create_input_widget("Enter the GPU IDs (e.g., 0,1 for multiple GPUs or 'all' for all GPUs)", "all").value
        docker_image = create_input_widget("Enter the Docker image (e.g., smallworld2020/lang_classifier:v3)", "smallworld2020/lang_classifier:v3").value if is_docker_available() else None
        num_trials = 1  # You can define default trials here
    else:  # Running in terminal, use input() for prompts
        script_path = get_user_input("Enter the path to the script (e.g., ./standalone_job_optim_test.sh)", "./standalone_job_optim_test.sh")
        gpu_ids = get_user_input("Enter the GPU IDs (e.g., 0,1 for multiple GPUs or 'all' for all GPUs)", "all")
        docker_image = get_user_input("Enter the Docker image (e.g., smallworld2020/lang_classifier:v3)", "smallworld2020/lang_classifier:v3") if is_docker_available() else None
        num_trials = int(get_user_input("Enter the number of trials", "1"))

    # Validate GPU IDs
    if gpu_ids != "all" and not all(x.isdigit() for x in gpu_ids.split(',')):
        print("Invalid GPU IDs format. Please provide a comma-separated list (e.g., 0,1 or 'all').")
        exit(1)

    # Generate a run timestamp in seconds with float precision
    run_timestamp = str(time.time())

    # Run the specified number of parallel trials with the same timestamp in the background
    for i in range(1, num_trials + 1):
        print(f"Starting trial {i} with timestamp {run_timestamp}.")
        
        run_trial(script_path, gpu_ids, run_timestamp, docker_image)
        time.sleep(30)  # Delay between trials

    print("All trials have started in the background.")
    print("Waiting for trials to finish...")

    # Wait for all subprocesses to finish
    while len(subprocess._active) > 0:
        time.sleep(1)

    print("All trials have finished.")

if __name__ == "__main__":
    main()

