import sys
import time
import subprocess
import ipywidgets as widgets
from IPython.display import display
import os

# Function to check if Docker is available (without large error trace)
def is_docker_available():
    try:
        subprocess.run(["docker", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Function to check if Conda is available (without large error trace)
def is_conda_available():
    try:
        subprocess.run(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

# Function to handle input in interactive environments (minimal prompt)
def get_user_input(prompt, default=None):
    """Prompt the user for input interactively (supports both terminal and Kaggle environments)."""
    if not sys.stdin.isatty():  # Check if running in a non-interactive environment (like Kaggle)
        # If running in a Kaggle notebook, use widgets
        return create_input_widget(prompt, default).value
    else:
        # If running in a terminal, use standard input() (silent with default)
        user_input = input(f"{prompt} (default: {default}): ").strip()
        return user_input if user_input else default

# Function to create an interactive widget in Kaggle/Jupyter
def create_input_widget(prompt, default_value):
    widget = widgets.Text(
        value=default_value,
        description=prompt,
        disabled=False
    )
    display(widget)
    return widget

# Function to validate GPU IDs
def validate_gpu_ids(gpu_ids):
    if gpu_ids != "all" and not all(x.isdigit() for x in gpu_ids.split(',')):
        print("Error: Invalid GPU IDs format. Please provide a comma-separated list (e.g., 0,1 or 'all').")
        return False
    return True

# Function to run a trial (using Docker or Conda)
def run_trial(script_path, gpu_ids, run_timestamp, docker_image=None):
    try:
        if docker_image:
            subprocess.Popen([
                "docker", "run", "--rm", "--runtime=nvidia", "--gpus", gpu_ids,
                "-v", os.getcwd() + ":" + os.getcwd(),
                "-w", "/app", "--ipc=host", docker_image,
                script_path, "--gpu-ids", gpu_ids, "--run_timestamp", run_timestamp
            ])
        elif is_conda_available():
            subprocess.Popen([
                "conda", "run", "-n", "lang_ident_classifier", "python", script_path,
                "--gpu-ids", gpu_ids, "--run_timestamp", run_timestamp
            ])
        else:
            print("Error: Neither Docker nor Conda was found on your system.")
            exit(1)
    except Exception as e:
        print(f"Error during trial execution: {str(e)}")

# Main function for the script
def main():
    # Silent prompts for user input with default values
    script_path = get_user_input("Enter the path to the script (e.g., ./standalone_job_optim_test.sh)", "./standalone_job_optim_test.sh")
    gpu_ids = get_user_input("Enter the GPU IDs (e.g., 0,1 for multiple GPUs or 'all' for all GPUs)", "all")

    # Validate GPU IDs format silently
    if not validate_gpu_ids(gpu_ids):
        return

    docker_image = None
    if is_docker_available():
        docker_image = get_user_input("Enter the Docker image (e.g., smallworld2020/lang_classifier:v3)", "smallworld2020/lang_classifier:v3")

    # Generate a run timestamp
    run_timestamp = time.time()

    # Ask for the number of trials silently
    num_trials = int(get_user_input("Enter the number of trials to run", "1"))

    # Run the trials silently
    for i in range(num_trials):
        try:
            run_trial(script_path, gpu_ids, run_timestamp, docker_image)
            time.sleep(2)  # Give some time between trials (optional)
        except Exception as e:
            print(f"Error during trial {i+1}: {str(e)}")
            continue

    print("All trials are running.")

if __name__ == "__main__":
    main()

