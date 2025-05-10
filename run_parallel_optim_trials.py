import ipywidgets as widgets
from IPython.display import display
import subprocess
import time

def get_user_input(prompt, default_value=None):
    """Helper function to create an interactive input widget in Jupyter/Colab/Kaggle"""
    widget = widgets.Text(
        value=default_value,
        placeholder=f'Enter {prompt}',
        description=prompt,
        disabled=False
    )
    display(widget)
    return widget

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
    if docker_image:
        print(f"Running trial {trial_num} with Docker...")
        try:
            subprocess.run([
                "docker", "run", "--rm", "--runtime=nvidia", "--gpus", gpu_ids,
                "-v", f"{script_path}:{script_path}",
                "-w", "/app", "--ipc=host", docker_image,
                "python", script_path, "--gpu-ids", gpu_ids, "--run_timestamp", run_timestamp
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during trial {trial_num} execution: {e}")
    elif is_conda_available():
        print(f"Running trial {trial_num} with Conda...")
        try:
            subprocess.run([
                "conda", "run", "-n", "lang_ident_classifier", "python", script_path,
                "--gpu-ids", gpu_ids, "--run_timestamp", run_timestamp
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during trial {trial_num} execution: {e}")
    else:
        print("Error: Neither Docker nor Conda found on your system.")


def main():
    # Prompt user for script path using a widget
    script_path_widget = get_user_input("Enter the path to the script (e.g., ./standalone_job_optim_test.sh)", "./standalone_job_optim_test.sh")
    
    # Wait until the user inputs the script path
    while not script_path_widget.value:
        time.sleep(1)  # Wait for input

    script_path = script_path_widget.value

    # Prompt user for number of trials
    num_trials_widget = get_user_input("Enter the number of trials to run", "1")
    
    while not num_trials_widget.value:
        time.sleep(1)

    try:
        num_trials = int(num_trials_widget.value)
        if num_trials < 1:
            raise ValueError("The number of trials must be at least 1.")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Prompt user for GPU IDs
    gpu_ids_widget = get_user_input("Enter the GPU IDs (e.g., 0,1 for multiple GPUs or 'all' for all GPUs)", "all")
    
    while not gpu_ids_widget.value:
        time.sleep(1)

    gpu_ids = gpu_ids_widget.value

    # Check if Docker is available
    docker_image = None
    if is_docker_available():
        docker_image_widget = get_user_input("Enter the Docker image (e.g., smallworld2020/lang_classifier:v3)", "smallworld2020/lang_classifier:v3")
        
        while not docker_image_widget.value:
            time.sleep(1)

        docker_image = docker_image_widget.value
    else:
        print("Docker not available. Will use Conda or fallback to Python environment.")

    # Generate a timestamp for run
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())

    # Running trials
    for i in range(1, num_trials + 1):
        # Start a new trial after a short interval
        time.sleep(5)

        # Run the trial
        run_trial(i, script_path, gpu_ids, run_timestamp, docker_image)

    print("All trials completed!")


if __name__ == "__main__":
    main()

