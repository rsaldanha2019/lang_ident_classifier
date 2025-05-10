import subprocess
import time
import sys

try:
    import ipywidgets as widgets
    from IPython.display import display
    # Flag indicating whether widgets are available (for notebooks)
    IS_NOTEBOOK = True
except ImportError:
    IS_NOTEBOOK = False


def get_user_input(prompt, default_value=None):
    """Helper function to get user input in both notebook (widgets) and terminal (input())."""
    if IS_NOTEBOOK:
        # Use widgets for input if in a notebook (Kaggle, Colab, Jupyter)
        text_widget = widgets.Text(
            value=default_value,
            description=prompt,
            placeholder=f"Enter {prompt}"
        )
        display(text_widget)

        # Make sure widget is interactive and waits for input from the user
        while not text_widget.value:
            time.sleep(0.1)  # Wait for the input to be entered by the user
        return text_widget.value
    else:
        # Use input() for terminal
        user_input = input(f"{prompt} (default: {default_value}): ")
        return user_input if user_input else default_value


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

        # Check if the script is a Python file or a shell script
        if script_path.endswith('.sh'):
            # Run the shell script with bash if it's a shell script
            try:
                subprocess.run(["bash", script_path, "--gpu-ids", gpu_ids, "--run_timestamp", run_timestamp], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during trial {trial_num} execution: {e}")
        else:
            # If it's a Python file, run it using Conda
            try:
                subprocess.run([
                    "conda", "run", "-n", "lang_ident_classifier", "python", script_path,
                    "--gpu-ids", gpu_ids, "--run_timestamp", run_timestamp
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during trial {trial_num} execution: {e}")
    else:
        print("Error: Neither Docker nor Conda found on your system.")


def run_trials(script_path, gpu_ids, num_trials):
    """Run the trials after collecting input"""
    # Generate a timestamp for run
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())

    # Running trials
    for i in range(1, num_trials + 1):
        # Start a new trial after a short interval
        time.sleep(5)

        # Run the trial
        run_trial(i, script_path, gpu_ids, run_timestamp)

    print("All trials completed!")


def main():
    """Main function to handle the user input and execute trials"""
    
    # Get input values using the appropriate method for the environment (widgets or input)
    script_path = get_user_input("Enter the path to the script (e.g., ./standalone_job_optim_test.sh)", "./standalone_job_optim_test.sh")
    num_trials = get_user_input("Enter the number of trials to run", "1")

    try:
        num_trials = int(num_trials)
        if num_trials < 1:
            raise ValueError("The number of trials must be at least 1.")
    except ValueError as e:
        print(f"Error: {e}")
        return

    gpu_ids = get_user_input("Enter the GPU IDs (e.g., 0,1 for multiple GPUs or 'all' for all GPUs)", "all")

    # Check if Docker is available
    docker_image = None
    if is_docker_available():
        docker_image = get_user_input("Enter the Docker image (e.g., smallworld2020/lang_classifier:v3)", "smallworld2020/lang_classifier:v3")
    else:
        print("Docker not available. Will use Conda or fallback to Python environment.")

    # Run trials with the collected user input
    run_trials(script_path, gpu_ids, num_trials)


if __name__ == "__main__":
    main()

