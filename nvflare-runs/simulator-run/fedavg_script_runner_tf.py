from src.tf_net import TFNet
from nvflare.app_opt.tf.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "src/hello-tf_fl.py"

    # Define input shape and number of classes based on your dataset
    input_shape = (None, 474)  # Replace with the correct number of features
    num_classes = 18  # Replace with the correct number of classes in your dataset

    # Initialize the job with the custom TFNet
    job = FedAvgJob(
        name="hello-tf_fedavg",
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=TFNet(input_shape=input_shape, num_classes=num_classes),
    )

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script,
            script_args="",  # Add necessary arguments if required
            framework=FrameworkType.TENSORFLOW,
        )
        job.to(executor, f"site-{i+1}")

    # Run the simulation
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
