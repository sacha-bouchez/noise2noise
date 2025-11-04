import os
import mlflow
from train.trainer import Noise2NoiseTrainer

if __name__ == "__main__":

    trainer = Noise2NoiseTrainer(
        binsimu=None,
        dest_path=f"{os.getenv('WORKSPACE')}/data/noise2noise",
        dataset_size=32,
        n_epochs=10,
        batch_size=8,
        metrics_configs=[
            ['PSNR', {'max_val': 1.0}],
            ['Mean', {'name': 'loss'}],
            ['SSIM', {'max_val': 1.0}],
        ],
        shuffle=True,
        image_size=[256, 256],
        voxel_size=[2, 2, 2],
        learning_rate=1e-3,
        seed=42
    )

    # setup mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    #
    # create experiment if not exists
    experiment = mlflow.get_experiment_by_name("Noise2Noise_Poisson_2DPET")
    if experiment is None:
        mlflow.create_experiment("Noise2Noise_Poisson_2DPET")
        experiment = mlflow.get_experiment_by_name("Noise2Noise_Poisson_2DPET")
    mlflow.set_experiment("Noise2Noise_Poisson_2DPET")
    #
    # Set run name. The same config of trainer is used for a single run.
    run_name = f"run_{trainer._id[:8]}"
    # if run exists, get its run_id
    run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"tags.run_name='{run_name}'")
    if not run.empty:
        run_id = run.iloc[0].name
    else:
        run_id = None
    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        # log parameters
        for key, value in trainer.__dict__.items():
            # check if value is json serializable
            try:
                mlflow.log_param(key, value)
            except:
                pass
        # start training
        trainer.fit()