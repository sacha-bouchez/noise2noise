import os
import mlflow
import datetime
from train.trainer import Noise2NoiseTrainer

if __name__ == "__main__":

    trainer = Noise2NoiseTrainer(
        binsimu=None,
        dest_path=f"{os.getenv('WORKSPACE')}/data/noise2noise",
        dataset_size=20000,
        n_epochs=200,
        batch_size=8,
        metrics_configs=[
            ['PSNR', {'max_val': 1.0}],
            ['Mean', {'name': 'loss'}],
            ['SSIM', {'max_val': 1.0}],
        ],
        shuffle=True,
        image_size=[256, 256],
        voxel_size=[2, 2, 2],
        learning_rate=1e-4,
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
    # find if there is a run to resume among not finished ones
    # the run shall have the same hash as the current trainer
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params._id = '{trainer._id}' and attributes.status != 'FINISHED'",
        order_by=["start_time DESC"]
    )
    run_id = None
    run_name = f"Noise2Noise_2DPET_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if len(runs) > 0:
        for _, run in runs.iterrows():
            params_id = run['params._id']
            if params_id == trainer._id:
                run_id = run['run_id']
                run_name = None
                print(f"Resuming run {run_id} ...")
                break
    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        # log parameters
        for key, value in trainer.__dict__.items():
            # check if value is json serializable
            try:
                mlflow.log_param(key, value)
            except:
                pass
        # resume model and optimizer if possible
        trainer.load_model_and_optimizer(path="reboot_model")
        # start training
        trainer.fit()