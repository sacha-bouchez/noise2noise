import os
import mlflow
import datetime
import torchsummary
from train.trainer import Noise2NoiseTrainer

if __name__ == "__main__":

    trainer = Noise2NoiseTrainer(
        dest_path=f"{os.getenv('WORKSPACE')}/data/noise2noise",
        dataset_train_size=2048,
        dataset_val_size=512,
        val_freq=1,
        n_epochs=25,
        batch_size=4,
        metrics_configs=[
            ['PSNR', {'max_val': 1.0}],
        ],
        shuffle=True,
        image_size=[160, 160],
        voxel_size=[2, 2, 2],
        n_angles=300,
        scanner_radius=300,
        nb_counts=1e6,
        learning_rate=1e-4,
        L2_weight=0.0,
        objective_type='poisson',
        num_workers=10,
        conv_layer_type='SinogramConv2d',
        n_levels=3,
        global_conv=64,
        seed=42
    )

    # setup mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    #
    # create experiment if not exists
    experiment_name = "Noise2Noise_Poisson_2DPET_v2_toy"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)
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
        trainer.load_model_and_optimizer(artifact_path="reboot_model")
        #
        torchsummary.summary(trainer.model, input_size=(1, trainer.image_size[0], trainer.image_size[1]))
        # start training
        trainer.fit()