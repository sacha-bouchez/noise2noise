import os

from train.trainer import Noise2NoiseTrainer

if __name__ == "__main__":

    trainer = Noise2NoiseTrainer(
        binsimu=None,
        dest_path=f"{os.getenv('WORKSPACE')}/data/noise2noise",
        dataset_size=1000,
        n_epochs=10,
        batch_size=16,
        shuffle=True,
        image_size=(256, 256),
        voxel_size=(2, 2, 2),
        learning_rate=1e-3,
        seed=42
    )
    trainer.fit()