import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_loader import SinogramGenerator

try:
    # prefer skimage if available for robust implementations
    from skimage.metrics import peak_signal_noise_ratio as sk_peak_signal_noise_ratio
    from skimage.metrics import structural_similarity as sk_structural_similarity
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

from model.unet_model import UNet

class Noise2NoiseTrainer:

    def __init__(self, binsimu=None,
                       dest_path='./',
                       dataset_size=10000,
                       n_epochs=200,
                       batch_size=64,
                       shuffle=False,
                       image_size=(256,256),
                       voxel_size=(2,2,2),
                       learning_rate=1e-3,
                       seed=42):
        if binsimu is None:
            self.binsimu = os.path.join(os.getenv("WORKSPACE"), "simulator", "bin")
        else:
            self.binsimu = binsimu
        self.dest_path = dest_path
        self.dataset_size = dataset_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.learning_rate = learning_rate
        self.seed = seed

        # create dataset / loader / model / optimizer
        self.loader = self.create_data_loader()
        self.model = self.create_model()
        self.signature = self.get_signature()
        self.optimizer = self.get_optimizer(learning_rate=self.learning_rate)
        self.objective = self.get_objective()

    def create_data_loader(self):

        self.dataset = SinogramGenerator(self.binsimu,
                                         dest_path=self.dest_path,
                                         length=self.dataset_size,
                                         image_size=self.image_size,
                                         voxel_size=self.voxel_size,
                                         seed=self.seed)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return loader

    def get_signature(self):
        sample = self.dataset.__getitem__(0)[0]
        signature = (1, ) + tuple(sample.shape)
        return signature

    def create_model(self):
        # model expects channel-first inputs: we'll add channel dimension when calling
        model = UNet(n_channels=1, n_classes=1)
        return model

    def get_optimizer(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    def get_objective(self):
        objective = torch.nn.MSELoss()
        return objective

    def fit(self):

        for epoch in range(self.n_epochs):
            for batch_idx, (noisy_img1, noisy_img2, clean_img) in enumerate(self.loader):
                # ensure channel dimension: dataset returns (B, H, W)
                if noisy_img1.dim() == 3:
                    noisy_img1 = noisy_img1.unsqueeze(1)
                if noisy_img2.dim() == 3:
                    noisy_img2 = noisy_img2.unsqueeze(1)
                if clean_img.dim() == 3:
                    clean_img = clean_img.unsqueeze(1)

                noisy_img1 = noisy_img1.float()
                noisy_img2 = noisy_img2.float()
                clean_img = clean_img.float()

                outputs = self.model(noisy_img1)

                loss = self.objective(outputs, noisy_img2)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging
                # TODO MLflow integration
                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.n_epochs}], Step [{batch_idx+1}/{len(self.loader)}], '
                          f'Loss: {loss.item():.4f}')