import os
import numpy as np
import mlflow
import torch
import hashlib, json
import torch.nn.functional as F
from torch.utils.data import DataLoader

from noise2noise.data.data_loader import SinogramGenerator

from pytorcher.trainer import PytorchTrainer
from pytorcher.models import UNet
from pytorcher.utils.processing import normalize_batch

class Noise2NoiseTrainer(PytorchTrainer):

    def __init__(self, binsimu=None,
                       dest_path='./',
                       dataset_train_size=10000,
                       dataset_val_size=500,
                       val_freq=1,
                       n_epochs=200,
                       batch_size=64,
                       metrics_configs=[],
                       shuffle=False,
                       image_size=(256,256),
                       voxel_size=(2,2,2),
                       learning_rate=1e-3,
                       conv_layer_type='standard',
                       num_workers=10,
                       L2_weight=1e-4,
                       objective_type='MSE',
                       seed=42):
        if binsimu is None:
            self.binsimu = os.path.join(os.getenv("WORKSPACE"), "simulator", "bin")
        else:
            self.binsimu = binsimu
        self.dest_path = dest_path
        self.dataset_train_size = dataset_train_size
        self.dataset_val_size = dataset_val_size
        self.val_freq = val_freq
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.metrics_configs = metrics_configs
        self.shuffle = shuffle
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.learning_rate = learning_rate
        self.objective_type = objective_type
        self.seed = seed

        self.conv_layer_type = conv_layer_type

        self.num_workers = num_workers
        self.L2_weight = L2_weight

        self._id = hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()

        if f'loss_{self.objective_type}' not in metrics_configs:
            metrics_configs.append(['Mean', {'name': f'loss_{self.objective_type}'}])

        super(Noise2NoiseTrainer, self).__init__(metrics=metrics_configs)


    def create_data_loader(self):

        self.dataset_train = SinogramGenerator(self.binsimu,
                                         dest_path=os.path.join(self.dest_path, 'train'),
                                         length=self.dataset_train_size,
                                         image_size=self.image_size,
                                         voxel_size=self.voxel_size,
                                         seed=self.seed)
        loader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        self.dataset_val_seed = int(1e5) # Seed is fixed to have consistent validation sets. Changing image size or voxel size will give different results.
        self.dataset_val = SinogramGenerator(self.binsimu,
                                         dest_path=os.path.join(self.dest_path, 'val'),
                                         length=self.dataset_val_size,
                                         image_size=self.image_size,
                                         voxel_size=self.voxel_size,
                                         seed=self.dataset_val_seed)
        loader_val = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return loader_train, loader_val

    def get_optimizer(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.L2_weight)
        return optimizer

    def get_signature(self):
        sample = self.dataset_train.__getitem__(0)[0]
        signature = (1, ) + tuple(sample.shape)
        return signature

    def create_model(self):
        # model expects channel-first inputs: we'll add channel dimension when calling
        model = UNet(n_channels=1, n_classes=1, bilinear=True, layer_type=self.conv_layer_type, normalize_input=True)
        model = model.to(self.device)
        return model

    def load_model_and_optimizer(self, path="reboot_model"):
        #
        try:
            mlflow.artifacts.download_artifacts(artifact_path=path, dst_path="/tmp/reboot_model", run_id=mlflow.active_run().info.run_id)
        except mlflow.exceptions.MlflowException:
            print("No reboot model found in mlflow artifacts.")
            return
        #
        if os.path.exists("/tmp/reboot_model/reboot_model.pth") and os.path.exists("/tmp/reboot_model/epoch.txt") and os.path.exists("/tmp/reboot_model/optimizer.pth"):
            self.model.load_state_dict(torch.load("/tmp/reboot_model/reboot_model.pth"))
            with open("/tmp/reboot_model/epoch.txt", "r") as f:
                self.initial_epoch = int(f.read()) + 1
            self.optimizer.load_state_dict(torch.load("/tmp/reboot_model/optimizer.pth"))
            print(f"Rebooted model from epoch {self.initial_epoch}")
        else:
            print("No reboot model found, training from scratch.")

    def get_objective(self):
        type = self.objective_type.lower()
        assert type in ['mse', 'poisson'], "Objective type not recognized. Supported types: 'MSE', 'Poisson'."
        if type == 'mse':
            objective = torch.nn.MSELoss()
        elif type == 'poisson':
            objective = torch.nn.PoissonNLLLoss(log_input=False, full=False, reduction='mean')
        return objective

    def log_and_reset_metrics(self, epoch):
        print(f'End of Epoch {epoch+1}, metrics: ')
        for metric in self.metrics:
            print(f'{metric.name}: {metric.result():.4f}')

            # Log metrics to MLflow
            mlflow.log_metric(metric.name, metric.result(), step=epoch)

            # Reset metric
            metric.reset_states()

    def fit(self):

        monitored_metrics = [ 'val_psnr' ]

        for epoch in range(self.initial_epoch, self.n_epochs):

            m_dict = {}

            # TRAIN
            for batch_idx, (noisy_img1, noisy_img2, clean_img) in enumerate(self.loader_train):

                # ensure channel dimension: dataset returns (B, H, W)
                if noisy_img1.dim() == 3:
                    noisy_img1 = noisy_img1.unsqueeze(1)
                if noisy_img2.dim() == 3:
                    noisy_img2 = noisy_img2.unsqueeze(1)
                if clean_img.dim() == 3:
                    clean_img = clean_img.unsqueeze(1)
                
                # Move data to device
                noisy_img1 = noisy_img1.to(self.device)
                noisy_img2 = noisy_img2.to(self.device)
                clean_img = clean_img.to(self.device)

                noisy_img1 = noisy_img1.float()
                noisy_img2 = noisy_img2.float()
                clean_img = clean_img.float()

                def inference_pair_and_loss(noisy_img_a, noisy_img_b):
                    """ Perform inference with a as input and b as target, and compute loss """
                    output = self.model(noisy_img_a)
                    if self.objective_type.lower() == 'poisson':
                        loss = self.objective(output, noisy_img_b)
                    elif self.objective_type.lower() == 'mse':
                        loss = self.objective(normalize_batch(output), normalize_batch(noisy_img_b))
                    self.update_metrics(loss, normalize_batch(clean_img), normalize_batch(output))
                    # free up memory
                    del output
                    torch.cuda.empty_cache()
                    return loss

                loss_1 = inference_pair_and_loss(noisy_img1, noisy_img2)
                loss_2 = inference_pair_and_loss(noisy_img2, noisy_img1)

                loss = 1/2 * ( loss_1 + loss_2 ) # NOTE the .5 factor is used for learning rate scaling consistency with standard supervised training
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Train,  Step [{batch_idx+1}/{len(self.loader_train)}]')

            for metric in self.metrics:
                print(f'{metric.name}: {metric.result():.4f}')
                m_dict.update({metric.name: metric.result()})
                # reset state
                metric.reset_states()

            # VALIDATION
            if (epoch + 1) % self.val_freq == 0:
                # run model in evaluation mode and avoid building computation graph
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (noisy_img, _, clean_img) in enumerate(self.loader_val):
                        noisy_img = noisy_img.to(self.device)
                        clean_img = clean_img.to(self.device)

                        if noisy_img.dim() == 3:
                            noisy_img = noisy_img.unsqueeze(1)
                        if clean_img.dim() == 3:
                            clean_img = clean_img.unsqueeze(1)

                        noisy_img = noisy_img.float()
                        clean_img = clean_img.float()

                        outputs = self.model(noisy_img)
                        if self.objective_type.lower() == 'poisson':
                            val_loss = self.objective(outputs, noisy_img)
                        elif self.objective_type.lower() == 'mse':
                            val_loss = self.objective(normalize_batch(outputs), normalize_batch(clean_img))

                        # Detach tensors and move to CPU before updating metrics to avoid
                        # holding the computation graph or GPU memory across batches
                        self.update_metrics(val_loss.detach().cpu(), normalize_batch(clean_img).detach().cpu(), normalize_batch(outputs).detach().cpu())
                        
                        print(f'Epoch [{epoch+1}/{self.n_epochs}], Validation, Step [{batch_idx+1}/{len(self.loader_val)}]')

                # print and reset metrics
                for metric in self.metrics:
                    print(f'{metric.name}: {metric.result():.4f}')
                    m_dict.update({f'val_{metric.name}': metric.result()})
                    metric.reset_states()

                # metric monitoring
                for metric_name in monitored_metrics:
                    self.mlflow_metric_monitoring(epoch, metric_name, m_dict[metric_name])

                # ensure we go back to training mode after validation
                self.model.train()

            # log metrics
            for metric_name, metric_value in m_dict.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch+1)

            # log reboot model as artifact
            self.mlflow_log_model_as_artifact(epoch, artifact_path="reboot_model")

        # log final best models
        for metric_name in monitored_metrics:
            mlflow.pytorch.log_model(self.model, artifact_path=f"best_model_{metric_name}", registered_model_name=f"Noise2Noise_2DPET_Model_{metric_name}")
