import os
import itertools
import numpy as np
import mlflow
import torch
import hashlib, json
import torch.nn.functional as F
from torch.utils.data import DataLoader

from noise2noise.data.data_loader import SinogramGenerator
from noise2noise.model.unet_noise2noise import UNetNoise2Noise as UNet

from pytorcher.trainer import PytorchTrainer

from pytorcher.utils import normalize_batch, iradon as iradon_torch


class Noise2NoiseTrainer(PytorchTrainer):

    def __init__(
            self,
            dest_path=f"{os.getenv('WORKSPACE')}/data/noise2noise",
            dataset_train_size=2048,
            dataset_val_size=512,
            val_freq=1,
            n_epochs=25,
            batch_size=4,
            shuffle=True,
            simulator_config={
                'image_size' : (160,160),
                'voxel_size' : (2,2,2),
                'n_angles' : 300,
                'acquisition_time' : 253.3, # temporary value, will be overridden
                'scanner_radius' : 300,
                'nb_counts' : 1e6,
            },
            learning_rate=1e-3,
            unet_config = {
                'conv_layer_type': 'SinogramConv2d',
                'n_levels': 4,
                'global_conv': 32,
            },
            unet_input_domain='photon',
            unet_output_domain='photon',
            reconstruction_algorithm='fbp',
            reconstruction_config={'filter_name': 'ramp'},
            n_splits=2,
            num_workers=10,
            objective_type='poisson',
            seed=42
        ):
        self.dest_path = dest_path
        self.dataset_train_size = dataset_train_size
        self.dataset_val_size = dataset_val_size
        self.val_freq = val_freq
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.simulator_config = simulator_config
        self.learning_rate = learning_rate

        # Validate task parameters
        assert unet_input_domain in ['photon', 'image'], "unet_input_domain must be either 'photon' or 'image'."
        assert unet_output_domain in ['photon', 'image'], "unet_output_domain must be either 'photon' or 'image'."
        if unet_input_domain not in ['photon', 'image'] or unet_output_domain not in ['photon', 'image']:
            raise ValueError("unet_input_domain and unet_output_domain must be either 'photon' or 'image'.")
        if unet_input_domain != unet_output_domain:
            assert unet_input_domain == 'photon' and unet_output_domain == 'image', "Only photon to image domain conversion is supported."
        if unet_input_domain == 'image':
            assert reconstruction_algorithm is not None and reconstruction_algorithm in ['fbp'], "Currently only 'fbp' reconstruction is supported."
        assert n_splits > 1, "n_splits must be greater than 1 for noise2noise training."
        self.unet_config = unet_config
        self.unet_input_domain = unet_input_domain
        self.unet_output_domain = unet_output_domain
        self.model_name = f'Noise2Noise_2DPET_{unet_input_domain}_to_{unet_output_domain}'
        self.reconstruction_algorithm = reconstruction_algorithm
        self.reconstruction_config = reconstruction_config
        self.n_splits = n_splits # n_splits means n * (n - 1) pairs will be used for noise2noise training

        # 
        if self.unet_output_domain == 'image':
            assert objective_type.lower() in ['mse'], "When output domain is 'image', only 'MSE' is supported."
        else:
            assert objective_type.lower() in ['poisson', 'mse', 'mse_anscombe'], "When output domain is 'photon', only 'Poisson' and 'MSE' are supported."
        self.objective_type = objective_type
        self.seed = seed

        self.num_workers = num_workers

        self._id = hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()

        super(Noise2NoiseTrainer, self).__init__()

    def get_metrics(self, metrics=[]):

        metrics.extend([
            [ 'PSNR', { 'name': 'im_psnr'} ],
            [ 'SSIM', { 'name': 'im_ssim'} ],
            [ 'PSNR', { 'name': 'n2n_psnr'} ],
        ])
        if self.unet_output_domain == 'image':
            metrics.append( [ 'SSIM', { 'name': 'n2n_ssim'} ] )
        if f'loss_{self.objective_type.lower()}' not in [ m[0].lower() for m in metrics ]:
            metrics.append( [ 'Mean', { 'name': f'loss_{self.objective_type.lower()}'} ] )
        #
        self.metrics = super(Noise2NoiseTrainer, self).get_metrics(metrics)
        return self.metrics

    def update_metrics(self, loss, y_true, y_pred, metric_names=[]):
        for metric in self.metrics:
            if metric_names and metric.name not in metric_names:
                continue
            if 'loss' in metric.name:
                if loss is not None:
                    metric.update_state(None, loss)
            else:
                metric.update_state(y_true, y_pred)

    def create_data_loader(self):

        self.dataset_train = SinogramGenerator(
                                         dest_path=os.path.join(self.dest_path, 'train'),
                                         length=self.dataset_train_size,
                                         seed=self.seed,
                                         **self.simulator_config
        )
        loader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        self.dataset_val_seed = int(1e5) # Seed is fixed to have consistent validation sets. Changing image size or voxel size will give different results.
        if 'acquisition_time' in self.simulator_config:
            self.simulator_config.pop('acquisition_time') # ensure acquisition time is same as training set
        self.dataset_val = SinogramGenerator(
                                         dest_path=os.path.join(self.dest_path, 'val'),
                                         length=self.dataset_val_size,
                                         seed=self.dataset_val_seed,
                                         acquisition_time=self.dataset_train.acquisition_time, # use same acquisition time as training set
                                         **self.simulator_config
        )
        loader_val = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.image_size = self.simulator_config.get('image_size', (160,160))

        return loader_train, loader_val

    def get_optimizer(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    def get_signature(self):
        sample = self.dataset_train.__getitem__(0)[0]
        signature = (1, ) + tuple(sample.shape)
        return signature

    def create_model(self):
        # model expects channel-first inputs: we'll add channel dimension when calling
        model = UNet(
            n_channels=1,
            n_classes=1,
            bilinear=True,
            normalize_input=True,
            **self.unet_config
        )
        model = model.to(self.device)
        return model

    def get_objective(self):
        type = self.objective_type.lower()
        if 'mse' in type:
            objective = torch.nn.MSELoss()
        elif type == 'poisson':
            objective = torch.nn.PoissonNLLLoss(log_input=False, full=False, reduction='mean')
        #
        self.model.loss_type = type # inform model about loss type for potential post-processing
        #
        return objective

    def log_and_reset_metrics(self, epoch):
        print(f'End of Epoch {epoch+1}, metrics: ')
        for metric in self.metrics:
            print(f'{metric.name}: {metric.result():.4f}')

            # Log metrics to MLflow
            mlflow.log_metric(metric.name, metric.result(), step=epoch)

            # Reset metric
            metric.reset_states()

    def reconstruction(self, *x, **kwargs):
        """
        Apply reconstruction algorithm to batches of sinograms x's.
        
        :param x: (B, C, H, W) sinogram tensor
        """
        batch_size = x[0].shape[0]
        out = []
        for i in range(batch_size):
            # jointly reconstruct i-th batch sample(s) (multiple sinograms may be provided)
            batch_i_sinos = [ xx.select(0, i) for xx in x ]  # list of (C, H, W)
            if self.reconstruction_algorithm.lower() == 'fbp':
                recon = [ iradon_torch(torch.transpose(sino.squeeze(0), 0, 1), circle=False, output_size=max(self.image_size), **kwargs).unsqueeze(0) for sino in batch_i_sinos ]  # list of (C, H, W)
                recon = torch.stack(recon, dim=0)  # (n_sinos, C, H, W)
                recon = torch.mean(recon, dim=0)  # (C, H, W)
            else:
                raise NotImplementedError(f'Reconstruction algorithm {self.reconstruction_algorithm} not implemented yet.')
            out.append(recon)
        out = torch.stack(out, dim=0)  # (B, C, H, W)
        # reshape to (B, C, H, W)
        out = out.view(-1, out.shape[1], out.shape[-2], out.shape[-1])
        return out

    def split_prompt(self, prompt, mode='multinomial'):
        """
        Split prompt sinogram into n_splits sinograms with multinomial statistics.
        
        :param prompt: (B, C, H, W) sinogram tensor
        :return: list of n_splits sinogram tensors, each of shape (B, C, H, W)
        """
        if mode == 'multinomial':
            n = self.n_splits
            # Sample probabilities from a Dirichlet distribution
            alpha = torch.ones(n, device=prompt.device)
            probs = torch.distributions.Dirichlet(alpha).sample(prompt.shape)  # (B, C, H, W, n)

            splits = torch.floor(probs * prompt.unsqueeze(-1)).permute(4, 0, 1, 2, 3)  # (n, B, C, H, W)

            residual = prompt - splits.sum(dim=0)  # (B, C, H, W)
            splits[-1] += residual  # add residual to last split
            for i in range(splits.shape[1]):
                assert prompt[i, ...].sum() == splits[:, i, ...].sum(), "Splits do not sum up to original prompt!"
            
            # Cast splits to list of tensors
            splits = torch.unbind(splits, dim=0)  # list of n tensors of shape (B, C, H, W)
        else:
            raise ValueError(f'Unknown split mode: {mode}')
        return splits

    def fit(self):

        def anscombe(x):
            return 2 * torch.sqrt( x + (3/8) )

        def compute_loss(output, target):
            #
            if self.unet_input_domain == self.unet_output_domain == 'photon':
                if self.objective_type.lower() == 'poisson':
                    loss = self.objective(output, target)
                elif self.objective_type.lower() == 'mse':
                    eps = 1.0 # to avoid division by zero, tuned according to Poisson range
                    loss = self.objective(output / torch.sqrt(target + eps), target / torch.sqrt(target + eps)) # heteroscedastic MSE
                elif self.objective_type.lower() == 'mse_anscombe':
                    if not self.model.training:
                        output = anscombe(output) # At inference time, rescale back to Anscombe domain
                    target = anscombe(target)
                    loss = self.objective(output, target)
            elif self.unet_input_domain == self.unet_output_domain == 'image':
                loss = self.objective(output, target)
            elif self.unet_input_domain == 'photon' and self.unet_output_domain == 'image':
                raise NotImplementedError("Loss computation for photon to image domain is not implemented yet.")
                # TODO: Radon forward projection must be implemented in torch to compute loss
            return loss

        def inference_pair_loss_metrics(noisy_img_a, noisy_img_b, clean_img):
            """ Perform inference with a as input and b as target, and compute loss """
            output = self.model(noisy_img_a)
            loss = compute_loss(output, noisy_img_b)
            # We only update n2n_ metrics and loss here
            metrics_to_update = [ m.name for m in self.metrics if m.name.startswith('n2n_') or m.name.startswith('loss_') ]
            self.update_metrics(loss, normalize_batch(clean_img), normalize_batch(output), metric_names=metrics_to_update)
            # free up memory
            del output
            torch.cuda.empty_cache()
            #
            return loss
        
        for epoch in range(self.initial_epoch, self.n_epochs):

            m_dict_train = {}
            # TRAIN
            for batch_idx, (prompt, nfpt, gth) in enumerate(self.loader_train):

                # Set seed for given batch for reproducibility
                torch.manual_seed(self.seed + epoch + batch_idx)
                torch.cuda.manual_seed_all(self.seed + epoch + batch_idx)

                # Set target for task
                if self.unet_output_domain == 'image':
                    target = gth
                else:
                    target = nfpt
                
                # Move data to device
                prompt = prompt.to(self.device).float()
                target = target.to(self.device).float()

                # split prompt with multinomial statistics
                splitted_prompts = self.split_prompt(prompt, mode='multinomial')
                pairwise_permutations = list(itertools.permutations(range(self.n_splits), 2))
                split_losses = []
                #

                # Apply reconstruction if needed
                if self.unet_input_domain == 'image':
                    x = [self.reconstruction(s) for s in splitted_prompts]
                else:
                    x = splitted_prompts

                # Denoise and compute loss on all pairs
                for (i, j) in pairwise_permutations:
                    x_i = x[i]
                    x_j = x[j]
                    # Inference and loss computation for pair (i, j)
                    loss_ij = inference_pair_loss_metrics(x_i, x_j, target)
                    split_losses.append(loss_ij)
                # global loss
                loss = sum(split_losses) / len(split_losses)  # average over all pairs
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #
                m_dict_train = {metric.name: metric.result() for metric in self.metrics if metric.name.startswith('n2n_') or metric.name.startswith('loss_')}
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Train,  Step [{batch_idx+1}/{len(self.loader_train)}]', f"Metrics : {m_dict_train}")

            for metric in self.metrics:
                # print(f'{metric.name}: {metric.result():.4f}')
                # m_dict.update({metric.name: metric.result()})
                # reset state
                metric.reset_states()

            torch.cuda.empty_cache()

            m_dict_val = {}
            # VALIDATION
            if (epoch + 1) % self.val_freq == 0:
                # run model in evaluation mode and avoid building computation graph
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (prompt, nfpt, gth) in enumerate(self.loader_val):

                        # Set seed for given batch for reproducibility
                        torch.manual_seed(self.seed + batch_idx)
                        torch.cuda.manual_seed_all(self.seed + batch_idx)

                        # set target for task
                        if self.unet_output_domain == 'image':
                            target = gth
                        else:
                            target = nfpt

                        # move data to device
                        prompt = prompt.to(self.device).float()
                        target = target.to(self.device).float()

                        # Split data with multinomial statistics. This is done to match training data distribution
                        splits = self.split_prompt(prompt, mode='multinomial')

                        # Concatenate splits along batch dimension
                        splits = torch.cat(splits, dim=0)

                        # Apply reconstruction if needed
                        if self.unet_input_domain == 'image':
                            splits = self.reconstruction(splits)

                        # Denoise
                        splits_denoised = self.model(splits)

                        # Expand splits to have (n_splits, B, C, H, W)
                        splits_denoised = torch.chunk(splits_denoised, self.n_splits, dim=0)  # list of (B, C, H, W)

                        # Update n2n_ and loss metrics for validation
                        metrics_to_update = [ m.name for m in self.metrics if m.name.startswith('n2n_') or m.name.startswith('loss_') ]
                        for splits_denoised_i in splits_denoised:
                            if self.unet_output_domain == 'photon':
                                target_ = target / self.n_splits # In photon domain, we divide poisson parameter accordingly
                            else:
                                target_ = target
                            val_loss = compute_loss(splits_denoised_i, target_)
                            self.update_metrics(val_loss, normalize_batch(target_), normalize_batch(splits_denoised_i), metric_names=metrics_to_update)

                        # Apply reconstruction if needed and average outputs
                        if self.unet_output_domain == 'photon':
                            output = self.reconstruction(*splits_denoised) # (B, C, H, W)
                        else:
                            splits_denoised = torch.stack(splits_denoised, dim=0)  # (n_splits, B, C, H, W)
                            output = torch.mean(splits_denoised, dim=0)  # (B, C, H, W)

                        # Update im_ metrics for validation
                        metrics_to_update = [ m.name for m in self.metrics if m.name.startswith('im_') ]
                        self.update_metrics(None, normalize_batch(gth), normalize_batch(output), metric_names=metrics_to_update)
                        
                        #
                        m_dict_val = {f'val_{metric.name}': metric.result() for metric in self.metrics}
                        print(f'Epoch [{epoch+1}/{self.n_epochs}], Validation, Step [{batch_idx+1}/{len(self.loader_val)}]', f'Metrics : {m_dict_val}')

                # print and reset metrics
                for metric in self.metrics:
                    # print(f'{metric.name}: {metric.result():.4f}')
                    # m_dict.update({f'val_{metric.name}': metric.result()})
                    metric.reset_states()

                # ensure we go back to training mode after validation
                self.model.train()

            # metric monitoring
            m_dict = {**m_dict_train, **m_dict_val}
            monitored_metrics = [ m for m in list(m_dict.keys()) if m.startswith('val_loss') ]  # currently only loss monitoring is supported
            for metric_name in monitored_metrics:
                if 'loss' in metric_name:
                    mode = 'min'
                else:
                    mode = 'max'
                self.mlflow_metric_monitoring(epoch, metric_name, m_dict[metric_name], mode=mode)

            # log metrics
            for metric_name, metric_value in m_dict.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch+1)

            # log reboot model as artifact
            self.mlflow_log_model_as_artifact(epoch, artifact_path="reboot_model")

            torch.cuda.empty_cache()

        # log final best models
        for metric_name in monitored_metrics:
            mlflow.pytorch.log_model(self.model, artifact_path=f"best_model_{metric_name}", registered_model_name=f"{self.model_name}_{metric_name}")

        # log final model
        mlflow.pytorch.log_model(self.model, artifact_path="final_model", registered_model_name=self.model_name)