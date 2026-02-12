import os
import itertools
import numpy as np
import mlflow
import torch
import hashlib, json
import torch.nn.functional as F
from torch.utils.data import DataLoader

from noise2noise.data.data_loader import SinogramGenerator
from noise2noise.model.unet_noise2noise import UNetNoise2NoisePET as UNet

from pytorcher.trainer import PytorchTrainer
from pytorcher.utils import normalize_batch, PetForwardRadon, tensor_hash

def anscombe(x):
    return 2 * torch.sqrt( x + (3/8) )

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
                'half_life': 109.8*60,
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
            physics='backward_pet_radon',
            supervised=False,
            reconstruction_type='fbp',
            reconstruction_config={},
            measurement_consistency_balance=0.0,
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
        self.supervised = supervised
        self.image_size = simulator_config.get('image_size', (160,160))
        # Validate task parameters
        assert unet_input_domain in ['photon', 'image'], "unet_input_domain must be either 'photon' or 'image'."
        assert unet_output_domain in ['photon', 'image'], "unet_output_domain must be either 'photon' or 'image'."
        if unet_input_domain not in ['photon', 'image'] or unet_output_domain not in ['photon', 'image']:
            raise ValueError("unet_input_domain and unet_output_domain must be either 'photon' or 'image'.")
        if unet_input_domain != unet_output_domain:
            assert unet_input_domain == 'photon' and unet_output_domain == 'image', "Only photon to image domain conversion is supported."
        if unet_input_domain == 'image':
            assert reconstruction_type is not None and reconstruction_type in ['fbp'], "Currently only 'fbp' reconstruction is supported."
        assert n_splits > 1, "n_splits must be greater than 1 for noise2noise training."
        if unet_input_domain == 'photon' and unet_output_domain == 'image':
            self.physics = physics
        else:
            self.physics = None # physics-informed training is only relevant for photon to image domain conversion task
        self.unet_config = unet_config
        self.unet_input_domain = unet_input_domain
        self.unet_output_domain = unet_output_domain
        self.model_name = f'Noise2Noise_2DPET_{unet_input_domain}_to_{unet_output_domain}'
        if self.supervised:
            self.model_name += '_supervised'
        else:
            self.model_name += '_N2N'
        self.reconstruction_type = reconstruction_type
        unet_config.update({'reconstruction_type': self.reconstruction_type})
        self.reconstruction_config = reconstruction_config
        self.n_splits = n_splits # n_splits means n * (n - 1) pairs will be used for noise2noise training
        unet_config.update({'n_splits': n_splits})
        #
        if (self.unet_output_domain == self.unet_input_domain == 'image') or (self.supervised and unet_output_domain == 'image'):
            assert objective_type.lower() in ['mse'], "When both input and output domain is 'image', only 'MSE' is supported."
        else:
            assert objective_type.lower() in ['poisson', 'mse', 'mse_anscombe'], "When output domain is 'photon', only 'Poisson' and 'MSE' are supported."
        self.objective_type = objective_type
        self.forward_operator_type = 'radon'
        if self.supervised:
            self.measurement_consistency_balance = 0.0 # no measurement consistency loss in supervised setting.
        else:
            self.measurement_consistency_balance = measurement_consistency_balance
        self.seed = seed

        self.num_workers = num_workers

        self._id = hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()
        #
        super(Noise2NoiseTrainer, self).__init__()
        #
        if self.unet_input_domain == 'photon' and self.unet_output_domain == 'image':
            self.get_pet_forward_operator() # initialize forward operator for potential use in photon to image domain conversion and measurement consistency loss

    def get_metrics(self, metrics=[]):

        metrics.extend([
            [ 'PSNR', { 'name': 'im_psnr'} ],
            [ 'SSIM', { 'name': 'im_ssim'} ],
            [ 'PSNR', { 'name': 'val_nfpt_im_psnr'} ],
            [ 'SSIM', { 'name': 'val_nfpt_im_ssim'} ],
            [ 'PSNR', { 'name': 'val_prompt_im_psnr'} ],
            [ 'SSIM', { 'name': 'val_prompt_im_ssim'} ],
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
        #
        # These parameters may be used for inference reconstruction or training backprojection later on, so we store them as trainer attributes so that they can be accessed from mlflow.
        self.scanner_radius = self.dataset_train.sinogram_simulator.scanner_radius
        self.voxel_size_mm = self.dataset_train.sinogram_simulator.voxel_size_mm
        self.gaussian_PSF = self.dataset_train.sinogram_simulator.gaussian_PSF
        self.n_angles = self.dataset_train.sinogram_simulator.n_angles
        #
        self.sinogram_size = tuple(self.dataset_train[0][1].shape[-2:]) # (H, W) of the sinogram, needed for UNet input size
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


        return loader_train, loader_val
    
    def get_pet_forward_operator(self):
        self.pet_forward = PetForwardRadon(
            n_angles=self.n_angles,
            scanner_radius_mm=self.scanner_radius,
            gaussian_PSF_fwhm_mm=self.gaussian_PSF,
            voxel_size_mm=self.voxel_size_mm,
            device=self.device
        )

    def get_optimizer(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    def get_signature(self):
        sample = self.dataset_train.__getitem__(0)[1]
        signature = (1, ) + tuple(sample.shape)
        return signature

    def create_model(self):
        # model expects channel-first inputs: we'll add channel dimension when calling
        model = UNet(
            n_channels=1,
            n_classes=1,
            bilinear=True,
            normalize_input=True,
            input_domain=self.unet_input_domain,
            output_domain=self.unet_output_domain,
            physics=self.physics,
            geometry={
                'n_angles': self.n_angles,
                'scanner_radius_mm': self.scanner_radius,
                'gaussian_PSF_fwhm_mm': self.gaussian_PSF,
                'voxel_size_mm': self.voxel_size_mm
             },
             image_size=self.image_size,
             sinogram_size=self.sinogram_size,
            **self.unet_config
        )
        model = model.to(self.device)
        #
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

    def compute_reference_metrics(self):
        """
        Compute some reference metrics on validation set before training starts.
        This is useful to compare reconstruction results against.
        We compute the reconstruction from the noise-free sinogram (nfpt) only. This metric cannot be beaten by denoising.
        We compute the reconstruction from the prompt sinogram only. This metric must be beaten by denoising.
        """
        if self.unet_output_domain == self.unet_input_domain:
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (path, prompt, nfpt, gth, att, scale) in enumerate(self.loader_val):

                    print(f'Computing reference metrics, batch {batch_idx+1}/{len(self.loader_val)} ...')

                    # move data to device
                    prompt = prompt.to(self.device).float()
                    nfpt = nfpt.to(self.device).float()
                    gth = gth.to(self.device).float()
                    scale = scale.to(self.device).float()

                    # reconstruction from noise-free sinogram
                    recon_nfpt = self.model.reconstruction(nfpt, scale=scale)
                    # update im_ metrics for reference
                    metrics_to_update = [ m.name for m in self.metrics if 'nfpt' in m.name ]
                    self.update_metrics(None, normalize_batch(gth), normalize_batch(recon_nfpt), metric_names=metrics_to_update)

                    # reconstruction from prompt sinogram
                    recon_prompt = self.model.reconstruction(prompt, scale=scale)
                    # update im_ metrics for reference
                    metrics_to_update = [ m.name for m in self.metrics if 'prompt' in m.name ]
                    self.update_metrics(None, normalize_batch(gth), normalize_batch(recon_prompt), metric_names=metrics_to_update)

                print('Reference metrics on validation set before training:')
                for metric in self.metrics:
                    if 'nfpt' in metric.name or 'prompt' in metric.name:
                        print(f'{metric.name}: {metric.result():.4f}')
                        mlflow.log_metric(metric.name, metric.result())
                        metric.reset_states()

    def pet_forward_operator(self, image, attenuation_map=None, scale=None, forward_operator_type='radon'):
        """
        Apply the PET forward operator for self-supervised reconstruction.
        This will be used only if input domain is 'photon' and output domain is 'image',
        in which case we need to convert the reconstructed image back to sinogram domain for loss computation and gradient computation.
        
        :param image: Model prediction in the image domain.
        :param attenuation_map: Description
        :param scale: Description
        :param forward_operator_type: Description
        """
        #
        assert forward_operator_type.lower() in ['radon'], "Currently only 'radon' forward operator is supported for photon to image domain conversion."
        if forward_operator_type.lower() == 'radon':
            sinogram = self.pet_forward.forward(
                image=image,
                attenuation_map=attenuation_map,
                scale=scale
            )
        else:
            raise NotImplementedError("Currently only 'radon' forward operator is supported for photon to image domain conversion.")
        return sinogram
    
    def compute_loss(self, input, output, target, attenuation_map=None, scale=None):

        def compute_count_loss(output, target):
            """
            Compute count loss for photon domain.
            Photon domain loss can be either Poisson NLL or heteroscedastic MSE or MSE in Anscombe domain.
            """
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
            else:
                raise ValueError("Invalid objective type for photon domain. Supported types are 'poisson', 'mse' and 'mse_anscombe'.")
            return loss
        
        if self.unet_input_domain == self.unet_output_domain == 'photon':
            loss = compute_count_loss(output, target)
        elif self.unet_input_domain == self.unet_output_domain == 'image':
            loss = self.objective(output, target)
        elif self.unet_input_domain == 'photon' and self.unet_output_domain == 'image':
            if attenuation_map is None:
                print("Warning: No attenuation map provided for photon to image domain conversion. Assuming no attenuation for forward operator.")

            if not self.supervised:
                #
                projected_output = self.pet_forward_operator(
                    output,
                    attenuation_map=attenuation_map,
                    scale=scale,
                    forward_operator_type=self.forward_operator_type
                )
                #
                loss = compute_count_loss(projected_output, target)
                if self.measurement_consistency_balance > 0:
                    loss += self.measurement_consistency_balance * compute_count_loss(projected_output, input)
            else:
                loss = self.objective(output, target)
        #
        return loss

    def get_cached_adjoint(self, x, path, att, scale):
        """
        Get cached adjoint for given input tensor x. If not cached, compute and cache it.
        Caching will be efficient if split_prompt was called with consistent=True, which ensures that the same input tensor will be generated for a given path across epochs.

        :param x: list of tensors, each (B, C, H, W)  ← splits
        :param path: list[str] length B
        :param att: (B, 1, H, W) or compatible
        :param scale: (B, 1, 1, 1) or compatible
        :return: list of adjoint tensors matching x
        """

        device = x[0].device
        adjoints = []

        for s in x:  # loop over splits (usually small, like 2–8)
            B = s.shape[0]

            cache_paths = []
            hashes = []

            # Build cache paths
            for i in range(B):
                h = tensor_hash(s[i] + scale[i])
                hashes.append(h)
                cache_paths.append(
                    os.path.join(path[i], f'adjoint/adjoint_{h}.pt')
                )

            # Determine which need computation
            to_compute_idx = []
            adj_s = [None] * B

            for i, pth in enumerate(cache_paths):
                if os.path.exists(pth):
                    adj_s[i] = torch.load(pth, map_location=device)
                else:
                    to_compute_idx.append(i)

            # Batch compute missing adjoints
            if len(to_compute_idx) > 0:
                s_batch = torch.stack([s[i] for i in to_compute_idx], dim=0)
                att_batch = torch.stack([att[i] for i in to_compute_idx], dim=0)
                scale_batch = torch.stack([scale[i] for i in to_compute_idx], dim=0)

                adj_batch = self.model.adjoint(
                    s_batch,
                    attenuation_map=att_batch,
                    scale=scale_batch
                )

                # Save and place back in order
                for k, i in enumerate(to_compute_idx):
                    adj_i = adj_batch[k:k+1]  # keep batch dim
                    os.makedirs(os.path.dirname(cache_paths[i]), exist_ok=True)
                    torch.save(adj_i.cpu(), cache_paths[i])
                    adj_s[i] = adj_i.to(device)

            # Concatenate back to (B, C, H, W)
            adj_s = torch.cat(adj_s, dim=0)
            adjoints.append(adj_s)

        return adjoints

    def fit(self):

        def inference_pair_loss_metrics(noisy_img_a, noisy_img_b, clean_img, attenuation_map=None, scale=None):
            """ Perform inference with a as input and b as target, and compute loss """
            output = self.model(
                noisy_img_a,
                attenuation_map=attenuation_map,
                scale=scale
            )
            if self.supervised:
                loss_target = clean_img
            else:
                loss_target = noisy_img_b
            loss = self.compute_loss(input=noisy_img_a, output=output, target=loss_target, attenuation_map=attenuation_map, scale=scale)
            # We only update n2n_ metrics and loss here
            metrics_to_update = [ m.name for m in self.metrics if m.name.startswith('n2n_') or m.name.startswith('loss_') ]
            self.update_metrics(loss, normalize_batch(clean_img), normalize_batch(output), metric_names=metrics_to_update)
            # free up memory
            torch.cuda.empty_cache()
            #
            return loss

        if self.initial_epoch == 0:
            self.compute_reference_metrics()
        
        # Remove reference metrics from trainer metrics list
        self.metrics = [ m for m in self.metrics if 'nfpt' not in m.name and 'prompt' not in m.name ]
        
        for epoch in range(self.initial_epoch, self.n_epochs):

            m_dict_train = {}
            # TRAIN
            self.model.train()
            for batch_idx, (path, prompt, nfpt, gth, att, scale) in enumerate(self.loader_train):

                # Set target for taské
                if self.unet_output_domain == 'image':
                    target = gth
                else:
                    target = nfpt
                
                # Move data to device
                prompt = prompt.to(self.device).float()
                target = target.to(self.device).float()
                scale = scale.to(self.device).float()
                att = att.to(self.device).float()

                # split prompt with multinomial statistics
                split_losses = []
                if not self.supervised:
                    splitted_prompts = self.model.split_prompt(prompt, mode='multinomial')
                    pairwise_permutations = list(itertools.permutations(range(self.n_splits), 2))
                    scale = scale / self.n_splits
                else:
                    # In supervised setting, we force n_splits = 1 and use the prompt as is with the ground truth as target
                    splitted_prompts = [prompt]
                    pairwise_permutations = [(0, 0)] # dummy pair
                #
                # Apply reconstruction if needed
                if self.unet_input_domain == 'image':
                    x = [self.model.reconstruction(s, scale=scale) for s in splitted_prompts]
                else:
                    x = splitted_prompts

                # Compute adjoints if    needed for physics-informed training
                if self.physics is not None:
                    adjoints = self.get_cached_adjoint(x, path, att, scale)
                    # adjoints = [self.model.adjoint(x_, attenuation_map=att, scale=scale) for x_ in x]

                if self.unet_output_domain == 'photon' and not self.supervised:
                    target = target / self.n_splits # In photon domain, we divide poisson parameter accordingly
                else:
                    target = target

                # Denoise and compute loss on all pairs
                for (i, j) in pairwise_permutations:
                    if self.unet_input_domain == 'photon' and self.unet_output_domain == 'image' and self.physics is not None:
                        x_i = adjoints[i] # we use adjoint as input for inference and loss computation in physics-informed training
                    else:
                        x_i = x[i]
                    x_no_adjoint_i = x[i]
                    x_j = x[j]
                    # Inference and loss computation for pair (i, j)
                    if not self.supervised:
                        loss_ij = inference_pair_loss_metrics(x_i, x_j, target, attenuation_map=att, scale=scale)
                    else:
                        loss_ij = inference_pair_loss_metrics(x_i, x_no_adjoint_i, target, attenuation_map=att, scale=scale)
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
                    for batch_idx, (path, prompt, nfpt, gth, att, scale) in enumerate(self.loader_val):

                        # set target for task
                        if self.unet_output_domain == 'image':
                            target = gth
                        else:
                            target = nfpt

                        # move data to device
                        prompt = prompt.to(self.device).float()
                        target = target.to(self.device).float()
                        gth = gth.to(self.device).float()
                        scale = scale.to(self.device).float()
                        att = att.to(self.device).float()

                        if not self.supervised:
                            # Split data with multinomial statistics. This is done to match training data distribution
                            x = self.model.split_prompt(prompt, mode='multinomial')
                            #
                            scale = scale / self.n_splits
                        else:
                            x = [prompt, ]

                        # Apply reconstruction if needed
                        if self.unet_input_domain == 'image':
                            x = [self.model.reconstruction(s, scale=scale) for s in x]
                        

                        # Compute adjoints if needed for physics-informed training
                        if self.physics is not None:
                            adjoints = self.get_cached_adjoint(x, path, att, scale)
                            x_ = adjoints
                        else:
                            x_ = x
                        
                        # Denoise
                        splits_infered = self.model(
                            torch.cat(x_, dim=0), # concatenate splits along batch dimension for efficient inference
                            attenuation_map=att, # repeat attenuation map accordingly
                            scale=scale
                        )
                        if self.supervised:
                            splits_infered = splits_infered.chunk(1, dim=0) # dummy chunk to have same format as unsupervised case for easier code reuse
                        else:
                            splits_infered = torch.chunk(splits_infered, self.n_splits, dim=0)  # list of (B, C, H, W)
                        #
                        if self.unet_output_domain == 'photon' and not self.supervised:
                            target = target / self.n_splits # In photon domain, we divide poisson parameter accordingly
                        else:
                            target = target

                        for (i, j) in pairwise_permutations:
                            if self.unet_input_domain == 'photon' and self.unet_output_domain == 'image' and self.physics is not None:
                                x_i = adjoints[i] # we use adjoint as input for inference and loss computation in physics-informed training
                            else:
                                x_i = x[i]
                            x_j = x[j]
                            # inference on i
                            out_i = splits_infered[i]
                            # Loss computation for pair (i, j)
                            if not self.supervised:
                                val_loss = self.compute_loss(input=x_i, output=out_i, target=x_j, attenuation_map=att, scale=scale)
                            else:
                                val_loss = self.compute_loss(input=x_i, output=out_i, target=target, attenuation_map=att, scale=scale)

                            # Update n2n_ and loss metrics for validation
                            metrics_to_update = [ m.name for m in self.metrics if m.name.startswith('n2n_') or m.name.startswith('loss_') ]
                            self.update_metrics(val_loss, normalize_batch(target), normalize_batch(out_i), metric_names=metrics_to_update)

                        # Apply reconstruction if needed and average outputs
                        if self.unet_output_domain == 'photon':
                            output = self.model.reconstruction(*splits_infered, scale=scale) # (B, C, H, W)
                        else:
                            splits_infered = torch.stack(splits_infered, dim=0)  # (n_splits, B, C, H, W)
                            output = torch.mean(splits_infered, dim=0)  # (B, C, H, W)

                        # Update im_ metrics for validation
                        metrics_to_update = [ m.name for m in self.metrics if m.name.startswith('im_') ]
                        self.update_metrics(None, normalize_batch(gth), normalize_batch(output), metric_names=metrics_to_update)
                        # TODO compare im_ metrics with reconstruction without denoising and with nfpt only
                        
                        #
                        m_dict_val = {f'val_{metric.name}': metric.result() for metric in self.metrics}
                        print(f'Epoch [{epoch+1}/{self.n_epochs}], Validation, Step [{batch_idx+1}/{len(self.loader_val)}]', f'Metrics : {m_dict_val}')

                # print and reset metrics
                for metric in self.metrics:
                    # print(f'{metric.name}: {metric.result():.4f}')
                    # m_dict.update({f'val_{metric.name}': metric.result()})
                    metric.reset_states()

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