import os
import random
import shutil

import torch
from torch.utils.data import Dataset

from pet_simulator import SinogramSimulatorCastor, SinogramSimulator
from phantom_simulation import Phantom2DPetGenerator

from tools.image.castor import read_castor_binary_file

class SinogramGenerator(Dataset):

    def __init__(self, simulator_type, binsimu=None, dest_path='./', length=10, image_size=(256,256), voxel_size=(2,2,2), seed=None):
        assert simulator_type in ['castor', 'toy'], "simulator_type must be either 'castor' or 'toy'"
        if simulator_type == 'castor':
            assert binsimu is not None, "binsimu path must be provided for castor simulator"
            self.binsimu = binsimu
        self.dest_path = os.path.join(dest_path, simulator_type)
        if not os.path.exists(self.dest_path):
            os.makedirs(self.dest_path)
        self.length = length
        self.image_size = image_size
        self.voxel_size = voxel_size
        #
        volume_activity = 1e3  # in kBq/ml, this is a reasonable pre-computed value for toy simulator
        self.phantom_generator = Phantom2DPetGenerator(shape=image_size, voxel_size=voxel_size, volume_activity=volume_activity)
        #
        if seed is None:
            self.seed = random.randint(0, 1e32)
        self.seed = seed
        #
        # Simulation parameters
        self.nb_counts = 3e6
        self.half_life = 109.8 * 60  # in seconds, F-18
        self.acquisition_time = 600  # in seconds, 600 seconds is a pre-computed reasonable value for toy simulator with 3e6 counts
        self.scatter_component = 0.35
        self.random_component = 0.40
        self.gaussian_PSF = 4  # in mm
        #
        if simulator_type == 'castor':
            self.sinogram_simulator = SinogramSimulatorCastor(binsimu=binsimu, save_castor=False, seed=seed) # NOTE this seed is useless
        else:
            self.sinogram_simulator = SinogramSimulator(seed=seed)
            if self.acquisition_time is None:
                self.acquisition_time = self.sinogram_simulator.set_acquisition_time(n_samples=100, nb_counts=self.nb_counts, half_life=self.half_life, volume_activity=self.phantom_generator.volume_activity)
        #
        self.hashcode = self.get_generator_hashcode()

        #

    def __len__(self):
        return self.length

    def get_generator_hashcode(self):
        return hash((
                     tuple(self.image_size),
                     tuple(self.voxel_size),
                     self.seed)) & 0xffffffff


    def simulate_sinogram(self, idx):

        # Set seeds
        torch.manual_seed(self.seed + idx)
        torch.cuda.manual_seed_all(self.seed + idx)
        self.phantom_generator.set_seed(self.seed + idx)

        # Create unique hashcode for data sample with idx and dataset generator hashcode
        data_hashcode = hash((idx, self.hashcode)) & 0xffffffff

        #
        dest_path = os.path.join(self.dest_path, f'data_{data_hashcode}')
        if not os.path.exists(f'{dest_path}/simu/simu_nfpt.s.hdr'):

            # Generate phantom
            obj_path, att_path = self.phantom_generator.run(os.path.join(self.dest_path, f'data_{data_hashcode}', f'object'))
            # Simulate sinogram
            self.sinogram_simulator.run(img_path=obj_path, img_att_path=att_path, dest_path=dest_path,
                                        nb_counts=self.nb_counts,
                                        half_life=self.half_life,
                                        acquisition_time=self.acquisition_time,
                                        scatter_component=self.scatter_component,
                                        random_component=self.random_component,
                                        gaussian_PSF=self.gaussian_PSF,)

        # Read hdr file to get matrix size
        data_nfpt = read_castor_binary_file(f'{dest_path}/simu/simu_nfpt.s.hdr').squeeze()
        data_nfpt = torch.from_numpy(data_nfpt)

        return dest_path, data_nfpt


    def generate_sample(self, idx):

        # Simulate sinogram
        _, data_nfpt = self.simulate_sinogram(idx)

        # Generate 2 Poisson noisy versions
        data_noisy_1 = torch.poisson(data_nfpt)
        data_noisy_2 = torch.poisson(data_nfpt)


        return data_noisy_1, data_noisy_2, data_nfpt

    def __getitem__(self, idx):

        noisy_1, noisy_2, clean = self.generate_sample(idx)
        return noisy_1, noisy_2, clean

class SinogramGeneratorReconstructionTest(SinogramGenerator):

    """
    This generator is quite different in the sense that it is aimed to evaluate post-denoising reconstruction.
    Therefore, it only generates one noisy sinogram and the clean ground truth image (not clean sinogram).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def generate_sample(self, idx):

        # Simulate sinogram
        dest_path, data_nfpt = self.simulate_sinogram(idx)
        
        # Read Image ground truth
        with open(f'{dest_path}/object/object.img', 'rb') as f:
            object_gth = torch.frombuffer(f.read(), dtype=torch.float32)
            object_gth = object_gth.reshape(self.image_size)

        # Generate 1 Poisson noisy version
        data_noisy = torch.poisson(data_nfpt)

        return dest_path, data_noisy, data_nfpt, object_gth

    def __getitem__(self, idx):

        dest_path, noisy, sinogram_clean, image_clean = self.generate_sample(idx)
        return dest_path, noisy, sinogram_clean, image_clean        

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    from tools.image.metrics import PSNR, SSIM

    binsimu = os.path.join(os.getenv("WORKSPACE"), "simulator", "bin")
    dest_path = os.path.join(os.getenv("WORKSPACE"), "data")
    dataset = SinogramGenerator('toy', binsimu=binsimu, dest_path=dest_path, length=2, seed=42)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (noisy_1, noisy_2, clean) in enumerate(loader):
        print(f'Batch {i}:')
        print(f'  Noisy 1 sum: {torch.sum(noisy_1)},')
        print(f'  Noisy 1 PSNR: {SSIM(noisy_1.numpy(), clean.numpy()):.2f} dB')
        print(f'  Noisy 2 sum: {torch.sum(noisy_2)}')
        print(f'  Noisy 2 PSNR: {SSIM(noisy_2.numpy(), clean.numpy()):.2f} dB')
        # print(f'  Clean avg: {torch.sum(clean)}')

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(noisy_1[0], cmap='gray')
    ax[0].set_title('Noisy 1')
    ax[1].imshow(noisy_2[0], cmap='gray')
    ax[1].set_title('Noisy 2')
    ax[2].imshow(clean[0], cmap='gray')
    ax[2].set_title('Clean')
    plt.show()
