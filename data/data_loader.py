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
        self.phantom_generator = Phantom2DPetGenerator(shape=image_size, voxel_size=voxel_size)
        if seed is None:
            self.seed = random.randint(0, 1e32)
        self.seed = seed
        if simulator_type == 'castor':
            self.sinogram_simulator = SinogramSimulatorCastor(binsimu=binsimu, save_castor=False, seed=seed) # NOTE this seed is useless
        else:
            self.sinogram_simulator = SinogramSimulator(seed=seed)
        #
        self.hashcode = self.get_generator_hashcode()

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
            self.sinogram_simulator.run(img_path=obj_path, img_att_path=att_path, dest_path=dest_path)

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

    binsimu = os.path.join(os.getenv("WORKSPACE"), "simulator", "bin")
    dest_path = os.path.join(os.getenv("WORKSPACE"), "data")
    dataset = SinogramGenerator(binsimu=binsimu, dest_path=dest_path, length=2, seed=42)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (noisy_1, noisy_2, clean) in enumerate(loader):
        print(f'Batch {i}:')
        print(f'  Noisy 1 avg: {torch.mean(noisy_1)},')
        print(f'  Noisy 2 avg: {torch.mean(noisy_2)}')
        print(f'  Clean avg: {torch.mean(clean)}')

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(noisy_1[0], cmap='gray')
    ax[0].set_title('Noisy 1')
    ax[1].imshow(noisy_2[0], cmap='gray')
    ax[1].set_title('Noisy 2')
    ax[2].imshow(clean[0], cmap='gray')
    ax[2].set_title('Clean')
    plt.show()
