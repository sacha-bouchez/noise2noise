import os

import torch
from torch.utils.data import Dataset

from phantom_simulation.object_simulator import Phantom2DPetGenerator
from phantom_simulation.sinogram_simulator import SinogramSimulator

class SinogramGenerator(Dataset):

    def __init__(self, binsimu, dest_path='./', length=10, image_size=(256,256), voxel_size=(2,2,2), seed=None):
        self.binsimu = binsimu
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        self.dest_path = dest_path
        self.length = length
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.phantom_generator = Phantom2DPetGenerator(shape=image_size, voxel_size=voxel_size)
        self.sinogram_simulator = SinogramSimulator(binsimu=binsimu, save_castor=False, seed=seed)
        self.seed = seed

    def __len__(self):
        return self.length

    def normalize(self, x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

    def generate_sample(self, idx):

        # Set seeds
        if self.seed is not None:
            torch.manual_seed(self.seed + idx)
            torch.cuda.manual_seed_all(self.seed + idx)
            self.phantom_generator.set_seed(self.seed + idx)

        #
        dest_path = os.path.join(self.dest_path, f'data_{idx}')
        if not os.path.exists(f'{dest_path}/simu/simu_nfpt.s.hdr'):
        
            # Generate phantom
            obj_path, att_path = self.phantom_generator.run(os.path.join(self.dest_path, f'data_{idx}', f'object'))

            # Simulate sinogram
            self.sinogram_simulator.run(img_path=obj_path, img_att_path=att_path, dest_path=dest_path)

        # Read hdr file to get matrix size
        with open(f'{dest_path}/simu/simu_nfpt.s.hdr', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('matrix size [1]'):
                    n_x = int(line.split('=')[1].strip())
                if line.startswith('matrix size [2]'):
                    n_y = int(line.split('=')[1].strip())
        # Read Noise-Free Prompt data
        with open(f'{dest_path}/simu/simu_nfpt.s', 'rb') as f:
            data_nfpt = torch.frombuffer(f.read(), dtype=torch.float32)
            data_nfpt = data_nfpt.reshape((n_y, n_x))

        # Generate 2 Poisson noisy versions
        data_noisy_1 = torch.poisson(data_nfpt)
        data_noisy_2 = torch.poisson(data_nfpt)

        # Normalize
        data_noisy_1 = self.normalize(data_noisy_1)
        data_noisy_2 = self.normalize(data_noisy_2)
        data_nfpt = self.normalize(data_nfpt)

        return data_noisy_1, data_noisy_2, data_nfpt

    def __getitem__(self, idx):

        noisy_1, noisy_2, clean = self.generate_sample(idx)
        return noisy_1, noisy_2, clean

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
