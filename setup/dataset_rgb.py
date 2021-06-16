from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import torch
import numpy as np
from PIL import Image
import os

class WeedDataset_RGB(Dataset):
    def __init__(self, root, random_rotate=True):
        self.root = root
        self.random_transform = {'hflip': TF.hflip,
                                'vflip': TF.vflip,
                                'rotate': TF.rotate}
        self.lb_arr = np.asarray([
            [0, 0, 0], # 0 for background
            [0, 255, 0], # 1 for plants
            [255, 0, 0]    # 2 for weeds
        ])

    def __len__(self):
        return len(os.listdir(os.path.join(self.root, 'nir')))

    def __getitem__(self, index):
        rgb_name = os.path.join(self.root, 'rgb', str(index)+'.png')
        nir_name = os.path.join(self.root, 'nir', str(index)+'.png')
        ndvi_name = os.path.join(self.root, 'ndvi', str(index)+'.png')

        rgb = Image.open(rgb_name)
        nir = Image.open(nir_name)
        ndvi = Image.open(ndvi_name)

        # nir =nir.crop((171, 93, 1060, 592))

        rgb = transforms.Resize((512, 512))(rgb)
        nir = transforms.Resize((512, 512))(nir)
        ndvi = transforms.Resize((512, 512))(ndvi)

        rgb = rgb.convert("RGB")
        r,g,b = rgb.split()

        image = Image.merge('RGB', (r,nir,ndvi))
        # image = transforms.Resize((512, 512))(image)

        image = TF.to_tensor(image)

        sample = {'index': int(index), 'image': image, 'rgb': rgb}
        return sample