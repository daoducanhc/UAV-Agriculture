from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import torch
import numpy as np
from PIL import Image
import os

class WeedDataset(Dataset):
    def __init__(self, root, random_rotate=True):
        self.root = root
        self.random_rotate = random_rotate
        self.random_transform = {'hflip': TF.hflip,
                                'vflip': TF.vflip,
                                'rotate': TF.rotate}
        self.lb_arr = np.asarray([
            [0, 0, 0], # 0 for background
            [0, 255, 0], # 1 for plants
            [255, 0, 0]    # 2 for weeds
        ])

    def __len__(self):
        return len(os.listdir(os.path.join(self.root, 'mask')))

    def __getitem__(self, index):
        red_name = os.path.join(self.root, 'red', str(index)+'.png')
        nir_name = os.path.join(self.root, 'nir', str(index)+'.png')
        ndvi_name = os.path.join(self.root, 'ndvi', str(index)+'.png')
        mask_name = os.path.join(self.root, 'mask', str(index)+'.png')


        red = Image.open(red_name)
        nir = Image.open(nir_name)
        ndvi = Image.open(ndvi_name)
        mask = Image.open(mask_name)

        # nir =nir.crop((171, 93, 1060, 592))

        red = transforms.Resize((512, 512))(red)
        nir = transforms.Resize((512, 512))(nir)
        ndvi = transforms.Resize((512, 512))(ndvi)
        mask = transforms.Resize((512, 512))(mask)

        # r,g,b = rgb.split()
        image = Image.merge('RGB', (red,nir,ndvi))
        # image = transforms.Resize((512, 512))(image)

        if self.random_rotate==True:
            image, mask = self._random_transform(image, mask)

        mask = np.array(mask)
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.lb_arr):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        result = torch.tensor(label_mask)

        image = TF.to_tensor(image)
        # rgb = TF.to_tensor(rgb)

        sample = {'index': int(index), 'image': image, 'mask': result}
        return sample


    def _random_transform(self, image, mask):
        choice_list = list(self.random_transform)
        for _ in range(len(choice_list)):
            choice_key = random.choice(choice_list)

            action_prob = random.randint(0, 1)
            if action_prob >= 0.5:
                if choice_key == 'rotate':
                    rotation = random.randint(15, 75)
                    image = self.random_transform[choice_key](image, rotation)
                    mask = self.random_transform[choice_key](mask, rotation)
                else:
                    image = self.random_transform[choice_key](image)
                    mask = self.random_transform[choice_key](mask)
            choice_list.remove(choice_key)

        return image, mask
