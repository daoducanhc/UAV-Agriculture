import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np

TRAIN_DATASET_PATH = 'dataset_augmentation/sub'

tensor_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

all_data = ImageFolder(TRAIN_DATASET_PATH, tensor_transform)
# print(all_data)
data_loader = torch.utils.data.DataLoader(all_data, batch_size=512, shuffle=True)

tol_mean = []
tol_std = []

for i, data in enumerate(data_loader):
    # tensor to numpy
    numpy_image = data[0].numpy()

    # compute mean and standard devation
    # axis=023 bcs data[0] size is (batchsize, channels, width, height)
    # so we want to compute mean and std of 3 channels
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std = np.std(numpy_image, axis=(0,2,3))

    tol_mean.append(batch_mean)
    tol_std.append(batch_std)

tol_mean = np.array(tol_mean).mean(axis=0)
tol_std = np.array(tol_std).mean(axis=0)
print(tol_mean)
print(tol_std)
