import os
import torch
import numpy as np
import setup.dataset as dataset
import setup.ResUNet as ResUNet
import setup.UNet as UNet
import setup.CNN as CNN
import setup.HSCNN as HSCNN
import setup.DeepLabV3 as DeepLabV3
import setup.SegNet as SegNet
import setup.classifier as classifier
from torch.utils.data import SubsetRandomSampler

np.random.seed(0)
torch.manual_seed(0)

# TRAIN_DATASET_PATH = 'dataset/train'
# name = 'outputs/original_dataset/'

TRAIN_DATASET_PATH = 'dataset/train'
name = 'outputs/'

TEST_DATASET_PATH = 'dataset/test'
BATCH_SIZE = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def sampler_indices(length):
    indices = list(range(length))
    np.random.shuffle(indices)
    index = int(np.floor(0.5 * length))
    valid_indices, train_indices = indices[:index], indices[index:]
    return train_indices, valid_indices

train_weed_dataset = dataset.WeedDataset(TRAIN_DATASET_PATH)
test_weed_dataset = dataset.WeedDataset(TEST_DATASET_PATH, random_rotate=False)

train_indices, valid_indices = sampler_indices(len(train_weed_dataset))
test_indices = list(range(len(test_weed_dataset)))
train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(train_weed_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_weed_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_weed_dataset, batch_size=1, sampler=test_sampler)

model = ResUNet.ResUNet().to(device)
name = name + 'ResUNet'
classifier = classifier.WeedClassifier(model, device)
history = classifier.train(train_loader, valid_loader, test_loader, learning_rate=0.001, epochs=100, name=name)

score = classifier.test(test_loader)
print(f'F1 Score {score}')
