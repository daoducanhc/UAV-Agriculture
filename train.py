import os
import torch
import numpy as np
import setup.dataset as dataset
import setup.ResUNet as ResUNet
import setup.classifier as classifier
from torch.utils.data import SubsetRandomSampler

np.random.seed(0)
torch.manual_seed(0)

TRAIN_DATASET_PATH = 'dataset/train'
TEST_DATASET_PATH = 'dataset/test'
BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def sampler_indices(length):
    indices = list(range(length))
    np.random.shuffle(indices)
    index = int(np.floor(0.1 * length))
    valid_indices, train_indices = indices[:index], indices[index:]
    return train_indices, valid_indices

train_weed_dataset = dataset.WeedDataset(TRAIN_DATASET_PATH)
test_weed_dataset = dataset.WeedDataset(TEST_DATASET_PATH)

train_indices, valid_indices = sampler_indices(len(train_weed_dataset))
train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)

train_loader = torch.utils.data.DataLoader(weed_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(weed_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_weed_dataset, batch_size=1)

FILTER_LIST = [16,32,64,128,256]
model = ResUNet.ResUNet(FILTER_LIST).to(device)
name = 'outputs/ResUNet'
classifier = classifier.WeedClassifier(model, device)

model.train()
history = classifier.train(train_loader, valid_loader, learning_rate=0.001, epochs=50, name=name)

model.eval()
score = classifier.test(test_loader)
print(f'\nDice Score {score}')
