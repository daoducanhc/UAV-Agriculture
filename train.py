import os
import torch
import numpy as np
import setup.dataset as dataset
import setup.ResUNet as ResUNet
import setup.classifier as classifier
from torch.utils.data import SubsetRandomSampler

np.random.seed(0)
torch.manual_seed(0)

# TRAIN_DATASET_PATH = 'dataset/train'
TRAIN_DATASET_PATH = 'dataset_augmentation'
TEST_DATASET_PATH = 'dataset/test'
BATCH_SIZE = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def sampler_indices(length):
    indices = list(range(length))
    np.random.shuffle(indices)
    index = int(np.floor(0.1 * length))
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

FILTER_LIST = [16,32,64,128,256]
model = ResUNet.ResUNet(FILTER_LIST).to(device)
name = 'outputs/ResUNet'
classifier = classifier.WeedClassifier(model, device)

model.train()
history = classifier.train(train_loader, valid_loader, learning_rate=0.001, epochs=25, name=name)

# BATCH_SIZE = 6
# lr=0.001 ep=25 step=7 gamma=0.5   =>   score=0.3697

# BATCH_SIZE = 8
# ---------------old loss (cross entropy))
# lr=0.001 ep=30 step=7 gamma=0.5   =>   score=0.6128
# lr=0.001 ep=30 step=5 gamma=0.5   =>   score=0.5965
# lr=0.001 ep=50 step=5 gamma=0.5   =>   score=0.5822
# ---------------old loss (cross entropy)

# lr=0.001 ep=30 step=7 gamma=0.4   =>   score=0.5943

# BATCH_SIZE = 12
# lr=0.001 ep=25 step=7 gamma=0.5   =>   score=0.613

# BATCH_SIZE = 25
# lr=0.001 ep=25 step=7 gamma=0.5   =>   score=0.5922
# lr=0.001 ep=50 step=7 gamma=0.5   =>   score=0.5900

### F1

# BATCH_SIZE = 4, size (1024,1024)
# lr=0.001 ep=25 step=7 gamma=0.5   =>   score=0.7008

# BATCH_SIZE = 8
# lr=0.001 ep=25 step=7 gamma=0.5   =>   score=0.7317 ***
# one more layer 512                =>   score=0.6453

# BATCH_SIZE = 10
# lr=0.001 ep=25 step=7 gamma=0.5   =>   score=0.7089

# RED EQUALIZE 0.633
# NIR EQUALIZE 0.678
# NDVI EQUALIZE 0.6536
# EQUALIZE ALL 0.5470

# 2 channels: red & nir : 0.6315

model.eval()
score = classifier.test(test_loader)
print(f'F1 Score {score}')
