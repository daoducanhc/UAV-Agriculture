import os
from random import random
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

TEST_DATASET_PATH = 'dataset/test'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

test_weed_dataset = dataset.WeedDataset(TEST_DATASET_PATH, random_rotate=False)

test_indices = list(range(len(test_weed_dataset)))
test_sampler = SubsetRandomSampler(test_indices)

test_loader = torch.utils.data.DataLoader(test_weed_dataset, batch_size=1, sampler=test_sampler)

model = ResUNet.ResUNet().to(device)
path = 'outputs/original_dataset/256/ResUNet.pt'

classifier = classifier.WeedClassifier(model, device)
if str(device) == 'cpu':
    classifier.model.load_state_dict(torch.load(path, map_location='cpu'))
else:
    classifier.model.load_state_dict(torch.load(path))

score = classifier.test(test_loader)
print(f'\n F1 Score {score}')
