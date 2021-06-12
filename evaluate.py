import os
import torch
import numpy as np
import setup.dataset as dataset
import setup.ResUNet as ResUNet
import setup.classifier as classifier
from torch.utils.data import SubsetRandomSampler

np.random.seed(0)
torch.manual_seed(0)

DATASET_PATH = 'dataset'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def sampler_indices(length):
    indices = list(range(length))
    np.random.shuffle(indices)
    split = int(np.floor(0.1 * length))
    test_indices = indices[:split]
    return test_indices

tumor_dataset = dataset.WeedDataset(DATASET_PATH)

test_indices = sampler_indices(len(tumor_dataset))
test_sampler = SubsetRandomSampler(test_indices)

test_loader = torch.utils.data.DataLoader(tumor_dataset, batch_size=1, sampler=test_sampler)

FILTER_LIST = [16,32,64,128,256]

model = ResUNet.ResUNet(FILTER_LIST).to(device)
path = 'outputs/ResUNet.pt'

classifier = classifier.WeedClassifier(model, device)
if str(device) == 'cpu':
    classifier.model.load_state_dict(torch.load(path, map_location='cpu'))
else:
    classifier.model.load_state_dict(torch.load(path))

model.eval()
score = classifier.test(test_loader)
# print(f'\nDice Score {score}')
# Dice Score 0.8537711366007329

print(f'\n mIoU Score {score}')
# mIoU Score 0.7995237626468983
