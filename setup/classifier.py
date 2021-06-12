import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import DiceCELoss
import torch.optim as optim
import numpy as np
import time

class WeedClassifier():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = DiceCELoss()
        self.lb_arr = np.asarray([
            [0, 0, 0], # 0 for background
            [0, 255, 0], # 1 for plants
            [255, 0, 0]    # 2 for weeds
        ])


    def train(self, trainLoader, validLoader, learning_rate=0.001, epochs=20, name="state_dict_model"):
        last_loss = 1000

        dataLoader = {
            'train': trainLoader,
            'valid': validLoader
        }

        history = {
            'train': list(),
            'valid': list()
        }

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.5)
        print('Starting...')

        for epoch in range(epochs):

            print("\nEpoch {}/{}:".format(epoch+1, epochs))
            epoch_time = time.time()

            for phase in ['train', 'valid']:
                epoch_loss, iteration = 0, 0

                if phase == 'train':
                    self.scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()

                for data in dataLoader[phase]:
                    iteration+=1
                    image = data['image'].to(self.device)
                    mask = data['mask'].to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model(image)

                        loss_val = self.criterion(output, mask)
                        if phase == 'train':
                            loss_val.backward()
                            self.optimizer.step()

                    epoch_loss += loss_val.item()

                epoch_loss /= (iteration * dataLoader[phase].batch_size)
                history[phase].append(epoch_loss)

                print('{} Loss:{:.7f}'.format(phase, epoch_loss))
                if phase == 'valid' and last_loss > epoch_loss:
                    if last_loss != 1000:
                        torch.save(self.model.state_dict(), name + '.pt')
                        print('Saved')
                    last_loss = epoch_loss

            end = time.time() - epoch_time
            m = end//60
            s = end - m*60
            print("Time {:.0f}m {:.0f}s".format(m, s))

        import json
        history_file = open(name + '.json', "w")
        json.dump(history, history_file)
        history_file.close()


    def test(self, testLoader):
        self.model.eval()
        test_data_indexes = testLoader.sampler.indices[:]
        data_len = len(test_data_indexes)
        mean_val_score = 0

        batch_size = testLoader.batch_size
        if batch_size != 1:
            raise Exception("Set batch size to 1 for testing purpose")
        testLoader = iter(testLoader)
        while len(test_data_indexes) != 0:
            data = testLoader.next()
            index = int(data['index'])
            if index in test_data_indexes:
                test_data_indexes.remove(index)
            else:
                continue

            image = data['image'].view((-1, 4, 512, 512)).to(self.device)
            mask = data['mask']

            output = self.model(image).cpu()

            # mean_val_score += self._dice_coefficient(output, mask)
            mean_val_score += self.miou(output, mask)

        mean_val_score = mean_val_score / data_len
        return mean_val_score

    def _dice_coefficient(self, predicted, target):
        # predicted = F.softmax(predicted, dim=1)
        predicted = torch.argmax(predicted, dim=1)

        predicted = predicted.view(-1)
        target = target.view(-1)

        smooth = 1
        coef_list = list()

        for sem_class in range(self.lb_arr.shape[0]):
            pred_inds = (predicted == sem_class)
            target_inds = (target == sem_class)

            intersection = (pred_inds[target_inds]).long().sum().item()
            union = pred_inds.long().sum().item() + target_inds.long().sum().item()
            coefficient = (2*intersection + smooth) / (union + smooth)
            coef_list.append(coefficient)

        return np.mean(coef_list)

    def miou(self, predicted, target):
        # predicted = F.softmax(predicted, dim=1)
        predicted = torch.argmax(predicted, dim=1)

        predicted = predicted.view(-1)
        target = target.view(-1)

        smooth = 1
        coef_list = list()

        for sem_class in range(self.lb_arr.shape[0]):
            pred_inds = (predicted == sem_class)
            target_inds = (target == sem_class)

            intersection = (pred_inds[target_inds]).long().sum().item()
            union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
            coefficient = (intersection + smooth) / (union + smooth)
            coef_list.append(coefficient)

        return np.mean(coef_list)

    def predict(self, data):
        self.model.eval()
        # image = data['image'].numpy()
        # mask = data['mask'].numpy()

        # image_tensor = torch.Tensor(data['image'])
        # output = self.model(image_tensor).detach().cpu()
        # output = (output > threshold)
        # output = output.numpy()

        image = data['image']
        mask = data['mask']
        rgb = data['rgb']

        image = image.view((-1, 4, 512, 512)).to(self.device)

        output = self.model(image)
        # score = self._dice_coefficient(output, mask)
        score = self.miou(output, mask)

        # output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)

        # image = image.numpy()
        output = self.decode_segmap(output)
        mask = self.decode_segmap(mask)

        # image = np.resize(image, (512, 512, 3))
        mask = np.resize(mask, (512, 512, 3))
        output = np.resize(output, (512, 512, 3))
        return rgb, mask, output, score

    def decode_segmap(self, mask):
        mask = mask.detach().cpu().clone().numpy()
        # mask = mask.numpy()
        mask = np.resize(mask, (512, 512))
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()
        for ll in range(self.lb_arr.shape[0]):
            r[mask == ll] = self.lb_arr[ll, 0]
            g[mask == ll] = self.lb_arr[ll, 1]
            b[mask == ll] = self.lb_arr[ll, 2]
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
