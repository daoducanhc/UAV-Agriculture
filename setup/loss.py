import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()

    def forward(self, predicted, target, num_classes=3):

        # target's shape: batchsize, height, width
        # predicted's shape: batchsize, num_classes, height, width

        CE = F.cross_entropy(predicted, target)

        batch = predicted.size()[0]
        batch_loss = 0
        smooth = 1

        predicted = F.softmax(predicted, dim=1)
        # predicted = torch.argmax(predicted, dim=1)

        for index in range(batch):
            preds = predicted[index]
            tar = target[index]

            # predicted = predicted.view(-1)
            tar = tar.view(-1)

            coef_list = list()

            for sem_class in range(num_classes):
                pred = preds[sem_class]
                pred = pred.view(-1)
                # pred_inds = (predicted == sem_class)
                tar_inds = (tar == sem_class)
                # print(sem_class)
                # print(pred.shape)
                # print(tar_inds.shape)

                intersection = (pred[tar_inds]).long().sum().item()
                union = pred.long().sum().item() + tar_inds.long().sum().item()
                coefficient = (2*intersection + smooth) / (union + smooth)
                coef_list.append(coefficient)

            batch_loss += np.mean(coef_list)
        
        batch_loss = batch_loss / batch

        Dice_CE = CE + (1 - batch_loss)

        return Dice_CE

def test():
    pred = torch.rand((2, 3, 512, 512))
    target = torch.randint(3, (2, 512, 512))
    # print(pred.shape)
    # print(target.shape)
    # print(target[0])
    loss = DiceCELoss()
    print(loss(pred, target))

if __name__ == "__main__":
    test()