import cv2
import numpy as np
import os

lb_arr = np.asarray([
            [0, 0, 0], # 0 for background
            [0, 255, 0], # 1 for plants
            [0, 0, 255]    # 2 for weeds
        ])

def decode_segmap(mask):
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()
    for ll in range(lb_arr.shape[0]):
        r[mask == ll] = lb_arr[ll, 0]
        g[mask == ll] = lb_arr[ll, 1]
        b[mask == ll] = lb_arr[ll, 2]
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

phases = ['train', 'test']

for phase in phases:
    path = os.path.join('dataset', phase, 'mask')
    for filename in os.listdir(path):
        filename = os.path.join(path, filename)
        mask = cv2.imread(filename)
        mask = mask[:,:,0]
        rgb = decode_segmap(mask)
        cv2.imwrite(filename, rgb)
