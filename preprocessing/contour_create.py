import cv2
import numpy as np

for name in range(125):
    img = cv2.imread('dataset/train/mask/{}.png'.format(name))
    green = img[:,:,1]
    if(green[green==255].sum()!=0):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((20,20), np.uint8)

        dilated = cv2.dilate(img, kernel, iterations=1)
        dilated = cv2.erode(dilated, np.ones((12,12), np.uint8), iterations=1)
        mask = np.zeros(dilated.shape, dtype=np.uint8)
        clone = mask.copy()

        _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if(area > 9000):
                cv2.drawContours(mask, contours, i, 255, -1)

                box = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(box))
                cv2.drawContours(clone, [box], -1, 255, -1)

        cv2.imwrite('dataset/train/augmentation/{}.png'.format(name), clone)
        # cv2.imshow('Img', cv2.resize(img, (600,600)))
        # cv2.imshow('Dilated', cv2.resize(dilated, (600,600)))
        # cv2.imshow('Mask', cv2.resize(mask, (600,600)))
        # cv2.imshow('clone', cv2.resize(clone, (600,600)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
