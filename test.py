import cv2

img = cv2.imread('dataset/test/mask/0.png')
img = cv2.resize(img, (512,512))
cv2.imshow('a', img)
cv2.waitKey(0)