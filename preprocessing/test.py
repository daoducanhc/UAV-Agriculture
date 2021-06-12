from PIL import Image
import os
import cv2

red_name = os.path.join('dataset', 'train', 'red', '0.png')
nir_name = os.path.join('dataset', 'train', 'nir', '0.png')
ndvi_name = os.path.join('dataset', 'train', 'ndvi', '0.png')
mask_name = os.path.join('dataset', 'train', 'mask', '0.png')

red = Image.open(red_name)
nir = Image.open(nir_name)
ndvi = Image.open(ndvi_name)
mask = Image.open(mask_name)

image = Image.merge('RGB', (red,nir,ndvi))
image.show()
