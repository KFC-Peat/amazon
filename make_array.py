import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.misc
import sys
import cv2
import time

print('\n')

# Constants

img_num = 40479
img_size = 64


def down_sample(image):
	image_ds = np.zeros([64,64,4], dtype=np.uint8)

	x = [0,0,0,0]

	for i in range(64):
		for j in range(64):
			x[0] = sum(sum(image[4*i:4*(i+1), 4*j:4*(j+1), 0])) // 256 // 4**2
			x[1] = sum(sum(image[4*i:4*(i+1), 4*j:4*(j+1), 1])) // 256 // 4**2
			x[2] = sum(sum(image[4*i:4*(i+1), 4*j:4*(j+1), 2])) // 256 // 4**2
			x[3] = sum(sum(image[4*i:4*(i+1), 4*j:4*(j+1), 3])) // 256 // 4**2

			image_ds[i,j,0] = x[0]
			image_ds[i,j,1] = x[1]
			image_ds[i,j,2] = x[2]
			image_ds[i,j,3] = x[3]

	return image_ds


# Make Image Array

image_array = np.zeros([img_num,img_size,img_size,4], dtype=np.uint8)

for i in range(img_num):

	if i%100 == 0:
		print(i)

	fp = '../train-tif-v2/train_{}.tif'.format(i)
	image = cv2.imread(fp, -1)
	
	image_ds = down_sample(image)

	image_array[i,:,:,:] = image_ds


with open('../numpy_data/test_images.npy', 'wb') as f:
	np.save(f, image_array)