import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

image = cv2.imread('../train-tif-v2/train_10.tif', -1)

image = image // 256

print(np.shape(image))
print(image.dtype)

plt.imshow(image)
plt.show()