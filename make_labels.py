import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.misc
import sys
import time


# Constants

data_size = 40479
classes = {
	'haze': 0,
	'primary': 1,
	'agriculture': 2,
	'clear': 3,
	'water': 4,
	'habitation': 5,
	'road': 6,
	'cultivation': 7,
	'slash_burn': 8,
	'cloudy': 9,
	'partly_cloudy': 10,
	'conventional_mine': 11,
	'bare_ground': 12,
	'artisinal_mine': 13,
	'blooming': 14,
	'selective_logging': 15,
	'blow_down': 16
}

print(len(classes))

# Load labels into array

labels = np.zeros([data_size, 17], dtype=np.float32)

filepath = '../train_v2.csv'
file = open(filepath)

for i in range(data_size+1):
	line = file.readline()
	line = line.split(',')

	if line[-1][-1] == '\n':
		line[-1] = line[-1][:-1]

	tags = line[-1]
	tags = tags.split(' ')

	if i != 0:
		for tag in tags:
			labels[i-1, classes[tag]] = 1

with open('../numpy_data/labels.npy', 'wb') as f:
	np.save(f, labels)



print('\nLoaded labels array...\n')