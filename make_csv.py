import numpy as np
import sys


# Constants
img_num = 61191
test_num = 40668
file_num = 20521


with open('./numpy_data/cross_entropy_test.npy', 'rb') as f:
	cross_entropy = np.load(f)

with open('./numpy_data/thresh_holds.npy', 'rb') as f:
	thresh_holds = np.load(f)

def array2string(ce, th):

	classes = [
		'haze',
		'primary',
		'agriculture',
		'clear',
		'water',
		'habitation',
		'road',
		'cultivation',
		'slash_burn',
		'cloudy',
		'partly_cloudy',
		'conventional_mine',
		'bare_ground',
		'artisinal_mine',
		'blooming',
		'selective_logging',
		'blow_down'
	]

	line = []
	for i in range(12):
		if ce[i] > th[i]:
			line.append(classes[i])

	string = ' '.join(line)

	return string

f = open('./numpy_data/test_results.csv', 'w')
f.write('image_name,tags\n')

for i in range(file_num+1):
	string = array2string(cross_entropy[i,:], thresh_holds[:,0])
	line = 'file_{},{}\n'.format(i, string)
	f.write(line)

for i in range(test_num+1):
	string = array2string(cross_entropy[i+file_num,:], thresh_holds[:,0])
	line = 'test_{},{}\n'.format(i, string)
	f.write(line)

f.close()