import numpy as np
import math as m

def loader():
	with open('./numpy_data/cross_entropy.npy', 'rb') as f:
		cross_entropy = np.load(f)
	with open('./numpy_data/labels.npy', 'rb') as f:
		labels = np.load(f)

	return cross_entropy, labels

def get_thresh_hold(feature, cross_entropy, labels, beta=2, density=100, rnge = 10):
	data_size = np.shape(cross_entropy)[0]

	highest_score = 0
	best_thresh_hold = 0

	for i in range(density):
		thresh_hold = m.e**((i-density/2) / (density/rnge))

		tp = 0
		fp = 0
		fn = 0

		for j in range(data_size):

			if (cross_entropy[j, feature] > thresh_hold) and (labels[j, feature] == 1):
				tp += 1
			if (cross_entropy[j, feature] <= thresh_hold) and (labels[j, feature] == 1):
				fn += 1
			if (cross_entropy[j, feature] > thresh_hold) and (labels[j, feature] == 0):
				fp += 1

		try:
			p = tp / (tp + fp)
			r = tp / (tp + fn)

			score = (1 + beta**2) * (p*r) / (beta**2*p + r)
		except:
			score = 0

		if score > highest_score:
			highest_score = score
			best_thresh_hold = thresh_hold

		if i%10 == 0:
			print(feature, i)


	for i in range(2):

		if i == 0:
			thresh_hold = 0
		if i == 1:
			thresh_hold = 1000000000

		tp = 0
		fp = 0
		fn = 0

		for j in range(data_size):

			if (cross_entropy[j, feature] > thresh_hold) and (labels[j, feature] == 1):
				tp += 1
			if (cross_entropy[j, feature] <= thresh_hold) and (labels[j, feature] == 1):
				fn += 1
			if (cross_entropy[j, feature] > thresh_hold) and (labels[j, feature] == 0):
				fp += 1

		try:
			p = tp / (tp + fp)
			r = tp / (tp + fn)

			score = (1 + beta**2) * (p*r) / (beta**2*p + r)
		except:
			score = 0

		if score > highest_score:
			highest_score = score
			best_thresh_hold = thresh_hold


	return best_thresh_hold, highest_score



cross_entropy, labels = loader()

thresh_holds = np.zeros([17, 2])

for i in range(17):

	thresh_hold, highest_score = get_thresh_hold(i, cross_entropy, labels)

	thresh_holds[i, 0] = thresh_hold
	thresh_holds[i, 1] = highest_score

	print('')
	print(i, thresh_hold, highest_score)
	print('')

with open('./numpy_data/thresh_holds.npy', 'wb') as f:
	np.save(f, thresh_holds)