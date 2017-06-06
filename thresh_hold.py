import numpy as np
import math as m

def loader():
	with open('./numpy_data/cross_entropy.npy', 'rb') as f:
		cross_entropy = np.load(f)
	with open('./numpy_data/labels.npy', 'rb') as f:
		labels = np.load(f)

	return cross_entropy, labels

def get_thresh_hold(feature, cross_entropy, labels, beta=2, density=1000, rnge = 10):
	data_size = np.shape(cross_entropy)[0]

	lowest_score = 1000000

	for i in range(density):
		thresh_hold = m.e**((i-density/2) / (density/rnge))

		tp = 0
		fp = 0
		fn = 0

		for j in range(data_size):

			if (cross_entropy[j, feature] >= thresh_hold) and (labels[j, feature] == 1):
				tp += 1
			if (cross_entropy[j, feature] < thresh_hold) and (labels[j, feature] == 1):
				fn += 1
			if (cross_entropy[j, feature] >= thresh_hold) and (labels[j, feature] == 0):
				fn += 1

		p = tp / (tp + fp)
		r = tp / (tp + fn)

		score = (1 + beta**2) * (p*r) / (beta**2*p + r)

		if score < lowest_score:
			lowest_score = score
			best_thresh_hold = thresh_hold

		if i%10 == 0:
			print(i)

	return best_thresh_hold, lowest_score



cross_entropy, labels = loader()

thresh_hold, lowest_score = get_thresh_hold(0, cross_entropy, labels)

print('')
print(thresh_hold, lowest_score)
print('')