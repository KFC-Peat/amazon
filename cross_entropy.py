import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.misc
import sys
import tensorflow as tf
import time

from neural_network import neural_net



# load image data a previous cross entropy data
def loader():
	with open('./numpy_data/train_images_64.npy', 'rb') as f:
		image_array = np.load(f)

	try:
		with open('./numpy_data/cross_entropy.npy', 'rb') as f:
			ce_data = np.load(f)
	except:
		ce_data = np.zeros([np.shape(image_array)[0], 17], dtype=np.float32)

	print('Loaded data...\n')

	return image_array, ce_data




def cross_entropy(image_array, ce_data, feature):

	# Constants
	img_num = np.shape(image_array)[0]


	# Reload the neural net to classify input
	sess = tf.InteractiveSession()

	x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 4])
	y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
	keep_prob = tf.placeholder(tf.float32)

	y_ = neural_net(x, y, keep_prob)

	saver = tf.train.Saver()
	filepath = './models/feature_{}'.format(feature)
	saver.restore(sess, filepath)

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)

	print('\nLoaded neural network...\n')




	# Calculate cross entropy for each a feature on every input image

	print('Start cross entropy calculations for feature {}...\n'.format(feature))

	batch_size = 100
	zeros_onehot = np.zeros([batch_size,2], dtype=np.uint8)

	for i in range(batch_size):
		zeros_onehot[i,0] = 1

	for i in range(img_num // 100):
		ce = cross_entropy.eval(feed_dict={x: image_array[batch_size*i:batch_size*(i+1),:,:,:], y: zeros_onehot, keep_prob: 1.0})
		ce_data[batch_size*i:batch_size*(i+1),feature] = ce[:]

	ce = cross_entropy.eval(feed_dict={x: image_array[img_num-batch_size:img_num,:,:,:], y: zeros_onehot, keep_prob: 1.0})
	ce_data[img_num-batch_size:img_num,feature] = ce[:]

	print('Finish cross entropy calculations for feature {}...\n'.format(feature))

	sess.close() # close tensorflow session

	return ce_data



image_array, ce_data = loader()

ce_data = cross_entropy(image_array, ce_data, 9)

with open('./numpy_data/cross_entropy.npy', 'wb') as f:
	np.save(f, ce_data)