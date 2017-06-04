import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.misc
import sys
import tensorflow as tf
import time

from neural_network import neural_net



# Load the training data
def data_loader():
	# Load the training data
	with open('../numpy_data/train_images_64.npy', 'rb') as f:
		image_array = np.load(f)

	with open('../numpy_data/labels.npy', 'rb') as f:
		label_array = np.load(f)

	return image_array, label_array



# This function trains a binary neural network to determine whether a feature is present
def trainer(feature, image_array, label_array):

	# Get data dimentions
	img_num = np.shape(image_array)[0]
	img_size = np.shape(image_array)[1]



	# Initialise neural network

	sess = tf.InteractiveSession()

	x = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, 4])
	y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
	keep_prob = tf.placeholder(tf.float32)

	y_ = neural_net(x, y, keep_prob)

	print('Initialised neural network...\n')



	# More neural network initialisation

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver()
	filepath = '../models/one/feature_{}'.format(feature)

	print('Initialised neural network P2...\n')



	# Train the network

	print('Start neural network training for feature {}...\n'.format(feature))

	batch_size = 32
	epochs = 1000
	sess.run(tf.global_variables_initializer())

	yes_count = 0
	no_count = 0

	image_batch = np.zeros([batch_size, 64, 64, 4], dtype=np.uint8)
	label_batch = np.zeros([batch_size, 2], dtype=np.uint8)

	# Set the label batch to half true and half false
	for i in range(16):
		label_batch[i,1] = 1
	for i in range(16):
		label_batch[i+16,0] = 1

	for i in range(epochs):

		# Fill the first half of batch with true images
		yes = 0
		while yes < 16:
			yes_count = yes_count % img_num
			if label_array[yes_count, feature] == 1:
				image_batch[yes,:,:,:] = image_array[yes_count,:,:,:]
				yes += 1
			yes_count += 1


		# Fill the second half of batch with false images
		no = 0
		while no < 16:
			no_count = no_count % img_num
			if label_array[no_count, feature] == 0:
				image_batch[no+16,:,:,:] = image_array[no_count,:,:,:]
				no += 1
			no_count += 1


		if i%100 == 0:
			print(i, accuracy.eval(feed_dict={x: image_batch, y: label_batch, keep_prob: 1.0}))

		train_step.run(feed_dict={x: image_batch, y: label_batch, keep_prob: 0.5})


	# Training complete
	print('\nFinished training for feature {}...\n\n'.format(feature))

	saver.save(sess, filepath) # save neural net
	sess.close() # close tensorflow session




# Make a neural net for each feature

image_array, label_array = data_loader()

for i in range(17):
	trainer(i, image_array, label_array)
