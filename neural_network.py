import numpy as np
import tensorflow as tf

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
	return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def neural_net(x, y, keep_prob):

	# conv layers varibles
	W_conv1 = weight_variable([5, 5, 4, 64])
	b_conv1 = bias_variable([64])
	W_conv2 = weight_variable([5, 5, 64, 64])
	b_conv2 = bias_variable([64])
	W_conv3 = weight_variable([5, 5, 64, 64])
	b_conv3 = bias_variable([64])

	# fully connected varibles
	W_fc1 = weight_variable([8 * 8 * 64, 1024])
	b_fc1 = bias_variable([1024])
	W_fc2 = weight_variable([1024, 2])
	b_fc2 = bias_variable([2])

	# 64x64 to 32x32, 64 filters
	h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	# 32x32 to 16x16, 64 filters
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# 16x16 to 8x8, 64 filters
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
	h_pool3 = max_pool_2x2(h_conv3)

	# 8*8*64 to 1024, fully connected
	h_pool_flat = tf.reshape(h_pool3, [-1, 8*8*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

 	# 1024 layer dropout
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# 1024 to 2, fully conncected
	y_ = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	return y_