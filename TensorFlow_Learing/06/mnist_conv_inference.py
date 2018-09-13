#!/user/bin/env python
# -*- coding: utf-8 -*-
#@Time     :2018/9/13/0013 18:52
#@Author   :zhupengfei
#@Site     :
#@File     :mnist_conv_inference.py
#@Software :PyCharm
#@descibe  ：LeNet-5网络
import tensorflow as tf

#配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层神经网络的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
#第二层神经网络的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
#全连接层的节点个数
FC_SIZE = 512

#LeNet的训练过程的前向传播中加入dropout,测试过程并不加入dropout
def inference(input_tensor, regularizer, train):
	#第一层卷积,输入尺寸28*28*1， 卷积核5*5*32 步长1 ， SAME填充，输入通道1 输出通道32，得到尺寸28*28*32
	with tf.get_variable_scope('layer-conv1'):
		conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		conv1_bias = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
		#卷积计算
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

	#池化层，输入尺寸28*28*32， 池化的卷积核2*2， 步长2 SAME填充， 输出为14*14*32
	with tf.get_variable_scope('layer-pooling1'):
		pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#第二层卷积，输入尺寸14*14*32，卷积核5*5*64，步长1，SAME填充， 输出为14*14*64
	with tf.get_variable('layer-conv2'):
		conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_bias = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
		#计算第二层卷积
		conv2 = tf.nn.conv2d(pool1, filter = conv2_weights, strides = [1, 1, 1, 1], padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))

	#池化层，输入尺寸14*14*64， 池化卷积核2*2， 步长2， SAME填充， 输出为7*7*64
	with tf.get_variable('layer-pooling2'):
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#第一层全连接，要将卷积后的featuremap拉直
	pool_shape = pool2.get_shape().as_list()
	#batch_size
	batch_size = pool_shape[0]
	#节点数是7*7*64拉直
	node = pool_shape[1]*pool_shape[2]*pool_shape[3]
	reshaped_pool2 = tf.reshape(pool2, [batch_size, node])

	#第一层全连接
	with tf.get_variable_scope('layer-fc1'):
		fc1_weights = tf.get_variable('weight', [node, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc1_bias = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(reshaped_pool2, fc1_weights) + fc1_bias)
		#全连接的w才会加入正则
		if regularizer!=None:
			tf.add_to_collection('losses',regularizer(fc1_weights))
		#如果在训练，则加入dropout
		if train:
			fc1 = tf.nn.dropout(fc1, 0.5)

	#第二层的全连接
	with tf.get_variable_scope('layer-fc2'):
		fc2_weights = tf.get_variable('weight', [FC_SIZE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc2_bias = tf.get_variable('bias', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(fc1, fc2_weights) + fc2_bias
		if regularizer != None:
			tf.add_to_collection('losses', regularizer(fc2_weights))

	return logit
