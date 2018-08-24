#!/user/bin/env python
# -*- coding: utf-8 -*-
#@Time     :2018/8/22/0022 20:55
#@Author   :zhupengfei
#@Site     :hangzhou
#@File     :vgg11_test.py
#@Software :PyCharm
#@describe : 采用训练好的数据，观察每一层特征图，并没有FC
import scipy.io  # 加载和保存mat格式的文件
import scipy.misc # 图像预处理模块
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

pd.set_option('display.max_columns', None) #设置显示数据最大列
pd.set_option('display.max_rows', None) #设置显示数据最大行
pd.set_option('display.width', 1000)

# 使用ImageNet训练好的参数，大概看看参数的布局
# data = scipy.io.loadmat('./data/imagenet-vgg-verydeep-19.mat')
# print(data)

# 卷积层
def _conv_layer(input, weights, bias):
	result = tf.nn.conv2d(input, tf.constant(weights), [1, 1, 1, 1], padding='SAME')
	return tf.nn.bias_add(result, bias)

# 池化层
def pool_layer(input):
	result = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME') # 这里的ksize和strides前后都是1，固定语法格式
	return result

# 均值预处理
def preprocess(input, mean_value):
	return input - mean_value

# 还原均值处理后的图像
def unpreprocess(input, mean_value):
	return input + mean_value

# 读取图片
def imread(path):
	return scipy.misc.imread(path).astype(np.float)

# 保存图片
def imsave(image, path):
	im = np.clip(image, 0, 255).astype(np.uint8)
	return scipy.misc.imsave(path, im)

def net(data_path, input_image):
	layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
	          'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
	          'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
	          'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
	          'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4')
	data = scipy.io.loadmat(data_path) # 数据很复杂，需要在jupyter中单独仔细看看数据的存放结构
	mean = data['normalization'][0][0][0] #存放图片的均值 shape = 224 * 224 * 3
	mean_pixel = np.mean(mean, axis=(0, 1)) #对 0， 1 轴取均值作为图片均值化的值 需要返回
	net = {} #返回的字典，包括每一层的特征图， mean_pixel, layers
	current = input_image
	weights = data['layers'][0] #这个结构存放每一层卷积的权重
	for i, name in enumerate(layers):
		kind = name #对应layers中不同的操作，在weights中是分开存放的
		if kind[:4] == 'conv': #卷积层
			kernels , bias = weights[i][0][0][0][0] #这个结构存放了可以用的kernels和bias shape = 1 *2 如果weights[0][0][0][0][0][0]就到了kernel
			# 这里注意，因为mat文件个是和tensorflow的格式有所不同
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
			kernels = np.transpose(kernels, (1, 0, 2, 3))
			bias = np.reshape(bias, -1) # 变为列向量
			current = _conv_layer(current, kernels, bias) # 计算卷积层
		elif kind[:4] == 'relu': # 激活
			current = tf.nn.relu(current)
		elif kind[:4] == 'pool': # 池化层
			current == pool_layer(current)
		net[name] = current # 将每一步结果存入字典

	assert len(layers) == len(net)
	return net, mean_pixel, layers

cwd = os.getcwd() # 返回当前工作目录
Data_Path = cwd + '/data/imagenet-vgg-verydeep-19.mat'
Image_Path = cwd + '/girl.jpg'

input_image = imread(Image_Path) #shape = 854 * 1280 * 3
# print(input_image.shape)
shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2])

with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess: # 如果指定的设备不存在，允许tf自动调用设备
	image = tf.placeholder('float', shape = shape)
	nets, mean_pixel , layers = net(Data_Path, image)
	preimage = np.array([preprocess(input_image, mean_pixel)]) # 数据预处理，零均值化
	for i, layer in enumerate(layers):
		print('[%d/%d] %s'%(i+1, len(layers), layer))
		features = nets[layer].eval(feed_dict = {image:preimage})
		print('type of features is :', type(features))
		print('shape of features is : ', features.shape)
		if 1:
			plt.figure(i + 1, figsize=(10, 5))
			plt.matshow(features[0, :, :, 0], fignum= i + 1)
			plt.title(' ' + layer)
			plt.colorbar()
			plt.ion() #交互环境开启，不用手动关掉图片就能显示下一张
			plt.pause(2)
			plt.close()
			plt.show()
