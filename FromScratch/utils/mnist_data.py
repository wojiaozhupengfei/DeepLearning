#!/user/bin/env python
# -*- coding: utf-8 -*-
#@Time     :2018/9/3/0003 21:33
#@Author   :zhupengfei
#@Site     :
#@File     :mnist_data.py
#@Software :PyCharm
#@descibe  ：
import sys, os
sys.path.append(os.pardir)  # 这个相当于include
from data.mnist import load_mnist
from PIL import Image  # python内置的图像处理模块
import numpy as np
import pickle
from utils.ActivationFunction import *

(x_train, y_train), (x_test, y_test) = load_mnist(one_hot_label=False, flatten=True, normalize=False)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# 显示图像
def image_show(img):
	image = Image.fromarray(np.uint8(img))
	image.show()

# x = x_train[0] # 取出一张图片
# print(x.shape)
# x = x.reshape(28, 28) # 因为是黑白图像，所以这样还原
# print(x.shape)
# image_show(x)

def get_data():
	(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test

# 取得模型是pickle库中已经训练好的 三层网络,激活函数是sigmoid 最后的分类函数是softmax
def init_network():
	with open('sample_weigth.pkl', 'rb') as f:
		network = pickle.load(f)
	return network

# 其实就是network的前向计算
def predict(network, x):
	w1, w2, w3 = network['w1'], network['w2'], network['w3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']
	a1 = np.dot(x, w1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, w2) + b2
	z2 = sigmoid(a2)
	z3 = np.dot(z2, w3) + b3
	y = softmax(z3)

	return y

# 1 获取数据 2 获取模型 3 预测 4 计算准确率
x, t = get_data()
batch_size = 100
acc_count = 0

network = init_network()
for i in range(0, len(x), batch_size):
	x = x[i:batch_size + i]
	pred_y = predict(network, x)
	p = np.argmax(pred_y, aixs = 1)
	acc_count += np.sum(p == t[i:i+batch_size])
print('acc is %.5f'%(acc_count / len(x)))
