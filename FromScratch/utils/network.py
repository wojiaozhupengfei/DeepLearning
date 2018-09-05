#!/user/bin/env python
# -*- coding: utf-8 -*-
#@Time     :2018/9/3/0003 20:23
#@Author   :zhupengfei
#@Site     :
#@File     :network.py
#@Software :PyCharm
#@descibe  ：实现一个小三层全连接神经网络
import numpy as np
# from ActivationFunction import indentity_function
from utils.ActivationFunction import *
from utils.LossAndGradient import *

# 创建一个单层网络
class simpltNet():
	def __init__(self):
		self.w = np.random.randn(2, 3)

	def predict(self, x):
		z = np.dot(x, self.w)
		return z

	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y, t)
		return loss

net = simpltNet()
# print(net.w)
x = np.array([0.6, 0.9])
p = net.predict(x)
print('yucezhi',p)
print(np.argmax(p))
t = np.array([0, 0, 1])
loss = net.loss(x, t)
print(loss)
f = lambda w:net.loss(x, t)
dw = numerical_gradient(f, net.w)
print(dw)

sys.exit()
# 1 初始化每一层的参数
# 输入2 隐层3+2 输出2
def init_network():
	network = {}
	network['w1'] = np.random.randn(2, 3)
	network['b1'] = np.random.randn(3)
	network['w2'] = np.random.randn(3, 2)
	network['b2'] = np.random.randn(2)
	network['w3'] = np.random.randn(2, 2)
	network['b3'] = np.random.randn(2)

	return network

# 2 计算 入参输入数据x和网络  输出最后一层的结果
def forword(network, x):
	w1, w2, w3 = network['w1'], network['w2'], network['w3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	#计算每一层的输出
	a1 = np.dot(x, w1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, w2) + b2
	z2 = sigmoid(a2)
	z3 = np.dot(z2, w3) + b3
	y = indentity_function(z3)

	return y

# network = init_network() # 初始化网络
# x = np.array([1, 5])
# y = forword(network, x)
# print(y)

