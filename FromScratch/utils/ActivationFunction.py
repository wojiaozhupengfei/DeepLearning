#!/user/bin/env python
# -*- coding: utf-8 -*-
#@Time     :2018/9/3/0003 19:44
#@Author   :zhupengfei
#@Site     :
#@File     :ActivationFunction.py
#@Software :PyCharm
#@descibe  ：
import numpy as np
import matplotlib.pylab as plt

# 阶跃函数 入参可以是数组
def step_function(x):
	y = x > 0 # y是bool类型的数组
	return y.astype(np.int)

#test step_function
# x = np.random.randn(3)
# y = step_function(x)
# print(x)
# print(y)
xx = np.arange(-5.0, 5.0, 1.0)
yy = step_function(xx)
# plt.plot(x, y)
# plt.ylim(-1.0, 2.0)
# plt.show()


# sigmoid 函数
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

#test sigmoid
# x = np.arange(-5.0, 5.0, 1.0)
# y = sigmoid(x)
# print(y)
# plt.plot(x, y)
# plt.plot(xx, yy)
# plt.show()

# relu
def rule(x):
	return np.maximum(0, x)

#test rule
# x = np.array([-1, 5])
# x = np.arange(-5.0, 5.0, 1.0)
# y = rule(x)
# print(y)

# 恒等函数
def indentity_function(x):
	return x

# softmax
def softmax(a):
	m = np.max(a)
	exp_a = np.exp(a - m)  #防止a过大，导致内存溢出，所以给数组中的每一个值都进行归一化
	exp_sum = np.sum(exp_a)
	y = exp_a/exp_sum
	return y

#test softmax
# x = np.arange(-5.0, 5.0, 1.0)
# y = softmax(x)
# print(y)