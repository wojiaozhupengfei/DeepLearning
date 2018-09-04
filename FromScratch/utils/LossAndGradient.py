#!/user/bin/env python
# -*- coding: utf-8 -*-
#@Time     :2018/9/4/0004 19:39
#@Author   :zhupengfei
#@Site     :
#@File     :LossAndGradient.py
#@Software :PyCharm
#@descibe  ：
import numpy as np
import sys


def mean_square_error(x, t):
	return np.sum(0.5*(x - t)**2)
#
t = np.random.randint(2, size=(1, 10))
# print(t)
y = np.random.rand(10)
# print(y)
#
# a = mean_square_error(x, t)
# print(a)

def cross_entropy_error(y, t):
	delta = 1e-7
	if y.ndim == 1:
		y = y.reshape(1, y.size)
		t = t.reshape(1, t.size)
	batch_size = y.shape[0]
	return -np.sum(t*np.log(y + delta))/ batch_size

# a = cross_entropy_error(y, t)
# print(a)

# 定义中心差分
def numerical_diff(f, x):
	h = 1e-4
	return (f(x+h) - f(x-h))/2.*h

def test_function(x):
	return 0.01*x**2 + 0.1*x

def function_2(x):
	return x[0]**2 + x[1]**2

# x = 4.
# a = numerical_diff(test_function, x)
# print(a)
def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)  #给梯度数组初始化为x的形状
	for idx in range(x.size):
		temp = x[idx] # 先把原值寄存
		x[idx] = temp + h #计算f（x+h）
		fxh1 = f(x)
		x[idx] = temp - h #计算 f(x - h)
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = temp
	return grad
# print(numerical_gradient(function_2, np.array([3.0, 4.0])))
# print(numerical_gradient(function_2, np.array([0.0, 2.0])))
# print(numerical_gradient(function_2, np.array([3.0, 0.0])))

# x = np.array([3., 4.])
# a = numerical_gradient(test_function, x)
# print(a)


# 学习率过大过小都不合适，过大的话会发散成一个很大的值，过小的话参数还没开始更新就结束了，不信你可以试试
def gradient_descent(f, init_x, lr = 0.1, step_num = 100):
	x = init_x
	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x -= lr * grad
	return x

x = np.array([-3., 4.])
a = gradient_descent(function_2, x)
print(a)