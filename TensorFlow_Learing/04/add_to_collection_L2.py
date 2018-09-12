#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/11/0011 14:29
# @Author  : zhupengfei
# @Site    : 
# @File    : add_to_collection_L2.py
# @Software: PyCharm

#定义一个输入两个神经元，输出一个，5层的神经网络，熟悉add_to_collection函数
#这个程序有问题，loss竟然temd  你在多跑几次看看 ，一次一次都不一样，太奇怪了，w是按照标准正太分布来初始化的，有影响吗？


import tensorflow as tf
from numpy.random import RandomState
import numpy as np
import matplotlib.pyplot as plt

#该函数获取参数，并将参数的L2正则项加入到集合,其中lamdb是正则项系数
def get_weights(shape, lambd):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))  #将weights的L2正则项加入到集合
    return var

#定义全局参数
batch_size = 8
lambd = 0.001
dataset_size = 128
display_step = 50
train_iter = 5000
lr = 0.01
total_loss_list = []
# xx = np.arange(train_iter/display_step)
xx = np.arange(0, 20, 0.2)
# 定义输入
x = tf.placeholder(tf.float32, [None, 2], 'x_input')
y_ = tf.placeholder(tf.float32, [None, 1], 'y_input')

rdm = RandomState(1)
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0-0.05]for (x1, x2) in X]


#定义模型结构
layers_dim = [2, 10, 10, 10, 1] #每层有多少神经元
n_layers = len(layers_dim)#一共5层网络
input_dim = layers_dim[0] #第一层的输入2个神经元
cur_layer = x #当前层
#通过循环完成5层全连接神经网络
for i in range(1, n_layers):
    out_dim = layers_dim[i]
    weights = get_weights([input_dim, out_dim], lambd)
    bias = tf.constant(0.1, dtype=tf.float32, shape=[out_dim])
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weights) + bias) #激活后的输出
    input_dim = layers_dim[i]

#定义损失函数
mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer)) #计算均方误差
tf.add_to_collection('losses', mes_loss) #将均方误差加入集合
loss = tf.add_n(tf.get_collection('losses'))
#定义优化器
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
init_op = tf.global_variables_initializer()


#开始训练
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(train_iter):
        start = i*batch_size % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % display_step == 0:
            total_loss = sess.run(loss, feed_dict={x:X[start:end], y_:Y[start:end]})
            total_loss_list.append(total_loss)
            print('train iter is: %d, and loss is: %g '%(i, total_loss))

plt.plot(xx, total_loss_list)
plt.xlim(0, display_step)
plt.ylim(0, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


