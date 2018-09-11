#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/10/0010 16:17
# @Author  : zhupengfei
# @Site    : 
# @File    : 3.4.5.py
# @Software: PyCharm
# 建立一个简单的二分类网络，实现训练过程，该网络还是输入两个神经元（[[x1, x2]]）， 中间三个神经元,输出一个神经元

import tensorflow as tf
from numpy.random import RandomState  #随机输入数据
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(x):
    """用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

# 定义网络模型
batch_size = 8
train_iter = 5000
lr = 0.001
x = tf.placeholder(tf.float32, [None, 2], 'x_input')
y_ = tf.placeholder(tf.float32, [None, 1], 'y_input')

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数
y = tf.sigmoid(y)  #这个网络针对的是二分类任务
# 这里的tf.clip_by_value是为了防止出现log0或者y>1的特殊情况
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y_)*tf.log(tf.clip_by_value((1-y), 1e-10, 1.0)))

# 定义优化器
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# 获取数据
dataset_size = 128
rdm = RandomState(1)
X = rdm.rand(dataset_size, 2)
Y = [[(x1 + x2 < 1)] for (x1, x2) in X]

train_loss = []
xx = np.arange(10)
#开始训练
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op) #上来二话不说先初始化
    print(sess.run(w1)) #看看开始的w的值
    print(sess.run(w2))
    for i in range(train_iter):
        # 每次训练选取batch_size个数据
        start =i*batch_size % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        #隔一段就打印损失
        if i % 500 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            train_loss.append(total_cross_entropy)
            print('train_loss is:', train_loss)
            print('After %d training steps, cross entropy on all data is %g'%(i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))

plt.plot(xx, train_loss)
plt.xlim(0, 10)
plt.ylim(0, 2)
plt.show()