#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/28/0028 14:06
# @Author  : zhupengfei
# @Site    : 
# @File    : alexnettest.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # 导入tf库的手写字体识别的dataset
import numpy as np
import os

# 1.导入数据集
mnist = input_data.read_data_sets('./data//', one_hot=True)

# 2.定义全局参数
lr = 0.0001
train_iters = 10000
batch_size = 16
input_dim = 784  # 28*28*1 的单通道图像，拉伸成一维
mnist_class = 10
dropout = 0.5
display_step = 1

x = tf.placeholder(tf.float32, [None, input_dim])
y = tf.placeholder(tf.float32, [None, mnist_class])

# 3.定义函数，包括 卷积，池化，正则， alexnet
# 卷积
def conv2d(name, input_data, filter, bias):
    x = tf.nn.conv2d(input_data, filter=filter, strides=[1, 1, 1, 1], padding='SAME', name=None, use_cudnn_on_gpu=False, data_format='NHWC')
    x = tf.nn.bias_add(x, bias, data_format=None, name=None)
    return tf.nn.relu(x, name=name)
# 池化， ksize的数据结构[batche_number， k, k, channel]
def max_pooling(name, input_data, k):
    return tf.nn.max_pool(input_data, ksize=[1, k, k, 1],strides=[1, k, k, 1], padding='SAME', name=name)


# lrn  局部响应归一化
def norm(name, input_data, lsize = 4):
    return tf.nn.lrn(input_data, depth_radius=lsize, bias=1, alpha=1, beta=0.5, name=name)

# 定义网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 48])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 48, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 192])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 192, 192])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 192, 128])),
    'wfc1': tf.Variable(tf.random_normal([4*4*128, 4096])),
    'wfc2': tf.Variable(tf.random_normal([4096, 4096])),
    'wfc3': tf.Variable(tf.random_normal([4096, mnist_class]))
           }
# 定义偏移
bias = {
    'bwc1': tf.Variable(tf.random_normal([48])),
    'bwc2': tf.Variable(tf.random_normal([128])),
    'bwc3': tf.Variable(tf.random_normal([192])),
    'bwc4': tf.Variable(tf.random_normal([192])),
    'bwc5': tf.Variable(tf.random_normal([128])),
    'bwfc1': tf.Variable(tf.random_normal([4096])),
    'bwfc2': tf.Variable(tf.random_normal([4096])),
    'bwfc3': tf.Variable(tf.random_normal([mnist_class]))
}

# alexnet
def AlexNet(input_image, weights, bias, dropout):
    input_image = tf.reshape(input_image, shape=[-1, 28, 28, 1])

    # conv1
    conv1 = conv2d('conv1', input_image, weights['wc1'], bias['bwc1'])
    pooling1 = max_pooling('pooling1', conv1, k = 2) # 第一层输出要经过池化
    norm1 = norm('norm1', pooling1, lsize=4)

    # conv2
    conv2 = conv2d('conv2', norm1, weights['wc2'], bias['bwc2'])
    pooling2 = max_pooling('pooling2', conv2, k=2)  # 第二层输出要经过池化
    norm2 = norm('norm2', pooling2, lsize=4)

    # conv3
    conv3 = conv2d('conv3', norm2, weights['wc3'], bias['bwc3'])
    norm3 = norm('norm3', conv3, lsize=4)

    # conv4
    conv4 = conv2d('conv4', norm3, weights['wc4'], bias['bwc4'])
    norm4 = norm('norm4', conv4, lsize=4)

    # conv5
    conv5 = conv2d('conv5', norm4, weights['wc5'], bias['bwc5'])
    pooling5 = max_pooling('pooling5', conv5, k=2) # 第五层输出要经过池化
    norm5 = norm('norm5', pooling5, lsize=4)

    # fc1
    fc1_input = tf.reshape(norm5, shape=[-1, weights['wfc1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(fc1_input, weights['wfc1']) + bias['bwfc1'], name='fc1')
    dense1 = tf.nn.dropout(dense1, keep_prob=dropout) # 随机失活，防止过拟合, 增加准确率

    # fc2
    fc2_input = tf.reshape(dense1, shape=[-1, weights['wfc2'].get_shape().as_list()[0]])
    dense2 = tf.nn.relu(tf.matmul(fc2_input, weights['wfc2']) + bias['bwfc2'], name='fc2')
    dense2 = tf.nn.dropout(dense2, keep_prob=dropout)

    #out
    out = tf.matmul(dense2, weights['wfc3']) + bias['bwfc3']
    return out

# 构建模型, 包括 1.输出预测，2.学习率动态下降 3.损失函数 4.优化器 5.准确率
# 输出预测
pred = AlexNet(x, weights, bias, dropout=dropout)

# 学习率,计算公式 lr = lr * decay_rate^(global_step/decay_step)
# global_step = tf.constant(0, tf.int64)
# decay_rate = tf.constant(0.9, tf.float64)
# learn_rate = tf.train.exponential_decay(lr, global_step, decay_steps=10000, decay_rate = decay_rate)
#
# # 交叉熵损失函数
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
#
# # 优化器
# optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)
#
# # 准确率
# acc_tf = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#
# acc = tf.reduce_mean(tf.cast(acc_tf, tf.float32))

