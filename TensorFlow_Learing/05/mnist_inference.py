#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/13/0013 9:03
# @Author  : zhupengfei
# @Site    : 
# @File    : mnist_inference.py
# @Software: PyCharm
#定义了计算前向传播的函数和变量管理

import tensorflow as tf

#定义一些全局变量
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: #如果加入正则项
        tf.add_to_collection('losses', regularizer(weights))
    return weights

#前向传播的计算
def inference(input_tensor, regularizer):
    with tf.variable_scope('layers1'):
        w1 = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        bias1 = tf.get_variable('bias1', [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        layers1_out = tf.nn.relu(tf.matmul(input_tensor, w1) + bias1)
    with tf.variable_scope('layers2'):
        w2 = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        bias2 = tf.get_variable('bias2', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layers2_out = tf.matmul(layers1_out, w2) + bias2
    return layers2_out

