#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/10/0010 8:05
# @Author  : zhupengfei
# @Site    : 
# @File    : foreword.py
# @Software: PyCharm
import tensorflow as tf

#定义一个两个输入 隐层3个神经元 一个输出的简单网络
x = tf.constant([[0.7, 0.9]])
print(type(x), x.shape)
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    sess.run(init_op)
    result = sess.run(y)
    print(result)