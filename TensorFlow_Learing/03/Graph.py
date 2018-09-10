#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/10/0010 7:15
# @Author  : zhupengfei
# @Site    : 
# @File    : Graph.py
# @Software: PyCharm
import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable('v', shape=[1, 2], initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable('v', shape=[2, 3], initializer=tf.ones_initializer)

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))

a = tf.constant([1, 2], name='a')
b = tf.constant([3, 4], name='b')
with tf.Session() as sess:
    result = tf.add(a, b, 'add')
    add_result = sess.run(result)
    print('result is',result)
    print('add_result is', add_result)