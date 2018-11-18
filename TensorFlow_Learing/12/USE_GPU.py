#!-*- coding:utf-8 -*-
# author:zhupengfei
# datetime:2018/11/16 19:07
# software:PyCharm
# discribe:
import tensorflow as tf

# a = tf.constant([1., 2., 3.], shape=[3], name = 'a')
# b = tf.constant([1., 2., 3.], shape=[3], name = 'b')
# c = a+b
# #通过log_device_placement参数来输出运行每一个运算设备
# sess = tf.Session(config=tf.ConfigProto(log_device_placement = True))
# print(sess.run(c))

#通过tf.device将运算制定到特定的设备上
# with tf.device('/cpu:0'):
#     a = tf.constant([1., 2., 3.], shape=[3], name='a')
#     b = tf.constant([2., 3., 4.], shape=[3], name='b')
#
# with tf.device('/gpu:0'):
#     c = a+b
#
# with tf.Session(config=tf.ConfigProto(log_device_placement = True)) as sess:
#     print(sess.run(c))

#有些操作不能再GPU上执行，比如下面的整形var，gpu不支持，所以为了防止此类错误，调用参数allow_soft_placement使得gpu不能运行时就在cpu上运行
with tf.device('/gpu:0'):
    a_gpu = tf.Variable(0, name='gpu_0')

with tf.Session(config=tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)) as sess:
    sess.run(tf.initialize_all_variables())
