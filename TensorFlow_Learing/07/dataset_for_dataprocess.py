#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/21/0021 7:50
# @Author  : zhupengfei
# @Site    : 
# @File    : dataset_for_dataprocess.py
# @Software: PyCharm
#使用数据集的方式来读取数据，进行数据处理，上一节是按照文件队列对数据进行多线程的处理

from log import *
import tensorflow as tf
import numpy as np

def parser(record):
    features = tf.parse_single_example(record, features={
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64)
    })
    return features['pixels'], features['label'], features['image_raw'], features['height'], features['width']

# #1.构造数据集
# input_files = ['./output.tfrecords']
# dataset = tf.data.TFRecordDataset(input_files)
# #因为取出来的TFRecord数据集是二进制，要对二进制文件进行解析
# dataset = dataset.map(parser)
#
# #2.定义遍历数据的迭代器
# iterator = dataset.make_one_shot_iterator()
#
# #3.用迭代器取数据
# pixels, label, image_raw, height, width = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(30):
#         pixel, lab, img, h, w = sess.run([pixels, label, image_raw, height, width])
#         logger.info('Step %d is %s'%(i, pixel))
#         logger.info('Step %d is %s'%(i, lab))
#         logger.info('Step %d is %s'%(i, img))
#         logger.info('Step %d is %s'%(i, h))
#         logger.info('Step %d is %s \n'%(i, w))

#利用make_initializabal_iterator建立迭代器
#1.构建数据集
input_files = tf.placeholder(tf.string)#这里先不传入文件地址，在会话中传入
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parser)

#2.定义对应的迭代器
iterator = dataset.make_initializable_iterator()

#3.迭代器取数据
pixels, label, image_raw, height, width = iterator.get_next()

#4.创建会话
with tf.Session() as sess:
    #make_initializable_iterator定义的迭代器需要首先初始化，并feed给文件路径
    sess.run(iterator.initializer, feed_dict={input_files:['./output.tfrecords', './output_test.tfrecords']})

    #遍历所有数据的一个epoch，完成之后抛出OutOfRangeError
    while True:
        try:
            pixel, lab, img, h, w = sess.run([pixels, label, image_raw, height, width])
            logger.info('is %s' % (pixel))
            logger.info('is %s' % (lab))
            logger.info('is %s' % (img))
            logger.info('is %s' % (h))
            logger.info('is %s \n' % (w))
        except tf.errors.OutOfRangeError:
            break
