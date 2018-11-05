#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/21/0021 11:12
# @Author  : zhupengfei
# @Site    : 
# @File    : test1.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from log import *

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def _make_example(pixels, label, image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example

mnist = input_data.read_data_sets('../../dataset/mnist', dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
n_examples = mnist.train.num_examples

with tf.python_io.TFRecordWriter('output.tfrecords') as writer:
    for index in range(n_examples):
        example = _make_example(pixels, labels[index], images[index])
        writer.write(example.SerializeToString())
print("TFRecord训练文件已保存。")

# 读取mnist测试数据。
images_test = mnist.test.images
labels_test = mnist.test.labels
pixels_test = images_test.shape[1]
num_examples_test = mnist.test.num_examples

# 输出包含测试数据的TFRecord文件。
with tf.python_io.TFRecordWriter("output_test.tfrecords") as writer:
    for index in range(num_examples_test):
        example = _make_example(pixels_test, labels_test[index], images_test[index])
        writer.write(example.SerializeToString())
print("TFRecord测试文件已保存。")

reader = tf.TFRecordReader()
file_name_queue = tf.train.string_input_producer(['output.tfrecords'])
_, serialized_example = reader.read(file_name_queue)

features = tf.parse_single_example(serialized_example, features={
    'image_raw':tf.FixedLenFeature([], tf.string),
    'label':tf.FixedLenFeature([], tf.int64),
    'pixels':tf.FixedLenFeature([], tf.int64)
})

images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = features['label']
pixels = features['pixels']

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    for i in range(10):
        image, label, pixel = sess.run([images, labels, pixels])
        logger.info(image, label, pixel)

    coord.request_stop()
    coord.join(threads)