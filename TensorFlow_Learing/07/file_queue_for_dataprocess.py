#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20/0020 10:17
# @Author  : zhupengfei
# @Site    : 
# @File    : file_queue_for_dataprocess.py
# @Software: PyCharm
#一个完整的通过多线程处理数据的框架


from log import *
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys
import data_process
#注意不同文件夹的py文件的引用，先将对应文件夹加入进来，然后设置对应文件夹为根目录（右键文件目录，make direction as source root）
sys.path.append(r'../05')
import mnist_inference

# 一、将数据保存为TFRecord格式文件

#mnist数据集路径
dataset_path = '../dataset/mnist'

#定义变量转化函数
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


#将数据转化为tf.train.Example格式
def _make_example(mnist, index):
    image_train = mnist.train
    image_raw = image_train.images[index].tostring()
    example = tf.train.Example(features = tf.train.Features(feature = {
        'pixels':_int64_feature(image_train.images.shape[1]),
        'label':_int64_feature(np.argmax(image_train.labels)),
        'image_raw':_bytes_feature(image_raw),
        'height':_int64_feature(84),
        'width':_int64_feature(25)
    }))
    return example

#读取图片数据Mnist的训练数据
mnist = input_data.read_data_sets(dataset_path, dtype=tf.uint8, one_hot=True)
# images = mnist.train.images # 5500*784
# labels = mnist.train.labels #5500*10
# pixels = images.shape[1] #784
n_examples = mnist.train.num_examples #5500

#输出包含训练数据的TFRecord文件
if not os.path.exists('./output.tfrecords'):#如果文件已经存在就不执行
    with tf.python_io.TFRecordWriter('./output.tfrecords') as writer:
            #将每一张图片数据顺序保存为tfrecord文件
            for index in range(n_examples):
                example = _make_example(mnist, index)
                writer.write(example.SerializeToString()) #序列化
            logger.info('训练数据的TFReord文件已经保存！')
else:
    logger.info('数据已经存在!')
#q读取测试数据
images_test = mnist.test.images
# labels_test = mnist.test.labels
# pixels_test = images_test.shape[1]
n_examples_test = mnist.test.num_examples

#输出包含测试数据的TFRecord文件
if not os.path.exists('./output_test.tfrecords'): #如果文件已经存在就不执行
    with tf.python_io.TFRecordWriter('./output_test.tfrecords') as writer:
            for index in range(n_examples_test):
                example = _make_example(mnist, index)
                writer.write(example.SerializeToString())
            logger.info('测试数据的TFRecord文件已经保存！')
else:
    logger.info('数据已经存在!')

#二、读取TFRecord格式的文件
#获取文件列表,并根据文件列表来创建文件队列
files = tf.train.match_filenames_once('./output.tfrecords')
file_name_queue = tf.train.string_input_producer(files, shuffle=False)

#解析数据
reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_name_queue)
features = tf.parse_single_example(serialized_example, features={
    'image':tf.FixedLenFeature([], tf.string),
    'label':tf.FixedLenFeature([], tf.int64),
    'height':tf.FixedLenFeature([], tf.int64),
    'width':tf.FixedLenFeature([], tf.int64),
    'channels':tf.FixedLenFeature([], tf.int64)
})

image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

#三、处理图像数据
#从原始图像数据解析出像素矩阵，并根据图像尺寸还原数据
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])

#定义神经网络输入层的大小,并进行图像预处理
image_size = 299
distorted_image = data_process.processes_for_train(decoded_image, image_size, image_size, None)

#将处理后的图像和标签数据通过tf.train.shuffle_batch整理成神经网络需要的batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3*batch_size

image_batch, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

#四、定义神经网络的结构
learning_rate = 0.01 #学习率
STEPS = 5000
logit = mnist_inference.inference(image_batch)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#五、声明会话并运行神经网络的优化过程
with tf.Session() as sess:
    #tf.local_variables_initializer() 只要定义多线程，就要运行这个初始化
    sess.run(tf.global_variables_initializer(), tf.local_variables_initializer())

    #实例化一个管理多线程的实例
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    #神经网络的训练过程
    for i in range(STEPS):
        sess.run(train_step)

    #记得停止所有线程
    coord.request_stop()
    coord.join(threads)



