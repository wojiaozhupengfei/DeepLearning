#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20/0020 7:50
# @Author  : zhupengfei
# @Site    : 
# @File    : file_queue_batch.py
# @Software: PyCharm
#多线程队列操作batch个文件, 这个程序报错，原因还没找出来

from log import *
import tensorflow as tf

#tf.train.match_filename_once函数和tf.train.string_input_producer函数的使用方法
#获取文件列表
files = tf.train.match_filenames_once('./data.tfrecord-*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)

#读取并解析一个样本
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={'i':tf.FixedLenFeature([], tf.int64), 'j':tf.FixedLenFeature([], tf.int64)})

with tf.Session() as sess:
    tf.local_variables_initializer().run()

    logger.info(sess.run(files))

    #实例化tf.train.Coordinator协同不同线程,并启动多线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #多次执行并获取数据操作
    for i in range(6):
        logger.info(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)

#i表示特征向量，即像素矩阵， j表示对应的标签
example, label = features['i'], features['j']

batch_size = 3

#队列大小，一般队列大小与batch相关，如果队列太大，占用太多内存，如果队列太小，出队操作可能因为没有数据而被阻碍，影响训练效率
capacity = 1000+3*batch_size #队列的最大容量

example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity = capacity)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #获取并打印组合之后的样例，真是问题中，这个输出一般会作为神经网络的输入
    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        logger.info(cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)
