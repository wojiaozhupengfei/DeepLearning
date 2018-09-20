#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20/0020 6:54
# @Author  : zhupengfei
# @Site    : 
# @File    : file_queue.py
# @Software: PyCharm
#输入文件队列
from log import *
import tensorflow as tf

#创建TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

num_shards = 2 #总共写入多少文件
instance_per_share = 2 #每个文件中有多少个数据

for i in range(num_shards):
    filename = ('./data.tfrecord-%.5d-%.5d'%(i, num_shards))
    writer = tf.python_io.TFRecordWriter(filename)

    #将数据封装成Example结构并写入TFRecord文件
    for j in range(instance_per_share):
        example = tf.train.Example(features = tf.train.Features(feature = {'i':_int64_feature(i), 'j':_int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()


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