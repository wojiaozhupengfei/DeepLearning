#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/19/0019 15:42
# @Author  : zhupengfei
# @Site    : 
# @File    : threading_test.py
# @Software: PyCharm
#多线程操作

import tensorflow as tf
import numpy as np
import time
import threading
from log import *

# #线程中运行的程序，功能是 返回自身id
# def MyLoop(coord, worker_id):
#     #使用tf.Coordinator判断当前线程是否要停止
#     while  not coord.should_stop:
#         #随机停止所有线程
#         if np.random.rand() < 0.1:
#             logger.info('Stopping from id: %d\n'%worker_id)
#             #调用coord.request_stop停止所有其他程序
#             coord.request_stop()
#         else:
#             #执行程序的功能
#             logger.info('Working on id: %d\n'%worker_id)
#         time.sleep(1)
#
# #创建多线程的实例
# coord = tf.train.Coordinator()
# #创建5个线程
# threads = [threading.Thread(target=MyLoop, args=(coord, i)) for i in range(5)]
#
# #启动所有线程
# for t in threads:
#     t.start()
# #等待所有线程退出
# coord.join(threads)

#用多线程来操作同一个队列


# 声明一个队列，先进先出
queue = tf.FIFOQueue(100, 'float')
#定义入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

#tf.train.QueueRunner来创建多个线程运行入队操作
#第一个参数是被操作的队列，[enqueue_op] *5表示5个线程，每个线程运行的操作是enqueue_op
qr = tf.train.QueueRunner(queue, [enqueue_op] *5)

#将定义好的QueueRunner 加入tensorflow计算图上指定的集合
#如果tf.train.add_queue_runner函数没有指定集合，则加入默认集合tf.GraphKeys.QUEUE_RUNNERS
tf.train.add_queue_runner(qr)

#定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    #实例化coord 协同操作多线程
    coord = tf.train.Coordinator()

    #使用tf.train.QueueRunner时，需要明确调用tf.train.start_queue_runners来启动所有线程
    #否则因为没有线程的入队操作，当调用出队操作时，程序会一直等待入队操作被执行
    #tf.train.start_queue_runners一般会默认自动执行
    #tf.GraphKeys.QUEUE_RUNNERS 这个支持是启动指定集合中的QueueRunner,所以一般来说
    #tf.train.add_queue_runner和tf.train.start_queue_runners会指定在同一个集合
    threads = tf.train.start_queue_runners(sess = sess, coord= coord)

    #获取队列中的值
    for _ in range(3):
        logger.info(sess.run(out_tensor)[0])
    #使用tf.train.Coordinator来停止所有线程
    coord.request_stop()
    coord.join(threads)