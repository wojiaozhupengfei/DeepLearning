#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/13/0013 10:51
# @Author  : zhupengfei
# @Site    : 
# @File    : mnist_eval.py
# @Software: PyCharm
#测试文件
import time
import tensorflow as tf
import mnist_inference
import mnist_train
from tensorflow.examples.tutorials.mnist import input_data

#每10秒加载一次最新的模型
eval_time_secs = 2

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], 'x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], 'y-input')

        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        predict_correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(predict_correct, tf.float32))

        #这段代码没有看明白,根本就没有用到滑动平均模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DACAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:

                ckpt = tf.train.get_checkpoint_state(mnist_train.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #通过文件名称得到保存模型时的迭代轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    acc_score = sess.run(acc, validate_feed)
                    print('After %s training step, validation acc = %g'%(global_step, acc_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(eval_time_secs)
def main(argv = None):
    mnist = input_data.read_data_sets('../dataset', one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()