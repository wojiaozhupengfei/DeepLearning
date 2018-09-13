#!/user/bin/env python
# -*- coding: utf-8 -*-
#@Time     :2018/9/13/0013 20:00
#@Author   :zhupengfei
#@Site     :
#@File     :mnist_conv_train.py
#@Software :PyCharm
#@descibe  ：该文件修改一下输入数据的格式，用lenet-5训练
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from .mnist_inference import *
# from . import mnist_inference
import mnist_conv_inference
#定义训练过程中的全局变量
TRAINING_STEP = 30000  #一共进行30000次训练，这里和epoch不同，注意一下
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8 # 优化器的基础学习率
LEARNING_RATE_DECAY = 0.99 #优化器学习率的衰减系数
REGURLARIZER_RATE = 0.0001 #正则项的惩罚系数
MOVING_AVERAGE_DACAY  = 0.99 #滑动平均模型的衰减系数
display_step = 1000  #每1000次打印一下训练集的损失并保存模型

#路径
data_dir = '../dataset'
model_save_path = './model'
model_name = 'model.ckpt'

def train(mnist):
    # 取数据
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_conv_inference.IMAGE_SIZE, mnist_conv_inference.IMAGE_SIZE, mnist_conv_inference.NUM_CHANNELS], 'x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_conv_inference.OUTPUT_NODE], 'y-input')

    #定义正则项
    regularizer = tf.contrib.layers.l2_regularizer(REGURLARIZER_RATE)

    #前向计算
    y = mnist_conv_inference.inference(x, regularizer, 1)
    global_step = tf.Variable(0, trainable=False)

    #下面老套路，滑动平均模型,损失， 优化器，训练
    #滑动平均模型
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DACAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #交叉熵损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  #对batch_size个数据求个平均值
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #优化器
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    #合并op
    train_op = tf.group(train_step, variable_averages_op)

    #实例化模型持久化类
    saver = tf.train.Saver()

    #开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            #注意，这里也有个bug，不要用tf.reshape，这样转换得到的是tensor类型，然而feed_dict不接受tensor类型，只接受ndarray、list等类型
            reshaped_xs = xs.reshape([BATCH_SIZE, mnist_conv_inference.IMAGE_SIZE, mnist_conv_inference.IMAGE_SIZE, mnist_conv_inference.NUM_CHANNELS])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:reshaped_xs, y_:ys})
            #每1000次打印损失及保存模型
            if i % display_step == 0:
                print('After %d training step, loss on training batch is %g, global step is %d'%(i, loss_value, step))
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)
def main(argv = None):
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()