#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/12/0012 15:01
# @Author  : zhupengfei
# @Site    : 
# @File    : MNIST_use_Variable_scope.py
# @Software: PyCharm
#用一个三层的全连接神经网络实现MNIST数据集的训练 训练集55000  验证集5000 测试集10000  但是interface函数进行优化，使用了Variable_Scope来管理变量
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

data_dir = '../dataset'
#取数据
# mnist = input_data.read_data_sets(data_dir, one_hot=True)

#定义全局参数
INPUT_NODE = 784 #MNIST数据集一张图片拉伸
OUTPUT_NODE = 10 # 10个数字类别
LAYER1_NODE = 500 #中间的隐层的神经元个数
BATCH_SIZE = 100 #batch_size数
LEARNING_RATE_BASE = 0.8  #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减系数
REGULARIZATION_RATAE = 0.0001  #L正则的lambda系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均模型的衰减系数
display_iter = 1000 #每1000次输出一次验证集上的acc
validata_acc_list = []
test_acc_list = []

#传入输入及参数，计算模型最后一层的输出,该函数多一个参数avg_class用来控制启用不启用滑动平均模型
# def interface(input, avg_class, weights1, bias1, weights2, bias2):
#     if avg_class == None: #不启用滑动平均模型
#         layer1 = tf.nn.relu(tf.matmul(input, weights1)+bias1)
#         return tf.matmul(layer1, weights2) + bias2
#     else: #启用滑动平均模型
#         layer1 = tf.nn.relu(tf.matmul(input, avg_class.average(weights1)) + avg_class.average(bias1))
#         return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(bias2)

def get_weights_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

# interface函数进行优化，使用了Variable_Scope来管理变量
def interface(input, avg_class, regularizer, reuse = False):
    if avg_class == None: #不启用滑动平均模型
        #第一层
        with tf.variable_scope('layer1'): #reuse开启表示下次进来不是新建变量，而是利用上次存储好的变量复用，关键所在
            w1 = get_weights_variable([INPUT_NODE, LAYER1_NODE],regularizer)
            b1 = tf.get_variable('b1', [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
            layer1 = tf.nn.relu(tf.matmul(input, w1) + b1)
        #第二层
        with tf.variable_scope('layer2'):
            w2 = get_weights_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
            b2 = tf.get_variable('b2', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
            layer2 = tf.matmul(layer1, w2) + b2
        return layer2
    else: #启用滑动平均模型
        #第一层
        # !!!!!!!!注意这里有个坑，上面的w初始化一次再不进去了，所以reuse默认False不用管，但是！！！当你下面又调用了一次求滑动平均的y，就会进来这里，那么你的命名空间不能和上面一样layer1，变一个，或者一样，要reuse=True复用w就可以
        with tf.variable_scope('layer1', reuse=True): #reuse开启表示下次进来不是新建变量，而是利用上次存储好的变量复用，关键所在,
            w1 = get_weights_variable([INPUT_NODE, LAYER1_NODE], regularizer)
            b1 = tf.get_variable('b1', [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
            layer1 = tf.nn.relu(tf.matmul(input, avg_class.average(w1)) + avg_class.average(b1))
        #第二层
        with tf.variable_scope('layer2', reuse=True):
            w2 = get_weights_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
            b2 = tf.get_variable('b2', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
            layer2 = tf.matmul(layer1, avg_class.average(w2)) + avg_class.average(b2)
        return layer2

#训练, 只要传入参数为训练数据集
def train(mnist):
    #定义训练数据
    x = tf.placeholder(dtype=tf.float32, shape = [None, INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], 'y_output')

    # #随机初始化训练参数
    # w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    # bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # w2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    # bias2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    #计算正则项的损失,首先构建L2正则
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATAE)

    #计算不使用滑动平均模型的时候的前向传播的结果
    y = interface(x, None, regularizer)

    #构建滑动平均模型，并计算使用滑动平均模型的时候前向传播的结果
    global_step = tf.Variable(tf.constant(0), trainable=False) #不参与训练
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables()) #意思是给所有参与训练的额参数使用滑动平均模型
    y_average = interface(x, variable_averages, regularizer)

    #构建损失函数,计算损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, axis=1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  #取batch的平均损失
    # regularization = regularizer(w1) + regularizer(w2) #偏置项一般不进行正则
    #计算总的损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #构建优化器
    #设置衰减学习率
    learing_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step=global_step, decay_steps=mnist.train.num_examples/BATCH_SIZE, decay_rate=LEARNING_RATE_DECAY)
    #构建优化器,包含了交叉熵和l2正则项
    train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss, global_step)

    #构建组合操作，因为反向传播计算中不仅要更新w，b 而且因为训练参数用了滑动平均模型，所以这个模型的值也要更新，组合操作就是将二者操作合起来
    train_op = tf.group(train_step, variable_averages_op)

    #构建模型评价指标，准确率
    correction_predict = tf.equal(tf.argmax(y_average, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correction_predict, tf.float32))

    #开始训练
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        validata_feed = {x:mnist.validation.images, y_:mnist.validation.labels} #验证集数据
        test_feed = {x:mnist.test.images, y_:mnist.test.labels} #测试数据

        #开始迭代训练
        for i in range(TRAINING_STEPS):
            if i % display_iter == 0:# 每1000轮就看看验证集的正确率
                validata_acc = sess.run(accuracy, validata_feed)
                test_acc = sess.run(accuracy, test_feed) # 为了与验证集对比，所以也把测试集打印出来
                print('After %d training steps, validation accuracy using average model is %g，test accuracy using average model is %g'%(i, validata_acc,test_acc))
                validata_acc_list.append(validata_acc) #将每次的acc加入数组，画图用
                test_acc_list.append(test_acc)  # 将每次的acc加入数组，画图用
            xs, ys = mnist.train.next_batch(BATCH_SIZE) #batch个训练集
            sess.run(train_op, feed_dict={x:xs, y_:ys})
        #训练完成之后看看测试集的准确率
        test_acc = sess.run(accuracy, test_feed)
        print('After %d training steps, test accuracy using average model is %g'%(TRAINING_STEPS, test_acc))
        plt.xlim(0, 5, 5/30)
        plt.ylim(0.9, 1., (1.-0.9)/30)
        plt.plot(np.arange(TRAINING_STEPS/display_iter), validata_acc_list)
        plt.plot(np.arange(TRAINING_STEPS/display_iter), test_acc_list)
        plt.show()

# train(mnist)
#程序主入口
def main(argv = None):
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    train(mnist)

#tensorflow提供了一个主函数，tf.app.run会调用上面自定义的主函数main
if __name__ == '__main__':
    tf.app.run()