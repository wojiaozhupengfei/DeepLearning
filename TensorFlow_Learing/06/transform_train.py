#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/17/0017 15:36
# @Author  : zhupengfei
# @Site    : 
# @File    : transform_train.py
# @Software: PyCharm
# 该文件利用inception-V3训练好的模型完成fine_tuning

from logger_test import *
import glob
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import os.path
import tensorflow.contrib.slim as slim  #tensorflow高度封装化模块，模型建立过程简化

#通过tensorflow-slim加载训练好的inception-v3
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

#处理好的图片数据文件所在路径
DATA_FILE = '../dataset/flower_photos/flower_processed_data.npy'
#训练好的inception-v3模型文件所在路径
CKPT_FILE = '../models/inception_v3_2016_08_28/inception_v3.ckpt'
#保存fine_tuning好的模型文件
TRAIN_FILE_SAVE_PATH = './save_model/tuned_model'

#定义训练中使用的参数
LEARNING_RATE = 0.0001 #学习率
STEPS = 300
BATCH_SIZE = 5
N_CLASSES = 5
display_steps = 10
save_steps = STEPS/4

#定义inception_v3模型中不需要的参数，这些参数需要我们自己训练，就是最后的全连接层的参数
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
#模型不需要加载的数据就是需要训练的数据
TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'

#获取incetion_v3中训练好的参数，当然不包括我们要去掉的最后一层全连接层的参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')] #不需要加载的参数
    variable_to_restore = []#保存加载的参数

    #枚举inception_v3模型中所有参数，去掉不需要加载的
    for var in slim.get_model_variables(): #注意这个函数的使用
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variable_to_restore.append(var)
    return variable_to_restore
#获取需要训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variable_to_train = []
    #通过需要训练参数的前缀来找到对应的参数
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variable_to_train.extend(variables)#这里注意extend和append区别，extend只接收list类型
    return variable_to_train

def main(argv = None):
    #训练过程分为以下步骤：1 加载数据 2定义网络模型 3定义损失函数 4定义优化器 5定义评估指标

    #加载处理好的图片数据
    processed_data = np.load(DATA_FILE)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    n_validation_example = len(validation_images)
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    n_testing_example = len(testing_images)
    testing_labels = processed_data[5]
    logger.info('%d training examples, %d validation examples, %d testing examples.' %(n_training_example, n_validation_example, n_testing_example))

    #定义输入数据和label
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    #定义inception-v3模型。因为谷歌中给的inception_v3模型只有参数取值，所以这里要定义inception_v3模型结构。因为训练好的inception_v3模型中使用的Batch_normlization参数与新的数据会有差异，导致训练结果很差，所以这里直接使用一个模型进行测试，不区分训练模型和测试模型
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)

    #获取需要训练的变量
    trainable_variables = get_trainable_variables()

    #定义交叉熵损失函数，参数的正则项损失在定义模型的时候已经加载
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    #定义优化器
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    #计算正确率，评估模型
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evalution_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)

    #定义保存训练好的模型
    saver = tf.train.Saver()

    #开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #加载谷歌训练好的模型
        logger.info('Loading tuned variables from %s'%CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH_SIZE
        for i in range(STEPS):
            logger.info('Step %d-%d is training....'%(i, STEPS))
            try:
                sess.run(train_step, feed_dict = {images:training_images[start:end], labels:training_labels[start:end]})
            except Exception:
                logger.error('trainging fail', exc_info = True)

            #输出日志
            if i % display_steps == 0 or i + 1 == STEPS:
                validation_acc = sess.run(evalution_step, feed_dict = {images:validation_images, labels:validation_labels})
                logger.info('Step %d-%d:validation acc = %.1f%%'%(i, STEPS, validation_acc*100.0))
                #模型持久化
                if i % save_steps == 0 or i+1 == STEPS:
                    saver.save(sess, TRAIN_FILE_SAVE_PATH, global_step=i)
            #因为数据处理时就已经打乱了顺序，所以在这里直接顺序使用训练数据就可以
            start = end
            if start == n_training_example:
                start = 0
            end = end + BATCH_SIZE
            if end > n_training_example:
                end = n_training_example
        #最后在测试集上测试正确率
        test_acc = sess.run(evalution_step, feed_dict = {images:testing_images, labels:testing_labels})
        logger.info('Final test acc = %.1f%%'%(test_acc * 100.0))

if __name__ == '__main__':
    tf.app.run()