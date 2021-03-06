#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/28/0028 14:06
# @Author  : zhupengfei
# @Site    : 
# @File    : alexnettest.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # 导入tf库的手写字体识别的dataset
import numpy as np
import os

# 1.导入数据集
mnist = input_data.read_data_sets('./data//', one_hot=True)

# 2.定义全局参数
lr = 0.0001
train_iters = 10000
batch_size = 16
input_dim = 784  # 28*28*1 的单通道图像，拉伸成一维
mnist_class = 10
dropout = 0.5
display_step = 1

x = tf.placeholder(tf.float32, [None, input_dim])
y = tf.placeholder(tf.float32, [None, mnist_class])
drop_out = tf.placeholder(tf.float32)

# 3.定义函数，包括 卷积，池化，正则， alexnet
# 卷积
def conv2d(name, input_data, filter, bias):
    x = tf.nn.conv2d(input_data, filter=filter, strides=[1, 1, 1, 1], padding='SAME', name=None, use_cudnn_on_gpu=False, data_format='NHWC')
    x = tf.nn.bias_add(x, bias, data_format=None, name=None)
    return tf.nn.relu(x, name=name)
# 池化， ksize的数据结构[batche_number， k, k, channel]
def max_pooling(name, input_data, k):
    return tf.nn.max_pool(input_data, ksize=[1, k, k, 1],strides=[1, k, k, 1], padding='SAME', name=name)


# lrn  局部响应归一化
def norm(name, input_data, lsize = 4):
    return tf.nn.lrn(input_data, depth_radius=lsize, bias=1, alpha=1, beta=0.5, name=name)

# 定义网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 48])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 48, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 192])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 192, 192])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 192, 128])),
    'wfc1': tf.Variable(tf.random_normal([4*4*128, 4096])),
    'wfc2': tf.Variable(tf.random_normal([4096, 4096])),
    'wfc3': tf.Variable(tf.random_normal([4096, mnist_class]))
           }
# 定义偏移
bias = {
    'bwc1': tf.Variable(tf.random_normal([48])),
    'bwc2': tf.Variable(tf.random_normal([128])),
    'bwc3': tf.Variable(tf.random_normal([192])),
    'bwc4': tf.Variable(tf.random_normal([192])),
    'bwc5': tf.Variable(tf.random_normal([128])),
    'bwfc1': tf.Variable(tf.random_normal([4096])),
    'bwfc2': tf.Variable(tf.random_normal([4096])),
    'bwfc3': tf.Variable(tf.random_normal([mnist_class]))
}

# alexnet
def AlexNet(input_image, weights, bias, dropout):
    input_image = tf.reshape(input_image, shape=[-1, 28, 28, 1])

    # conv1
    conv1 = conv2d('conv1', input_image, weights['wc1'], bias['bwc1'])
    pooling1 = max_pooling('pooling1', conv1, k = 2) # 第一层输出要经过池化
    norm1 = norm('norm1', pooling1, lsize=4)

    # conv2
    conv2 = conv2d('conv2', norm1, weights['wc2'], bias['bwc2'])
    pooling2 = max_pooling('pooling2', conv2, k=2)  # 第二层输出要经过池化
    norm2 = norm('norm2', pooling2, lsize=4)

    # conv3
    conv3 = conv2d('conv3', norm2, weights['wc3'], bias['bwc3'])
    norm3 = norm('norm3', conv3, lsize=4)

    # conv4
    conv4 = conv2d('conv4', norm3, weights['wc4'], bias['bwc4'])
    norm4 = norm('norm4', conv4, lsize=4)

    # conv5
    conv5 = conv2d('conv5', norm4, weights['wc5'], bias['bwc5'])
    pooling5 = max_pooling('pooling5', conv5, k=2) # 第五层输出要经过池化
    norm5 = norm('norm5', pooling5, lsize=4)

    # fc1
    fc1_input = tf.reshape(norm5, shape=[-1, weights['wfc1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(fc1_input, weights['wfc1']) + bias['bwfc1'], name='fc1')
    dense1 = tf.nn.dropout(dense1, keep_prob=dropout) # 随机失活，防止过拟合, 增加准确率

    # fc2
    fc2_input = tf.reshape(dense1, shape=[-1, weights['wfc2'].get_shape().as_list()[0]])
    dense2 = tf.nn.relu(tf.matmul(fc2_input, weights['wfc2']) + bias['bwfc2'], name='fc2')
    dense2 = tf.nn.dropout(dense2, keep_prob=dropout)

    #out
    out = tf.matmul(dense2, weights['wfc3']) + bias['bwfc3']
    return out

# 构建模型, 包括 1.输出预测，2.学习率动态下降 3.损失函数 4.优化器 5.准确率
# 输出预测
pred = AlexNet(x, weights, bias, drop_out)

# 学习率,计算公式 lr = lr * decay_rate^(global_step/decay_step)
global_step = tf.constant(0, tf.int64)
decay_rate = tf.constant(0.9, tf.float64)
learn_rate = tf.train.exponential_decay(lr, global_step, decay_steps=10000, decay_rate = decay_rate)

# 交叉熵损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

# 优化器
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)

# 准确率
acc_tf = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

acc = tf.reduce_mean(tf.cast(acc_tf, tf.float32))

# 初始化所有的共享变量
init = tf.global_variables_initializer()

# 开启一个训练
def train():
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # 迭代次数不超过规定最大次数（training_iters）
        while step * batch_size < train_iters:
            # 每一次从mnist的训练集中取出batch_size个图片数据，进行训练
            # batch_xs为图片数据 ； batch_ys 为标签值
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 获取批数据，开始训练
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, drop_out: dropout})
            # 每次运行display_step=20步，计算精度，计算损失值和打印
            if step % display_step == 0:
                # 计算精度
                accplay = sess.run(acc, feed_dict={x: batch_xs, y: batch_ys, drop_out: 1.})
                # 计算损失值
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, drop_out: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss = " + "{:.6f}".format(
                    loss) + ", Training Accuracy = " + "{:.5f}".format(accplay))
            step += 1
        print("Optimization Finished!")
        # 保存模型
        # 初始化一个保存方法
        saver = tf.train.Saver()
        # 制定保存文件夹名称
        save_path = 'ckpt'
        # 检查文件夹是否存在，假如不存在文件夹，就创建一个文件夹
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # 制定模型保存的路径下的文件名，
        model_name = save_path + os.sep + "alexnet.ckpt"
        # 调用保存方法，sess就是模型的参数和对应的值
        saver.save(sess, model_name)

        # 可视化图片

        # img_input=mnist.test.images[img_index:img_index+1,:]
        # predict=sess.run(pred,feed_dict={x:img_input,drop_prob:1.0})
        # Nums = [0,1,2,3,4,5,6,7,8,9]

        # print ('prediction is:',np.where(np.int16(np.round(predict[img_index,:],0))==1)[0][0])
        ##可视化卷积核

        # 计算测试精度
        print("Testing Accuracy:",
              sess.run(acc, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], drop_out: 1.}))
if __name__=='__main__':
    if os.path.exists('ckpt'): # 如果训练好的文件夹找到了，就开始测试
        count = 0
        listd = os.listdir('ckpt') # 返回一个列表，存放了该文件下所有文件的name，不包括特殊字符
        for f in listd:
            count += 1  # 看看这个文件夹下是不是有四个文件
        if count == 4: # 如果存在模型，开始调用
            init = tf.global_variables_initializer()
            restore = tf.train.Saver()  #Saver有两个属性save和restore，保存和加载
            with tf.Session() as sess:
                # sess.run(init) #初始化,这里不需要，因为restore方法加载ckpt文件就是一种初始化
                '''
                该函数返回的是checkpoint文件CheckpointState proto类型的内容，
                其中有 model_checkpoint_path 和 all_model_checkpoint_paths 两个属性。
                其中model_checkpoint_path保存了最新的tensorflow模型文件的文件名，
                all_model_checkpoint_paths则有未被删除的所有tensorflow模型文件的文件名
                '''
                ckpt = tf.train.get_checkpoint_state('ckpt')
                if ckpt and ckpt.model_checkpoint_path: #文件存在
                    restore.restore(sess, ckpt.model_checkpoint_path) #加载训练好的模型，并加载进入sess
                image_index = 10 #测试的时候取一张图片，这个编号是10的图片
                test_image = tf.reshape(mnist.test.images, [-1, 28, 28, 1]) #将测试集还原为图片格式
                a_image = sess.run(test_image)[image_index, :, :, 0] #取出index为10的这张图片
                input_image = mnist.test.images[image_index:image_index + 1, :]
                predict = sess.run(pred, feed_dict={x:input_image, drop_out:1.})
                result = tf.argmax(predict, 1)
                a_image = tf.reshape(input_image, [1, 28, 28])

                print('prediction is :', sess.run(result))
                import matplotlib.pyplot as plt
                import pylab #该模块和pyplot基本一样，就是加入了一些numpy的计算哭，方便画图的时候来计个算，不推荐
                # plt.imshow(a_image)
                pylab.show()

                print(sess.run(weights['wc1']).shape)
                '''
                这里注意subplot和subplots的微妙区别
                fig：matplotlib.figure.Figure对象
                ax：Axes(轴)对象或Axes(轴)对象数组  
                '''
                f, axarr = plt.subplots(4, figsize = [10, 10])
                axarr[0].imshow(sess.run(weights['wc1'])[:, :, 0, 0])
                axarr[1].imshow(sess.run(weights['wc2'])[:, :, 23, 12])
                axarr[2].imshow(sess.run(weights['wc3'])[:, :, 41, 44])
                axarr[3].imshow(sess.run(weights['wc4'])[:, :, 45, 55])
                pylab.show()

    else:
        train()