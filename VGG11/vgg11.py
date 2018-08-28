#!/user/bin/env python
# -*- coding: utf-8 -*-
#@Time     :2018/8/22/0022 19:30
#@Author   :zhupengfei
#@Site     :
#@File     :vgg11.py
#@Software :PyCharm

#训练VGG11，数据集是oxFlower17,

import pandas as pd
import tensorflow as tf
from tflearn.datasets import oxflower17
import time

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#调用接口，自动下载数据集oxFlower17，并将数据进行解析为One_hot
X, Y = oxflower17.load_data(one_hot=True)
#数据集一共有1360张花，共有17类，输入数据x为1360*224*224*3，输入标签y为1360*17
# print(x, x.shape)
# print(y, y.shape)

#1.设计训练相关的参数

train_epoch = 1000 #训练的迭代次数，训练完1360张图片为一个epoch
learn_rate = tf.placeholder(tf.float32) #设置学习率
epoch_display = 100 #每100个epoch进行一次评估
batch_size = 16 #每个批次训练16张图片
class_num = Y.shape[1]
x = tf.placeholder(tf.float32, [None, 224, 224, 3]) #输入图片的尺寸，第一个参数为输入几张图片，设置为None，就是自动识别有几张
y = tf.placeholder(tf.float32, [None, class_num]) #输出类别为17类

#2.设计网络 vgg11
def vgg_network():
	# 各层权重
	weights = {
		'wc1_1' : tf.get_variable('wc1_1', [3, 3, 3, 64]), #该层之后有池化层，输出特征图 112 * 112 * 64
		'wc2_1' : tf.get_variable('wc2_1', [3, 3, 64, 128]), #该层之后有池化层， 输出特征图 56 * 56 * 128
		'wc3_1' : tf.get_variable('wc3_1', [3, 3, 128, 256]), #该层之后没有池化层
		'wc3_2' : tf.get_variable('wc3_2', [3, 3, 256, 256]), #该层之后有池化层， 输出特征图 28 * 28 * 256
		'wc4_1' : tf.get_variable('wc4_1', [3, 3, 256, 512]), #该层之后没有池化层
		'wc4_2' : tf.get_variable('wc4_2', [3, 3, 512, 512]), #该层之后有池化层， 输出特征图 14 * 14 * 512
		'wc5_1' : tf.get_variable('wc5_1', [3, 3, 512, 512]), #该层之后没有池化层
		'wc5_2' : tf.get_variable('wc5_2', [3, 3, 512, 512]), # 该层之后有池化层， 输出特征图 7 * 7 * 512
		'wfc_1' : tf.get_variable('wfc1', [7 * 7 * 512, 4096]), #将特征图拉直，进行全连接
		'wfc_2' : tf.get_variable('wfc2', [4096, 4096]), #第2个全连接层
		'wfc_3' : tf.get_variable('wfc3', [4096, class_num])#最后一层全连接输出class_num，softmax分类
	}
	# 各层偏置
	bias = {
		'bc1_1' : tf.get_variable('bc1_1', [64]),
		'bc2_1' : tf.get_variable('bc2_1', [128]),
		'bc3_1' : tf.get_variable('bc3_1', [256]),
		'bc3_2' : tf.get_variable('bc3_2', [256]),
		'bc4_1' : tf.get_variable('bc4_1', [512]),
		'bc4_2' : tf.get_variable('bc4_2', [512]),
		'bc5_1' : tf.get_variable('bc5_1', [512]),
		'bc5_2' : tf.get_variable('bc5_2', [512]),
		'bfc_1' : tf.get_variable('bfc_1', [4096]),
		'bfc_2' : tf.get_variable('bfc_2', [4096]),
		'bfc_3' : tf.get_variable('bfc_3', [class_num])
	}

	# 设计网络
	# 1.卷积， 2.激活， 3.池化
	# conv1,该层激活之后需要局部归一化LRN
	net = tf.nn.conv2d(input = x, filter = weights['wc1_1'], strides=[1, 1, 1, 1], padding='SAME') #卷积
	net = tf.nn.leaky_relu(tf.nn.bias_add(net, bias['bc1_1'])) #加偏置之后激活
	net = tf.nn.lrn(net) # 局部归一化 LRN
	net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # conv1之后有池化特

	# conv2
	net = tf.nn.conv2d(input=net, filter=weights['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
	net = tf.nn.leaky_relu(tf.nn.bias_add(net, bias['bc2_1']))
	net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID') #conv2之后有池化

	# conv3
	net = tf.nn.conv2d(net, weights['wc3_1'], [1, 1, 1, 1], padding='SAME')
	net = tf.nn.leaky_relu(tf.nn.bias_add(net, bias['bc3_1']))
	net = tf.nn.conv2d(net, weights['wc3_2'], [1, 1, 1, 1], padding='SAME')
	net = tf.nn.leaky_relu(tf.nn.bias_add(net, bias['bc3_2']))
	net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID') #conv3_2之后有池化

	# conv4
	net = tf.nn.conv2d(net, weights['wc4_1'], [1, 1, 1, 1], padding='SAME')
	net = tf.nn.leaky_relu(tf.nn.bias_add(net, bias['bc4_1']))
	net = tf.nn.conv2d(net, weights['wc4_2'], [1, 1, 1, 1], padding='SAME')
	net = tf.nn.leaky_relu(tf.nn.bias_add(net, bias['bc4_2']))
	net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID') # conv4_2之后有池化

	# conv5
	net = tf.nn.conv2d(net, weights['wc5_1'], [1, 1, 1, 1], padding='SAME')
	net = tf.nn.leaky_relu(tf.nn.bias_add(net, bias['bc5_1']))
	net = tf.nn.conv2d(net, weights['wc5_2'], [1, 1, 1, 1], padding='SAME')
	net = tf.nn.leaky_relu(tf.nn.bias_add(net, bias['bc5_2']))
	net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID') # conv5_2之后有池化

	#将featuremap拉伸到fc输入的维度
	net = tf.reshape(net, [-1, weights['wfc_1'].get_shape()[0]])

	#wfc_1
	net = tf.matmul(net, weights['wfc_1']) + bias['bfc_1']
	net = tf.nn.relu(net)

	#wfc_2
	net = tf.matmul(net, weights['wfc_2']) + bias['bfc_2']
	net = tf.nn.relu(net)

	#wfc_3
	net = tf.matmul(net, weights['wfc_3']) + bias['bfc_3']

	return net

# 3. 设计损失函数和优化器，并求精确率
pred = vgg_network() # vgg11 最后一次的输出, 没有经过softmax
# softmax交叉熵损失函数，该函数集成了softmax和交叉熵
loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
opt = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss) #优化器,BP计算就是在这里
acc_tf = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) #调试看看形状,最新文档改为tf.argmax
acc = tf.reduce_mean(tf.cast(acc_tf, tf.float32))

# 4.开始训练！！！！！！！！！！！！
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) # 初始化variable变量
	base_learn_rate = 0.001 #初始化学习率
	learn_rate1 = base_learn_rate  #这里的learn_rate1仅仅是为了和上面的learn_rate不重名
	for epoch in range(train_epoch): # 1000次epoch
		iterations = X.shape[0] // batch_size #计算一个epoch要进行几轮batch_size
		for i in range(iterations): # 1360 / 16 = 85
			iterations_start_time = time.time()
			print('当前epoch为: %s'%(str(epoch)) + ' ' + 'total_batch为: %s'%(str(i)) + '\n')
			x_trian, y_train = X[i * batch_size : i * batch_size + batch_size], Y[i * batch_size : i * batch_size + batch_size] #训练集和测试集取batch_size大小
			sess.run(opt, feed_dict={x:x_trian, y:y_train, learn_rate:learn_rate1})
			cost, accur_tf, accur = sess.run([loss, acc_tf, acc], feed_dict={x:x_trian, y:y_train, learn_rate:learn_rate1})
			# print('%s - %s 的loss为: %s'%(str(epoch), str(i), str(cost)) + '\n')
			# print('%s - %s 的accur_tf为: %s'%(str(epoch), str(i), str(accur_tf)) + '\n')
			# print('%s - %s 的accur为: %s' % (str(epoch), str(i), str(accur)) + '\n')

			if (epoch + 1) % epoch_display == 0: # 100个epoch打印一次
				cost, accur = sess.run([loss, acc], feed_dict={x:x_trian, y:y_train, learn_rate : learn_rate1})
				print('step: %s loss: %f acc: %f'%(str(epoch), cost[0], accur))
				learn_rate1 = base_learn_rate * (1 - epoch/train_epoch)**2  # 学习率退火

			iterations_end_tiem = time.time()
			print('一次迭代需要训练时间(s): %d'%(iterations_end_tiem - iterations_start_time) + '\n')









