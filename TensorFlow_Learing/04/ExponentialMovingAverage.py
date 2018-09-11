#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/12/0012 7:19
# @Author  : zhupengfei
# @Site    : 
# @File    : ExponentialMovingAverage.py
# @Software: PyCharm
#这里看一下tf中滑动平均模型怎么运作
'''
滑动平均模型     shadow_variable= decay*shadow_variable + (1-decay)*variable
其中decay衰减率一般设置为接近1的数，0.9    0.999  decay决定了模型的更新速度

如果调用函数ExponentialMovingAverage的时候指定了num_updates参数，那么每次使用的衰减率decay如下计算
decay = min{decay, (1+num_updates)/(10+num_updates)}
'''

import tensorflow as tf

#定义一个变量用于计算滑动平均，这个变量的初始值为0，注意一点，所有需要计算滑动平均的变量类型必须是实数型
v1 = tf.Variable(0, dtype=tf.float32)

#定义一个步长， 更新decay, 虽然是Variable类型，但是注意它不随着训练更新
step = tf.Variable(0, trainable=False)

#实例化一个滑动平均模型
ema = tf.train.ExponentialMovingAverage(0.9, step)

#每个变量都进行滑动平均
maintain_average_op = ema.apply([v1])
init_op = tf.global_variables_initializer()
#开始
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run([v1, ema.average(v1)])) #先看看初始值为0的时候滑动平均后的v1值

    sess.run(tf.assign(v1, 5)) # 将v1更新为5
    sess.run(maintain_average_op) #运行滑动平均计算，ema实例会根据公式重新计算
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 10000)) #更新一下num_updates,decay会根据公式更新
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
