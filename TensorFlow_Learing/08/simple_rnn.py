#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/21/0021 14:47
# @Author  : zhupengfei
# @Site    : 
# @File    : simple_rnn.py
# @Software: PyCharm
#一个简单的RNN前向的过程

import numpy as np
import sys
import os
sys.path.append('../07/log') #引入日志文件
from log import *


#输入向量，两个值
x = [1, 2]
#初始状态
state = [0.0, 0.0]
#不同部分的权重 !!!!!注意！！！！！nasdarray和asarray的区别，ndarray是整数，asarray才可以方浮点型
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])#状态权重
w_cell_input = np.asarray([0.5, 0.6])#输入权重
b_cell = np.asarray([0.1, -0.1]) #偏置

#定义输出的全连接参数
w_output = np.asarray([[1.0], [2.0]])
b_output = np.asarray([0.1])

#按照时间序列执行RNN的前向过程 输入x是两个，所有循环两次
for i in range(len(x)):
    before_activity = np.dot(state, w_cell_state)+x[i] *w_cell_input + b_cell
    state = np.tanh(before_activity)

    #根据当前状态计算该时间节点的输出
    final_output = np.dot(state, w_output) + b_output

    logger.info('before_activity is %s'%before_activity)
    logger.info('state is %s'%state)
    logger.info('final_output is %s'%final_output)