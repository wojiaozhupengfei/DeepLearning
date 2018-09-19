#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/19/0019 7:31
# @Author  : zhupengfei
# @Site    : 
# @File    : log.py
# @Software: PyCharm
import logging
import sys

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s-%(lineno)d-    %(message)s')
# logger = logging.getLogger(__name__)

#建立logger并初始化一些格式
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(lineno)d-    %(message)s')

# streamHandler 日志输出到控制台
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(level=logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#FileHandler  日志输出到文件
file_handler = logging.FileHandler('transform_train_log.log', mode='a')
file_handler.setLevel(level = logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)