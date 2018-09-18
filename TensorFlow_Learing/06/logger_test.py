#!/user/bin/env python
# -*- coding: utf-8 -*-
#@Time     :2018/9/16/0016 16:32
#@Author   :zhupengfei
#@Site     :
#@File     :logger_test.py
#@Software :PyCharm
#@descibe  ：log日志测试
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

# #这里设置了logger的显示格式，filename指定了日志输出文件，将日志打印到该文件
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(name)s-%(levelno)s-%(levelname)s-%(lineno)d-%(module)s    %(message)s', filename='output.log')
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)

# sys.exit()
# a = 2
# s = 'hahahaha'
#
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
#
# #streamHandler
# # stream_handler = logging.StreamHandler(sys.stdout)
# # stream_handler.setLevel(level=logging.DEBUG)
# # logger.addHandler(stream_handler)
#
# #FileHandler
# file_handler = logging.FileHandler('output.log')
# file_handler.setLevel(level = logging.INFO)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
#
# #log
# logger.info('Start')
# logger.warning('Something maybe fail.')
# try:
# 	result = 10 / 0
# except Exception:
# 	logger.error('Faild to get result', exc_info = True)
# 	logger.info('Finished')
#
# sys.exit()
# # HTTPHandler
# # http_handler = logging.HTTPHandler(host = 'localhost:8001', url = 'log', method = 'PSOT')
# # logger.addHandler(http_handler)
#
# logger.info('the a number is %d and s is %s'%(a, s))
# logger.info('the a number is {} and s is {}'.format(a, s))
# logger.debug('Debugging')
# logger.warning('Warning exists')
# logger.info('Finished')
#
# sys.exit()
#
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
# handler = logging.FileHandler('output.log')
# formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# logger.info('the a number is %d and s is %s'%(a, s))
# logger.info('the a number is {} and s is {}'.format(a, s))
# logger.debug('Debugging')
# logger.warning('Warning exists')
# logger.info('Finished')
#
#
# sys.exit()
# #这里设置了logger的显示格式，filename指定了日志输出文件，将日志打印到该文件
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(name)s-%(levelno)s-%(levelname)s-%(lineno)d-%(module)s    %(message)s', filename='output.log')
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
#
# handler = logging.FileHandler('output.log')
# logger.info('the a number is %d and s is %s'%(a, s))
#
#
# sys.exit()
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(name)s-%(levelno)s-%(levelname)s-%(lineno)d-%(module)s    %(message)s')
# logger = logging.getLogger(__name__)
#
# a = 2
# s = 'hahahaha'
#
# logger.info('the a number is %d and s is %s'%(a, s))
# logger.info('the a number is {} and s is {}'.format(a, s))
# logger.debug('Debugging')
# logger.warning('Warning exists')
# logger.info('Finished')

