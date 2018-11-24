#!-*- coding:utf-8 -*-
# author:zhupengfei
# datetime:2018/11/22 7:23
# software:PyCharm
# discribe:
from keras.datasets import imdb
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words = 10000)
print('xx')