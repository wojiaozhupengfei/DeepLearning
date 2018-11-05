#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/19/0019 14:31
# @Author  : zhupengfei
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm
#一个完整的图像数据预处理过程

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#随机调整颜色
def distort_color(image, color_ordering):
    if color_ordering ==0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image,lower=0.5, upper=1.5)
    if color_ordering ==1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image,lower=0.5, upper=1.5)
    if color_ordering ==2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image,lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
    if color_ordering == 3:
        image = tf.image.random_contrast(image,lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)

#对于解码后的图像数据根据bbox随机裁剪，返回模型要求的输入尺寸,只对训练数据做这个操作
def processes_for_train(image, height, width, bbox):
    #如果没有提供标注框，则保留整个图像
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], tf.float32, shape=[1, 1, 4])
    #转换图片tensor的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, tf.float32)
    #获取随机裁剪图像所要的参数begin和size
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bbox)
    distort_image = tf.slice(image, bbox_begin, bbox_size)

    #调整图像大小 适应网络输入
    distort_image = tf.image.resize_images(distort_image, [height, width], method=np.random.randint(4))

    #随机左右翻转
    distort_image = tf.image.random_flip_left_right(distort_image)
    #调整颜色
    distort_image = distort_color(distort_image, np.random.randint(4))

    return distort_image

image_raw_data = tf.gfile.FastGFile('./picture/mao.jpeg', 'rb').read()
with tf.Session() as sess:
    image_data = tf.image.decode_jpeg(image_raw_data, channels=3)
    bboxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    #运行6次
    for i in range(6):
        result = processes_for_train(image_data, 299, 299, bboxes)
        # plt.subplot(2, 3, i+1), plt.imshow(result.eval()), plt.title('%d'%(i+1))
        plt.imshow(result.eval()), plt.title('%d'%i)
        plt.show()





