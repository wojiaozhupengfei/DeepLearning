#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/19/0019 7:20
# @Author  : zhupengfei
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm

#图片的解码和编码过程
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from log import *

#读入原始图的数据,注意这里有bug报错，‘UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte’
#将tf.gfile.FastGFile('./picture/mao.jpeg', 'r')替换为tf.gfile.FastGFile('./picture/mao.jpeg', 'rb')
image_raw_data = tf.gfile.FastGFile('./picture/mao.jpeg', 'rb').read()
# logger.info(image_raw_data)

#tensorflow 还提供了一个解码png格式的函数 tf.image.decode_png
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data, channels = 3)
    # logger.info(img_data.eval())
    #打印图片
    plt.imshow(img_data.eval()) #要对tensor执行
    plt.show()

    #将表示一张图片的tensor编码为图片
    # encoded_image = tf.image.encode_jpeg(img_data)
    # with tf.gfile.FastGFile('./picture/output.jpeg', 'wb') as f:
    #     f.write(encoded_image.eval())

    #对图片tensor进行尺寸调整,该函数进行了归一化，图片值域0。0-1.0 如果不这么转换，就没有进行归一化，图片值域0-255
    # img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    # logger.info(img_data.eval())
    #method是图片大小的调整方法，0：双线性插值法 1：最近邻法 2:双三次插值法 3：面积插值法
    # resized0 = tf.image.resize_images(img_data, [300, 300], method=0)
    # plt.imshow(resized0.eval()) #要对tensor执行
    # plt.show()
    # resized1 = tf.image.resize_images(img_data, [300, 300], method=1)
    # plt.imshow(resized1.eval()) #要对tensor执行
    # plt.show()
    # resized2 = tf.image.resize_images(img_data, [300, 300], method=2)
    # plt.imshow(resized2.eval()) #要对tensor执行
    # plt.show()
    # resized3 = tf.image.resize_images(img_data, [300, 300], method=3)
    # plt.imshow(resized3.eval()) #要对tensor执行
    # plt.show()

    #对图片大小调整的填充和裁剪，不够就填充0  多了就裁剪只保留图片中心, 原图800*532
    # croped = tf.image.resize_image_with_crop_or_pad(img_data, 500, 500)
    # plt.imshow(croped.eval()) #要对tensor执行
    # plt.show()
    # paded = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    # plt.imshow(paded.eval()) #要对tensor执行
    # plt.show()

    # #图片上下翻转
    # filpped1 = tf.image.flip_up_down(img_data)
    # #50%概率上下翻转
    # filpped1_random = tf.image.random_flip_up_down(img_data)
    # #图片左右翻转
    # filpped2 = tf.image.flip_left_right(img_data)
    # #50%概率左右翻转
    # filpped2_random = tf.image.random_flip_left_right(img_data)
    # #图片对角线翻转
    # filpped3 = tf.image.transpose_image(img_data)
    # plt.imshow(filpped1.eval())
    # plt.show()
    # plt.imshow(filpped2.eval())
    # plt.show()
    # plt.imshow(filpped3.eval())
    # plt.show()

    # #图片的亮度调整
    # adjusted = tf.image.adjust_brightness(img_data, -0.5)
    # adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
    # plt.imshow(adjusted.eval())
    # plt.show()
    #
    # #在-max max 范围内随机调整亮度
    # adjusted_random = tf.image.random_brightness(img_data, 0.7)
    # adjusted_random = tf.clip_by_value(adjusted_random, 0.0, 1.0)
    # plt.imshow(adjusted_random.eval())
    # plt.show()
    #
    # #调整对比度
    # adjusted = tf.image.adjust_contrast(img_data, 0.5) #0.5倍
    # adjusted = tf.image.random_contrast(img_data, 1, 5)# 1-5范围内随机调整对比度
    #
    # #调整色相
    # adjusted = tf.image.adjust_hue(img_data, 0.1) #+0.1
    # adjusted = tf.image.random_hue(img_data, 0.3) #-0.3  0.3 之间调整色相
    #
    # #调整饱和度一样的  tf.image.adjust_saturation
    #
    # #标准化图片,tensor变为均值0  方差1 的正太分布
    # adjusted = tf.image.per_image_standardization(img_data)

    #给图像加入标注框  bounding # box
    #先将图片尺寸缩小一些，这样标注框清楚一些
    # img_data = tf.image.resize_images(img_data, [180, 267], method=1)
    # #tf.image.draw_bounding_boxes要求图像矩阵的数字为实数（归一化）,并且支持多张图片一起画框，所以图片输入维度变为四维
    # img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    # batched = tf.expand_dims(img_data, 0)
    # #boxes的位置 四个值，分别代表[ymin, xmin, ymax, xmax],比如下面的ymin是（180*0.05，180*0.35）
    # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7]]])
    # #注意这里有一个bug，tf.image.draw_bounding_boxes返回的是batch个图片的boundingbox  所以我们只有一张图片，可视化就是result[0]
    # result = tf.image.draw_bounding_boxes(batched,boxes)
    # plt.subplot(121), plt.imshow(img_data.eval()), plt.title('original')
    # plt.subplot(122), plt.imshow(result[0].eval()), plt.title('Now')
    # plt.show()

    #通过标注框来截取图像
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
#     '''
#     image_size： 是包含 [height, width, channels] 三个值的一维数组
#     bounding_boxes： 是一个 shape 为 [batch, N, 4] 的三维数组，数据类型为float32，第一个batch是因为函数是处理一组图片的，
#     N表示描述与图像相关联的N个边界框的形状，而标注框由4个数字 [y_min, x_min, y_max, x_max]表示出来。
#     # min_object_covered 表示boundingbox中至少要有这么多的内容被截取到
#     Return：一个Tensor对象的元组（begin，size，bboxes）
#     begin： 和 image_size 具有相同的类型。包含 [offset_height, offset_width, 0] 的一维数组。作为 tf.slice 的输入。
#     size： 和 image_size 具有相同的类型。包含 [target_height, target_width, -1] 的一维数组。作为 tf.slice 的输入。
#     bboxes：shape为 [1, 1, 4] 的三维矩阵，数据类型为float32，表示随机变形后的边界框。作为 tf.image.draw_bounding_boxes 的输入。
#     '''
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data), bounding_boxes=boxes, min_object_covered=0.8)
    #通过标注框可视化随机截取到的图像
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    distorted_image = tf.slice(img_data, begin, size)
    plt.subplot(121), plt.imshow(image_with_box[0].eval()), plt.title('image_with_box')
    plt.subplot(122), plt.imshow(distorted_image.eval()), plt.title('distorted_image')
    plt.show()
