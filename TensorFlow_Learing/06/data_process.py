#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/14/0014 10:17
# @Author  : zhupengfei
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm
#这个文件用于迁移学习中对原始图像的处理，将原始图片处理为网络结构能够接受的数据集

import tensorflow as tf
import os.path
import glob
import numpy as np
from tensorflow.python.platform import gfile
import time

#配置全局参数
INPUT_DATA_PATH = '../dataset/flower_photos/flower_photos' #输入数据集路径
OUTPUT_FILE = '../dataset/flower_photos/flower_processed_data.npy' #处理好的数据输出路径
VALIDATION_PERCENTAGE = 10 #验证集比例
TEST_PERCENTAGE = 10 #测试集比例

#读取数据并将数据划分为训练集，验证集，测试集
def create_image_lists(sess, testing_precentage, validation_precentage):
    # 获取所有子文件，包括子文件中的所有文件，这里希望得到的是每种花的目录名称
    '''
    ../dataset/flower_photos/flower_photos
    ../dataset/flower_photos/flower_photos\daisy
    ../dataset/flower_photos/flower_photos\dandelion
    ../dataset/flower_photos/flower_photos\roses
    ../dataset/flower_photos/flower_photos\sunflowers
    ../dataset/flower_photos/flower_photos\tulips
    '''
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA_PATH)]
    is_root_dir = True#sub_dirs里面包含了根目录，为了跳过根目录

    #初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validatino_labels = []
    current_label = 0

    start_time_sub_dir = time.time()
    #读取所有子目录
    for sub_dir in sub_dirs:
        #先跳过根目录
        if is_root_dir:
            is_root_dir = False
            continue
        #获取子目录中所有的图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG'] #文件扩展名
        # '''
        # 比如：
        # ['../dataset/flower_photos/flower_photos\\daisy\\100080576_f52e8ee070_n.jpg',
        # '../dataset/flower_photos/flower_photos\\daisy\\10140303196_b88d3d6cec.jpg',
        # '../dataset/flower_photos/flower_photos\\daisy\\10172379554_b296050f82_n.jpg',
        # '../dataset/flower_photos/flower_photos\\daisy\\10172567486_2748826a8b.jpg',
        # '../dataset/flower_photos/flower_photos\\daisy\\10172636503_21bededa75_n.jpg',
        # '../dataset/flower_photos/flower_photos\\daisy\\102841525_bd6628ae3c.jpg',
        # '../dataset/flower_photos/flower_photos\\daisy\\1031799732_e7f4008c03.jpg',..]
        # '''

        file_list = [] #所有图片路径列表
        #获取每种花所在文件夹的名称：比如 daisy
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA_PATH, dir_name, '*.'+extension)#每张图片的具体路径
            file_list.extend(glob.glob(file_glob))#所有匹配到的图片都加入到file_list
        if not file_list:
            continue

        length = len(file_list)
        start_time_file_name = time.time()
        #处理该种类文件夹内的所有花的图片，主要调用gfile来处理, 图片格式为299*299，以便inception-v3模型来处理
        for index_file_name,file_name in enumerate(file_list):
            print('start %s file_name_index is %d-%d'%(sub_dir, index_file_name, length))
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()#二进制数据
            image = tf.image.decode_jpeg(image_raw_data)#将二进制数据解码为数据矩阵，格式为W*H*C 数据类型为uint8
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image) #数据格式为299*299*3 类型float32

            #开始划分数据集
            chance = np.random.randint(100) #产生一个0-99的随机整数，根据这个将该图片划分到哪个数据集
            if chance < validation_precentage:
                validation_images.append(image_value)
                validatino_labels.append(current_label) #这个文件夹下的所有图片都是一类花，默认给第一个文件夹下的花打标签为0
            elif chance < (testing_precentage + validation_precentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1 #到下一个文件夹的时候标签就+1 所以五个文件夹的标签分别为0、1、2、3、4代表5中类别的花
        end_time_file_name = time.time()
        print('\n finished %s, time is %d \n'%(sub_dir, end_time_file_name-start_time_file_name))
    #数据集分好之后，将训练数据打乱，以获得更好的训练效果
    state = np.random.get_state() #产生一种打乱的状态
    np.random.shuffle(training_images) #根据这种打乱状态打乱数据
    np.random.set_state(state) #设置打乱状态和上一次一样，保证了训练数据打乱之后，和标签对应关系还是对的，因为打乱方式一样
    np.random.shuffle(training_labels)
    end_time_sub_dir = time.time()
    print('\n data process finished!!  time is %d \n'%(end_time_sub_dir - start_time_sub_dir))

    return np.array([training_images, training_labels, validation_images, validatino_labels, testing_images, testing_labels])




def main(argv):
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        #通过numpy格式保存处理后的数据
        np.save(OUTPUT_FILE,processed_data)

if __name__ == '__main__':
    tf.app.run()