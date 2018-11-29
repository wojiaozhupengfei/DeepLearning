#!-*- coding:utf-8 -*-
# author:zhupengfei
# datetime:2018/11/29 20:02
# software:PyCharm
# discribe:
from keras.preprocessing.image import ImageDataGenerator
from  file_process import *
import glog as log

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

for data_batch, label_batch in train_generator:
    log.info('data_batch_shape:{}'.format(data_batch.shape))
    log.info('label_batch_shape:{}'.format(label_batch.shape))
    break

