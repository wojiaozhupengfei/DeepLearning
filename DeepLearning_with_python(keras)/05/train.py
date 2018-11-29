#!-*- coding:utf-8 -*-
# author:zhupengfei
# datetime:2018/11/29 20:27
# software:PyCharm
# discribe:
from nets import *
from dataa_process import *

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)
model.save('cats_and_dogs_small_1.h5')
