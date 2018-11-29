#!-*- coding:utf-8 -*-
# author:zhupengfei
# datetime:2018/11/26 20:28
# software:PyCharm
# discribe:
import glog as log
import os, shutil
dirs = []
dirs_append = dirs.append
original_dataset_dir = r'../../datasets/Dog_and_Cat/kaggle猫狗大战数据集/train'
dirs.append(original_dataset_dir)

base_dir = './cats_and_dogs_small' #保存较小的数据集
dirs_append(base_dir)
# os.makedirs(base_dir)

# 创建train validation test目录
train_dir = os.path.join(base_dir, 'train')
dirs_append(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
dirs_append(validation_dir)
test_dir = os.path.join(base_dir, 'test')
dirs_append(test_dir)

#将猫和狗的三个目录分开
train_cats_dir = os.path.join(train_dir, 'cats')
dirs_append(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
dirs_append(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
dirs_append(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
dirs_append(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
dirs_append(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
dirs_append(test_dogs_dir)

for i in range(len(dirs)):
    if not os.path.exists(dirs[i]):
        os.makedirs(dirs[i])
    else:
        print('file {} is already exist'.format(dirs[i]))
#开始讲根目录的数据copy到各个子文件夹, 这里可以优化，现在来不及，先写出来
#1000张猫的训练集
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    else:
        pass
#500张猫的验证集
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    else:
        pass
#500张猫的测试集
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    else:
        pass

#1000张狗的训练集
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    else:
        pass
#500张狗的验证集
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    else:
        pass
#500张狗的测试集
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    else:
        pass

# log.info('total training cats images: %d'%(len(os.listdir(train_cats_dir))))
# log.info('total training dogs images: %d'%(len(os.listdir(train_dogs_dir))))
# log.info('total validation cats images: %d'%(len(os.listdir(validation_cats_dir))))
# log.info('total validation dogs images: %d'%(len(os.listdir(validation_dogs_dir))))
# log.info('total test cats images: %d'%(len(os.listdir(test_cats_dir))))
# log.info('total test dogs images: %d'%(len(os.listdir(test_dogs_dir))))







