#!-*- coding:utf-8 -*-
# author:zhupengfei
# datetime:2018/11/20 19:01
# software:PyCharm
# discribe:
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = './log'
BATCH_SIZE = 100
TRAIN_STEP = 30000

#生成变量监控信息
#入参 var 要监控的变量  name tensorboard显示的图标名称，一般与变量名相同
def variable_summaries(var, name):
    #将生成监控信息的操作放到同一个命名空间下
    with tf.name_scope('summaries'):
        #histogram记录var的取值分布
        tf.summary.histogram(name, var)
        #计算张量的均值和标准差，并监控
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/'+name, stddev)

#定义一个全连接层
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
    #同一个层放在同一个命名空间下
    with tf.name_scope(layer_name):
        #定义w
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')
        #定义bias
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(bias, layer_name + '/bias')

        #矩阵计算
        with tf.name_scope('Wx_plus_bias'):
            preactivate = tf.matmul(input_tensor, weights) + bias
            #记录网络经过激活之前的分布
            tf.summary.histogram(layer_name + '/preactivate', preactivate)
        activate = act(preactivate, name='activate')
        #记录激活之后的分布
        tf.summary.histogram(layer_name + '/activate', activate)
    return activate

#定义主函数
def main(args = None):
    #取数据
    mnist = input_data.read_data_sets('../dataset', one_hot=True)
    #x,y
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    #将输入图片数据还原成图片的矩阵形式，并写入日志
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input-image', image_shaped_input, 10)

    hidden1 = nn_layer(x, 784, 500, 'layer1', tf.nn.relu)
    y = nn_layer(hidden1, 500, 10, 'layer2', tf.identity)

    #定义损失函数
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)


    #计算正确率
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    #合并所有日志信息
    merge = tf.summary.merge_all()

    #设置GPU显存
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    #开始训练
    with tf.Session(config=config) as sess:
        #初始化日志记录
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()
        for i in range(TRAIN_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _ = sess.run([merge, train_step], feed_dict={x:xs, y_:ys})

            #将所有日志写入文件
            summary_writer.add_summary(summary, i)
            print('写入step：%d 完成'%i)
    summary_writer.close()

if __name__ == '__main__':
    tf.app.run()
