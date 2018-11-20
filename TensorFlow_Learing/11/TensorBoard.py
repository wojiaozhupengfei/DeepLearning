import tensorflow as tf
import sys
sys.path.append('../05/mnist_inference.py')
from mnist_inference import *
from tensorflow.examples.tutorials.mnist import input_data
import os
model_save_path = './model'
model_name = 'model.ckpt'


# input1 = tf.constant([1., 2., 3.], name='input1')
# input2 = tf.Variable(tf.random_uniform([3]), name='input2')
# output = tf.add_n([input1, input2], name='add')
#
# writer = tf.summary.FileWriter('./log', tf.get_default_graph())
# writer.close()

# with tf.variable_scope('foo'):
#     a = tf.get_variable('bar', [1])
#     print(a.name) #foo/bar:0
#
# with tf.variable_scope('bar'):
#     a = tf.get_variable('bar', [1])
#     print(a.name)
#
# with tf.name_scope('a'):
#     a = tf.Variable([1])
#     print(a.name)
#
# a = tf.get_variable('b', [1])
# print(a.name)
#
# #这里注意，name_scope并不能约束get_variable产生的变量，相当于c还是全局命名空间的
# with tf.name_scope('ddd'):
#     c=tf.get_variable('c', [1])
#     print(c.name)

# with tf.name_scope('input1_scope'):
#     input1 = tf.constant([5., 7., 3.], name='input1')
#     print(input1.name)
# with tf.name_scope('input2_scope'):
#     input2 = tf.Variable(tf.random_uniform([3]), name='input2')
#     print(input2.name)
#
# output = tf.add_n([input1, input2], name='output_scope')
# writer = tf.summary.FileWriter('./log', tf.get_default_graph())
# writer.close()



#定义训练过程中的全局变量
TRAINING_STEP = 30000  #一共进行30000次训练，这里和epoch不同，注意一下
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8 # 优化器的基础学习率
LEARNING_RATE_DECAY = 0.99 #优化器学习率的衰减系数
REGURLARIZER_RATE = 0.0001 #正则项的惩罚系数
MOVING_AVERAGE_DACAY  = 0.99 #滑动平均模型的衰减系数
display_step = 1000  #每1000次打印一下训练集的损失并保存模型

def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-cinput')
    regularizer = tf.contrib.layers.l2_regularizer(REGURLARIZER_RATE)
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    #将处理滑动平均相关的计算都放在一个命名空间下
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DACAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #将计算损失函数的相关计算都房子一个命名空间下
    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #将定义学习率，优化方法，每一轮训练需要执行的操作都放在一个命名空间下
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
    #实例化模型持久化类
    saver = tf.train.Saver()

    #设置GPU显存不超过30%
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    #开始训练
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            #每1000次记录一次运行状态
            if i%1000 == 0:
                #配置运行时需要记录的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #运行时记录运行信息的proto
                run_metadata = tf.RunMetadata()
                #将配置信息和运行信息的proto传入运行过程，从而记录运行过程中每一个节点的时间、空间开销信息
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys}, options=run_options, run_metadata=run_metadata)
                #将节点在运行时的信息写入日志文件
                writer = tf.summary.FileWriter('./log', tf.get_default_graph())
                writer.add_run_metadata(run_metadata, 'step%03d'%i)
                #每1000次打印损失及保存模型
                if i % display_step == 0:
                    print('After %d training step, loss on training batch is %g, global step is %d'%(i, loss_value, step))
                    saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

    # writer = tf.summary.FileWriter('./log', tf.get_default_graph())
    writer.close()

def main(argv = None):
    mnist = input_data.read_data_sets('../dataset', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
