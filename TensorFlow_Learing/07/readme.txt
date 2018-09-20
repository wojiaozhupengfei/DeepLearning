tensorflow的数据读取环节

Tensorflow读取数据的一般方式有下面3种：
    preloaded直接创建变量：在tensorflow定义图的过程中，创建常量或变量来存储数据
    feed：在运行程序时，通过feed_dict传入数据
    reader从文件中读取数据：在tensorflow图开始时，通过一个输入管线从文件中读取数据
    上面的两个方法在面对大量数据时，都存在性能问题。这时候就需要使用到第3种方法，文件读取，让tensorflow自己从文件中读取数据

Preloaded方法的简单例子
    import tensorflow as tf

    """定义常量"""
    const_var = tf.constant([1, 2, 3])
    """定义变量"""
    var = tf.Variable([1, 2, 3])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(var))
        print(sess.run(const_var))

Feed方法
可以在tensorflow运算图的过程中，将数据传递到事先定义好的placeholder中。方法是在调用session.run函数时，通过feed_dict参数传入。简单例子：
    import tensorflow as tf
    """定义placeholder"""
    x1 = tf.placeholder(tf.int16)
    x2 = tf.placeholder(tf.int16)
    result = x1 + x2
    """定义feed_dict"""
    feed_dict = {
    x1: [10],
    x2: [20]
    }
    """运行图"""
    with tf.Session() as sess:
        print(sess.run(result, feed_dict=feed_dict))

步骤：
    1 获取文件名列表list
    2 创建文件名队列，调用tf.train.string_input_producer，参数包含：文件名列表，num_epochs【定义重复次数】，shuffle【定义是否打乱文件的顺序】
    3 定义对应文件的阅读器>* tf.ReaderBase >* tf.TFRecordReader >* tf.TextLineReader >* tf.WholeFileReader >* tf.IdentityReader >* tf.FixedLengthRecordReader
    4 解析器 >* tf.decode_csv >* tf.decode_raw >* tf.image.decode_image >* …
    5 预处理，对原始数据进行处理，以适应network输入所需
    6 生成batch，调用tf.train.batch() 或者 tf.train.shuffle_batch()
    7 prefetch【可选】使用预加载队列slim.prefetch_queue.prefetch_queue()
    8 启动填充队列的线程，调用tf.train.start_queue_runners

读取文件格式举例
tensorflow支持读取的文件格式包括：CSV文件，二进制文件，TFRecords文件，图像文件，文本文件等等。具体使用时，需要根据文件的不同格式，选择对应的文件格式阅读器，再将文件名队列传为参数，传入阅读器的read方法中。方法会返回key与对应的record value。将value交给解析器进行解析，转换成网络能进行处理的tensor。

CSV文件读取：
阅读器：tf.TextLineReader
解析器：tf.decode_csv
    filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
    """阅读器"""
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    """解析器"""
    record_defaults = [[1], [1], [1], [1]]
    col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.concat([col1, col2, col3, col4], axis=0)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(100):
            example = sess.run(features)
        coord.request_stop()
        coord.join(threads)

二进制文件读取：
阅读器：tf.FixedLengthRecordReader
解析器：tf.decode_raw

图像文件读取：
阅读器：tf.WholeFileReader
解析器：tf.image.decode_image, tf.image.decode_gif, tf.image.decode_jpeg, tf.image.decode_png

TFRecords文件读取
TFRecords文件是tensorflow的标准格式。要使用TFRecords文件读取，事先需要将数据转换成TFRecords文件，具体可察看：convert_to_records.py 在这个脚本中，先将数据填充到tf.train.Example协议内存块(protocol buffer)，将协议内存块序列化为字符串，再通过tf.python_io.TFRecordWriter写入到TFRecords文件中去。
阅读器：tf.TFRecordReader
解析器：tf.parse_single_example
又或者使用slim提供的简便方法：slim.dataset.Data以及slim.dataset_data_provider.DatasetDataProvider方法
    def get_split(record_file_name, num_sampels, size):
        reader = tf.TFRecordReader()

        keys_to_features = {
            "image/encoded": tf.FixedLenFeature((), tf.string, ''),
            "image/format": tf.FixedLenFeature((), tf.string, 'jpeg'),
            "image/height": tf.FixedLenFeature([], tf.int64, tf.zeros([], tf.int64)),
            "image/width": tf.FixedLenFeature([], tf.int64, tf.zeros([], tf.int64)),
        }

        items_to_handlers = {
            "image": slim.tfexample_decoder.Image(shape=[size, size, 3]),
            "height": slim.tfexample_decoder.Tensor("image/height"),
            "width": slim.tfexample_decoder.Tensor("image/width"),
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers
        )
        return slim.dataset.Dataset(
            data_sources=record_file_name,
            reader=reader,
            decoder=decoder,
            items_to_descriptions={},
            num_samples=num_sampels
        )


    def get_image(num_samples, resize, record_file="image.tfrecord", shuffle=False):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            get_split(record_file, num_samples, resize),
            shuffle=shuffle
        )
        [data_image] = provider.get(["image"])
        return data_image