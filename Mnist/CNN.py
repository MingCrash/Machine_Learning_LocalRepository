import logging
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

logger = logging.getLogger('__name__')

#-----------------------------------------------
#使用 双层CNN+感知机MLP 模型，准确率达到
#-----------------------------------------------

mnist_data = input_data.read_data_sets(train_dir='MNIST_DATA/', one_hot=True) #读取训练集

#数组样式
x = tf.placeholder(dtype=tf.float32,shape=[None, 784],name='input_tra_images')

# [batch, in_height, in_width, in_channels]
x_image = tf.reshape(tensor=x,shape=[-1,28,28,1],name='input_image')

y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])   #储存理论正确值,赋予随机分布的初始值

#创建卷积核权重矩阵
def Weight_variable(shape,stddev):
    initial = tf.truncated_normal(shape=shape,stddev=stddev)
    return tf.Variable(initial)

def Bias_variable(value,shape):
    initial = tf.constant(value=value,shape=shape)
    return tf.Variable(initial)

#进行卷积运算，strides=[1,stride,stride,1]
def conv2d(x, kernel):
    return tf.nn.conv2d(input=x, filter=kernel, strides=[1,1,1,1],padding='SAME')

#用简单传统的2x2池化做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#定义 32堆 层数1 大小5X5 的卷积核, 1个输入通道，32个输出通道，
#[filter_height, filter_width, in_channels, out_channels]
W_conv1 = Weight_variable(shape=[5,5,1,32],stddev=0.1)
#32个输出通道对应的偏移量
B_conv1 = Bias_variable(shape=[32],value=0.1)
W_conv2 = Weight_variable(shape=[5,5,32,64],stddev=0.1)
B_conv2 = Bias_variable(shape=[64],value=0.1)

# 接受x_image的输入，输出28X28等大的feature map
h_conv1 = tf.nn.relu(conv2d(x=x_image, kernel=W_conv1) + B_conv1)
# h_pool1 是 32张14X14的feature map
h_pool1 = max_pool_2x2(h_conv1)

# 接受14X14的feature map,输出等大的feature map
h_conv2 = tf.nn.relu(conv2d(x=h_pool1, kernel=W_conv2) + B_conv2)
# h_pool2 是 64张7X7的feature map
h_pool2 = max_pool_2x2(h_conv2)

W_mlp_1 = Weight_variable(shape=[7*7*64, 1024],stddev=0.1)
B_mlp_1 = Bias_variable(shape=[1024],value=0.1)
W_out = Weight_variable(shape=[1024, 10],stddev=0.1)
B_out = Bias_variable(shape=[10],value=0.1)

#将feature map重整为向量形式，进行内积计算
h_pool2_vector = tf.reshape(tensor=h_pool2,shape=[-1,7*7*64])
#密集连接层
mlp1 = tf.nn.relu(tf.matmul(h_pool2_vector,W_mlp_1)+B_mlp_1)

#防止过拟合，在输出层之前加入dropout处理,keep_prob代表一个神经元的输出在dropout中保持不变的概率
keep_prob = tf.placeholder(dtype=tf.float32)
# keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(x=mlp1, keep_prob=keep_prob)

#输出层
Y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_out)+B_out)

loss_cross_entropy = -tf.reduce_sum(y_*tf.log(Y_conv))
AdamOptimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_cross_entropy)
correct_predication = tf.equal(tf.argmax(Y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predication, 'float'))

tf.summary.scalar(name='Loss_cross_entropy',tensor=loss_cross_entropy)
tf.summary.scalar(name='Accuracy',tensor=accuracy)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=1)

    writer1 = tf.summary.FileWriter('logs/training', sess.graph)
    writer2 = tf.summary.FileWriter('logs/testing', sess.graph)

    test_images = mnist_data.test.images
    test_labels = mnist_data.test.labels
    for epoch in range(0,4):      #整份数据重复次数
        for i in range(0,55):
            batch_xs, batch_ys= mnist_data.train.next_batch(1000)
            _, train_result, mergedvision = sess.run(feed_dict={x:batch_xs, y_:batch_ys, keep_prob:0.5}, fetches=[AdamOptimizer,accuracy,merged])
            writer1.add_summary(summary=mergedvision,global_step=55*epoch+i)

            test_result,mergedvision = sess.run(fetches=[accuracy,merged], feed_dict={x: test_images, y_: test_labels, keep_prob:0.5})
            writer2.add_summary(summary=mergedvision,global_step=55*epoch+i)

    saver.save(sess=sess,save_path='Model/CNN_Model')
    logger.warning('save model!!')

























