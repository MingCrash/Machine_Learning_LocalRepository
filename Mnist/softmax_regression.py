import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
# import sys

#-----------------------------------------------
#使用单层全连接感知机MLP模型，准确率达到 90% ~ 91%
#-----------------------------------------------

mnist_data = input_data.read_data_sets(train_dir='MNIST_DATA/', one_hot=True) #读取训练集

tra_images = mnist_data.train.images        #[55000, 784]
tra_labels = mnist_data.train.labels        #[55000, 10]
test_images = mnist_data.test.images
test_labels = mnist_data.test.labels

x = tf.placeholder(dtype=tf.float32,shape=[None, 784],name='input_tra_images')  #输入值  像素矩阵

W = tf.Variable(tf.zeros([784,10]))  #权重矩阵 设置全零初始值
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)  #输出模型实际值
tf.summary.histogram(name='Weight',values=W)

y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])   #储存理论正确值,赋予随机分布的初始值
# 交叉熵  -∑y_(x)log(y(x))
loss_cross_entropy = -tf.reduce_sum(y_*tf.log(y))
tf.summary.scalar(name='Loss_cross_entropy',tensor=loss_cross_entropy)

# 定义Adagrad梯度下降优化器
AdagradOptimizer = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(loss_cross_entropy)
init_op = tf.global_variables_initializer()

#argmax(取向量中最大值的位置(以one-hot-value形式展示)),tf.equal( 返回[True,False,True,True] )
correct_predication = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy(正确率) cast(把布尔转化为浮点数 [True,False,True,True]--> [1,0,1,1] ) reduce_mean(均值)
accuracy = tf.reduce_mean(tf.cast(correct_predication, 'float'))
tf.summary.scalar(name='Accuracy',tensor=accuracy)


with tf.Session() as sess:
    sess.run(init_op)
    #随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/training',sess.graph)
    writer2 = tf.summary.FileWriter('logs/testing',sess.graph)
    for i in range(55):
        batch_xs, batch_ys = mnist_data.train.next_batch(1000)
        _, train_result,mergedvision = sess.run(fetches=[AdagradOptimizer,accuracy,merged], feed_dict={x:batch_xs,y_:batch_ys})
        writer.add_summary(mergedvision,i)

        test_result,mergedvision = sess.run(fetches=[accuracy,merged], feed_dict={x: test_images, y_: test_labels})
        writer2.add_summary(mergedvision,i)


