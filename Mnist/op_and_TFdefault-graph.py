import tensorflow as tf
import tensorflow.python.framework.ops

# #下面的op在tensorflow默认图
# matrix1 = tf.constant([[3.0,3.0]]) #创建常量op,返回一个矩阵
# matrix2 = tf.constant([[2.0],[2.0]]) #创建另一个常量op，产生2*1矩阵
# print(type(matrix2),matrix2.shape,matrix2,matrix2._rank)
#
# # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# product = tf.matmul(matrix1,matrix2)
#
# #创建Session对象，来启动默认图
# with tf.Session() as sess:
#     result = sess.run(product)
#     print(result)

# state = tf.Variable(0,name='counter')
one = tf.constant(1,name='add')
state = tf.Variable(0,name='counter')
new_value = tf.add(state,one)
update = tf.assign(state,new_value)
print(type(one),one)
print(type(state),state)
print(type(new_value),new_value)

#启动图后，变量必须先经过'初始化'op 初始化
#首先必须先增加一个'初始化'op到图中
init_op = tf.global_variables_initializer()

# with tf.Session() as ss:
#     ss.run(init_op)
#     print(ss.run(state))
#     for _ in range(3):
#         ss.run(update)
#         print(ss.run(state))

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
# input3 = tf.Variable(2,name='ddd')
add_op = tf.add(input1,input2)
mul_op = tf.multiply(input1,add_op)
with tf.Session() as ddd:
    ddd.run(init_op)
    print(ddd.run([mul_op,add_op]))
