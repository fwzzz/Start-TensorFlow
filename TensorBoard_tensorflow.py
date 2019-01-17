import tensorflow as tf

# 运行TensorBoard，并将日志的地址只想程序日志输出的地址。命令如下：
# tensorboard --logdir=path/to/log


###############################
#   Section1： 加法计算的图     #
###############################

# # 定义一个简单的计算图，实现向量加法的操作
# input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
# input2 = tf.Variable(tf.random_uniform([3]), name="input2")
# output = tf.add_n([input1, input2], name="add")
#
# # 生成一个斜日志的 writer，并将当前的 TensorFlow 计算图写入日志。
# # TensorFlow 提供了多种写日志文件的API
# writer = tf.summary.FileWriter("path/to/log", tf.get_default_graph())
# writer.close()


###############################
#   Section2： 管理命名空间     #
###############################

# 将输入定义放入各自的命名空间中，从而使得TensorBoard可根据命名空间来整理可视化效果图上的节点
with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")
writer = tf.summary.FileWriter("path/to/log", tf.get_default_graph())
writer.close()











