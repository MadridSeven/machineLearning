import tensorflow as tf
# 加入忽略
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建标量
# python方式创建标量
a = 1.2
# TF方式创建标量
b = tf.constant(1.2)
print(type(a)), print(type(b)), print(tf.is_tensor(b))
print(a), print(b)

# 创建向量
# 创建一个单元素向量
c = tf.constant([1.2])
print(c), print(c.shape)
# 创建三个元素的向量
d = tf.constant([1, 2, 3.])
print(d)

# 定义矩阵
# 定义两行两列的矩阵
e = tf.constant([[1, 2], [3, 4]])
print(e)

# 创建三维张量
f = tf.constant([[1, 2], [3, 4]])