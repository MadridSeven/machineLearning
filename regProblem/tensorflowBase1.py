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
f = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f)

# 创建字符串
g = tf.constant('=Hello,TensorFlow')
print(tf.strings.lower(g))

# 布尔类型标量
h = tf.constant(True)
print(h)
# 布尔类型向量
h = tf.constant([True, False])
print(h)

# 指定数值类型张量的精度
print(tf.constant(123456, dtype=tf.int32))
print(tf.constant(123456, dtype=tf.int64))

# 需要优化、计算梯度的张量需要用tf.Variable()封装，以便跟踪相关梯度信息，自动求导
p = tf.constant([-1, 0, 1, 2])
q = tf.Variable(p)
r = tf.Variable([[1, 2], [3, 4]])
print(q), print(r)
