import tensorflow as tf
import numpy as np
# 加入忽略
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建张量
# Numpy Array 数组和 Python List 列表是 Python 程序中间非常重要的数据载体容器，
# 很多数据都是通过 Python 语言将数据加载至 Array 或者 List 容器，再转换到 Tensor 类型，
# 通过 TensorFlow 运算处理后导出到 Array 或者 List 容器，方便其他模块调用。
# tf.convert_to_tensor、tf.constant 函数可以创建新 Tensor，并将保存在 Python List 对象或者 Numpy Array 对象中的数据导入到新 Tensor 中

# 从列表创建张量
a = tf.convert_to_tensor([1, 2.])
print(a)
# 从数组中创建张量
b = tf.convert_to_tensor(np.array([[0, 1], [1, 2]]))
print(b)

# 创建为0和为1的标量
c = tf.zeros([])
d = tf.ones([])
print(c), print(d)

# 创建为0和为1的向量
c = tf.zeros([1])
d = tf.ones([1])
print(c), print(d)

# 创建为0和为1的矩阵
c = tf.zeros([2])
d = tf.ones([2])
print(c), print(d)

# 创建一个与某张量形状相同，但是全0或全1的新张量
a = tf.ones([2, 3])
b = tf.zeros_like(a)
print(a), print(b)

# 创建自定义数值的张量
# 创建所有元素为-1的向量 所有元素为99的矩阵
print(tf.fill([1], -1))
print(tf.fill([2, 2], 99))
