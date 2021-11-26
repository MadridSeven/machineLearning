import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 张量合并
# 拼接方式
# 可以在任何维度上合并，非合并维度长度必须一致
# tf.concat(tensors,axis)函数，tensors保存所有需要合并的张量List，axis参数指定需要合并的维度
# 如[4,35,8]和[6,35,8]合并为[10,35,8]
a = tf.random.normal([4, 35, 8])
b = tf.random.normal([6, 35, 8])
c = tf.concat([a, b], axis=0)
print(c.shape)

# 堆叠方式
# 待合并的shape必须完全一致
# tf.stack(tensors,axis)函数，tensors保存所有需要合并的张量List,axis指定新维度插入的位置
# 如 [35,8] + [35,8] = [2,35,8]
a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
c = tf.stack([a, b], axis=0)
print(c.shape)

# 分割
# 分割是合并的反向操作
# tf.split(x, num_or_size_splits, axis)
# x保存张量，num_or_size_splits表示切割方案，当是单个数字的时候，表示等长切割为十份，axis指定分割维度的索引号
a = tf.random.normal([10, 35, 8])
b = tf.split(a, num_or_size_splits=10, axis=0)
print(b)

# 数据统计
# 向量范数： L1范数，向量中所有元素绝对值之和 L2范数，向量中所有元素的平方和开根号，np.inf(无穷范数)向量中所有元素绝对值的最大值
# tf.norm(x,ord) 求解张量的范数，ord为1、2时 表示计算L1、L2范数，指定为np.inf时为无穷范数
x = tf.ones([2, 2])
print(tf.norm(x, ord=1))
print(tf.norm(x, ord=2))
print(tf.norm(x, ord=np.inf))

# 最值、均值、和
# 考虑shape为[4,10]的张量，第一个维度表示样本数量，第二个维度表示当前样本分别属于10个类别的概率
x = tf.random.normal([4, 10])
# 统计概率维度上的最大值,返回长度为4的向量，分表表示了每个样本最大的概率值
print(tf.reduce_max(x, axis=1))

