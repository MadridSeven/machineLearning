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
