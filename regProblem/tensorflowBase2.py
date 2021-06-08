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

# 创建形状为shape，均值为mean，标准差为stddev的正态分布
print(tf.random.normal([2, 2], mean=1, stddev=2))
# 创建采样自区间[0,10)，shape为[2,2] 的均匀分布矩阵
print(tf.random.uniform([2, 2], maxval=10))

# 创建序列
# 创建 0-10 不包含10 步长为1的序列
print(tf.range(10))
# 创建 0-10 不包含10 步长为2的序列
print(tf.range(10, delta=2))
# 创建 1-10 不包含10 步长为2的序列
print(tf.range(1, 10, delta=2))

# 均方误差
out = tf.random.uniform([4, 10])
y = tf.constant([2, 3, 2, 1])
y = tf.one_hot(y, depth=10)
# 计算每个样本的MSE
loss = tf.keras.losses.mse(y, out)
print(loss)
loss = tf.reduce_mean(loss)
print(loss)

# 向量
# 模拟获得激活函数的输入z
z = tf.random.normal([4, 2])
# 创建偏置向量
b = tf.zeros([2])
z = z + b

# 张量
# 创建32*32的彩色图片输入，个数为4
x = tf.random.normal([4, 32, 32, 3])
# 创建卷积神经网络
layer = tf.keras.layers.Conv2D(16, kernel_size=3)
# 前向计算
out = layer(x)
print(out)

# 索引
# 4张32*32大小的彩色图片
x = tf.random.normal([4, 32, 32, 3])
# 取一张图片的数据
print(x[0])
# 取第一张图片的第2行
print(x[0][1])
# 取第一张图片的第二行第三列
print(x[0][1][2])
# 取第一张图片的第二行第三列，B通道(第二个通道)的颜色强度
print(x[0][1][2][1])

# 切片
# start(开始读取位置的索引):end(结束的索引(不包含)):step(步长)


