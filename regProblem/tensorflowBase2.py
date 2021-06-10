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
# 取4张32*32大小的彩色图片的第2,3张图片
print(x[1:3:1])

# 维度变换
# 改变视图 改变张量的视图，仅仅是改变了张量的理解方式，并不需要该表张量的存储顺序(可以节省计算资源)
# 改变视图的前提是存储不需要改变
# 新试图的元素总理与存储区域大小要相等，即新试图的元素数量等于，b*h*w*c
x = tf.range(96)  # 生成向量
print(x)
x = tf.reshape(x, [2, 4, 4, 3])  # 改变x的视图，获得4D张量(存储并未改变)
print(x)
# 张量的维度数和形状
print(x.ndim), print(x.shape)

# 增、删维度
# 例：28*28的灰度图像，shape为[28,28]，增加一个新的维度，定义为通道数，则shape为[28,28,1]
x = tf.random.uniform([28, 28], maxval=10, dtype=tf.int32)
print(x)
# 通过 tf.expand_dims(x,axis) 可在指定的axis轴前可以插入一个新的维度
# 在宽维度的后面新增一个维度，定义为通道数，只能增加长度为1的维度
x = tf.expand_dims(x, axis=2)
print(x)
# 在最前面插入一个维度，定义为图片数量维度
x = tf.expand_dims(x, axis=0)
print(x)

# 删除维度
# tf.squeeze(x,axis),axis为待删除维度的索引号(如果不指定会删除所有长度为1的维度)
x = tf.squeeze(x, axis=0)
print(x)

