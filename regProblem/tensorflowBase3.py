import tensorflow as tf
import numpy as np
# 加入忽略
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 广播
# 广播机制是一种轻量级的张量复制手段，在逻辑上拓展张量的形状，在需要时才去执行存储复制操作
x = tf.random.uniform([2, 4], maxval=10)
w = tf.random.uniform([4, 3], maxval=10)
b = tf.random.uniform([3], maxval=10)
# x @ w = [2,3] b = [3] 两者直接相加是自动调用了广播函数 tf.broadcast_to(x,new_shape)
# 等效于 y=x @ w + tf.broadcast(b,[2,3])
y = x @ w + b
print(y)

# 基本的加减乘除运算已经被tensorFlow重载可直接使用运算符
a = tf.range(5)
b = tf.constant(2)
print(a / b)

# 乘方运算 tf.pow(a,x) a**x
a = tf.range(4)
print(a)
print(tf.pow(a, 3))
print(a ** 3)
# 平方
tf.square(a)

# 指数运算 tf.pow(x,a) x**a
print(tf.pow(3, a))
