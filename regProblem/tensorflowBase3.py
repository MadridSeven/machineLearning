import tensorflow as tf
import numpy as np

# 广播
# 广播机制是一种轻量级的张量复制手段，在逻辑上拓展张量的形状，在需要时才去执行存储复制操作
x = tf.random.uniform([2, 4], maxval=10)
w = tf.random.uniform([4, 3], maxval=10)
b = tf.random.uniform([3], maxval=10)
# x @ w = [2,3] b = [3] 两者直接相加是自动调用了广播函数 tf.broadcast_to(x,new_shape)
# 等效于 y=x @ w + tf.broadcast(b,[2,3])
y = x @ w + b
print(y)
