import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from PIL import Image
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置全局随机因子
tf.random.set_seed(22)
np.random.seed(22)
batchsz = 512
# 打印的日志等级 0：info，1：warning，2：error，3：fatal，
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 最低维度的隐藏向量的节点数
h_dim = 20
# 学习率
lr = 1e-3

# 数据准备
# (60000, 28, 28) (60000,)  |  (10000, 28, 28) (10000,) 数值的范围为[0,255]
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# 归一化
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# 训练集
# from_tensor_slices 创建数据集
train_db = tf.data.Dataset.from_tensor_slices(x_train)
print(train_db)
# 从数据集中按顺序抽取batchsz * 5个样本放在buffer中，然后打乱buffer中的样本,每次从buffer中抽取batchsz个样本，用于小批的进行梯度下降
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
print(train_db)
# 测试集
# 创建数据集
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 创建模型
# 继承 Model 类并在 call 方法中实现前向传播，创建你完全定制化的模型
# 编码器：输入图片x∈R(784)，3层全连接层网络，输出节点数分别为256、128、20
# 解码器：输入降维后的向量h∈R(20)，3层全连接层网络，输出节点数分别为128、256、784
# 每层使用ReLU激活函数，最后一层不适用激活函数
# Sequential：顺序模型，多个网络层的线性堆叠，可以通过向Sequential模型传递一个layer(层)的list来构造该模型
# layers.Dense：全连接层
# keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# units: 正整数，输出空间维度。
# activation: 激活函数 (详见 activations)。 若不指定，则不使用激活函数 (即，「线性」激活: a(x) = x)。
# use_bias: 布尔值，该层是否使用偏置向量。
# kernel_initializer: kernel 权值矩阵的初始化器 (详见 initializers)。
# bias_initializer: 偏置向量的初始化器 (see initializers).
# kernel_regularizer: 运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
# bias_regularizer: 运用到偏置向的的正则化函数 (详见 regularizer)。
# activity_regularizer: 运用到层的输出的正则化函数 (它的 "activation")。 (详见 regularizer)。
# kernel_constraint: 运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
# bias_constraint: 运用到偏置向量的约束函数 (详见 constraints)。
class AutoEncoder(keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # encoder
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])
        # decoder
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        # [b, 784] => [b, 10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat


def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


model = AutoEncoder()
# 训练模型
model.build(input_shape=(None, 784))
# 打印出模型的概述信息
model.summary()
# 优化器，定义学习率
optimizer = tf.optimizers.Adam(lr=lr)

# 整个数据集需要训练多少次 一次为一个 epoch
for epoch in range(100):
    for step, x in enumerate(train_db):
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])
        # GradientTape构建梯度计算环境， 被包裹的值可以自动求导
        with tf.GradientTape() as tape:
            # 模型生成的x
            x_rec_logits = model(x)
            # 使用交叉熵误差函数
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            # 误差的标量
            rec_loss = tf.reduce_mean(rec_loss)
        # 计算误差关于模型参数的导数(梯度)
        grads = tape.gradient(rec_loss, model.trainable_variables)
        # 更新网络参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, float(rec_loss))

        # 测试网络
        x = next(iter(test_db))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        # [b, 784] => [b, 28, 28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])

        # [b, 28, 28] => [2b, 28, 28]
        x_concat = tf.concat([x, x_hat], axis=0)
        x_concat = x_hat
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, 'ae_images/rec_epoch_%d.png' % epoch)
