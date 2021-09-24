import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.datasets as datasets
# 加入忽略
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 三层神经网络 out=ReLU{ReLU{ReLU{X@W1+B1}@W2+B2}@W3+B3}

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False


# 初始化数据
def load_data():
    # 加载MNIST数据集
    # x是像素集合 y是图片结果
    # 训练集有 60000个28*28的像素
    # 测试集有 10000个28*28的像素
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # 转换为浮点张量， 并缩放到-1~1
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.
    # y_train 转换为整形张量
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    # one-hot编码 数据集是1-10的手写数字故depth=10
    y = tf.one_hot(y_train, depth=10)
    # [b,28,28] => [b,28*28]
    x = tf.reshape(x_train, (-1, 28 * 28))
    # 构建数据集对象
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # 批量训练
    train_dataset = train_dataset.batch(200)
    return train_dataset


# 初始化参数
# 输入节点数为784，第一层输出节点数为256，第二层输出节点数为128，第三层输出节点数为10
def init_paramaters():
    # 每层的张量都要被定义成Variable 用正态分布初始化权值张量 用0初始化偏执向量
    # 第一层的参数
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
    # 第二层的参数
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))
    # 第三层的参数
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    return w1, b1, w2, b2, w3, b3


def train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # 第一层计算，[b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b,256] + [b, 256]
            h1 = x @ w1 + b1
            # 通过激活函数
            h1 = tf.nn.relu(h1)
            # 第二层计算
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # 第三层
            out = h2 @ w3 + b3

            # 计算误差，使用均方误差
            loss = tf.square(y - out)
            # 误差的标量
            loss = tf.reduce_mean(loss)

            # 梯度下降
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 梯度更新，assign_sub 将当前值减去参数值，原地更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())
    return loss.numpy()


def train(epochs):
    losses = []
    train_dataset = load_data()
    w1, b1, w2, b2, w3, b3 = init_paramaters()
    for epoch in range(epochs):
        loss = train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001)
        losses.append(loss)

    x = [i for i in range(0, epochs)]
    # 绘制曲线
    plt.plot(x, losses, color='blue', marker='s', label='训练')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('MNIST数据集的前向传播训练误差曲线.png')
    plt.close()


if __name__ == '__main__':
    train(epochs=20)
