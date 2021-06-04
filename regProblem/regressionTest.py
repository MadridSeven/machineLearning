# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def ready():
    # 𝑦 = 𝑤𝑥 + 𝑏 + 𝜖
    # 数据采样：准备数据集
    # 为了能够很好地模拟真实样本的观测误差，我们给模型添加误差自变量𝜖，它采样自均值为 0，方差为 2 的正态分布分布
    # y=1.477*x+0.089+N(0，0.1^2)

    # 保存样本集的列表
    data = []
    # 循环采样100个点
    for i in range(100):
        # 随机采样输入x
        # numpy.random.uniform(low,high,size):从一个均匀分布[low,high)中随机采样
        #   low: 采样下界，float类型，默认值为0；
        #   high: 采样上界，float类型，默认值为1；
        #   size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m*n*k个样本，缺省时输出1个值。
        x = np.random.uniform(-10, 10)
        # 从正态分布中采样误差值
        # numpy.random.normal(loc,scale,size):从一个均值为loc，方差为scale的正态分布中随机取样，size:输出的shape，默认为None，只输出一个值
        eps = np.random.normal(0, 2)
        # 得到输出
        y = 1.477 * x + 0.089 + eps
        # 保存样本
        data.append([x, y])
        # 画点
        plt.plot(x, y, 'bo-')
    data = np.array(data)
    return data


# 𝑦 = 𝑤𝑥 + 𝑏
# 计算损失函数(均方误差)：预测值与真实值之间差的平方和
def mse(b, w, points):
    # 根据当前的w,b参数计算均方误差
    totalError = 0
    # 循环迭代所有的点
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 计算差的平方，累加
        totalError += (y - (w * x + b)) ** 2
    # 将累加的误差求平均，得到均方误差
    return totalError / float(len(points))


# 计算梯度(对损失函数的梯度)
def step_gradient(b_current, w_current, points, lr):
    # b的梯度
    b_gradient = 0
    # w的梯度
    w_gradient = 0
    # 总样本数
    N = float(len(points))
    # 计算b和w的梯度
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        w_gradient += (2 / N) * ((w_current * x + b_current) - y) * x
    # 求梯度下降后的b和w
    new_b = b_current - (b_gradient * lr)
    new_w = w_current - (w_gradient * lr)
    return [new_b, new_w]


# 从初始值开始更新梯度num次
def gradient_descent(points, start_b, start_w, lr):
    # b的初始值
    b = start_b
    # w的初始值
    w = start_w
    while True:
        old_loss = mse(b, w, points)
        # 计算梯度并更新一次
        b, w = step_gradient(b, w, np.array(points), lr)
        # 计算误差
        loss = mse(b, w, points)
        # 打印误差
        print(f"loss为{loss}，w为{w},b为{b}")
        # 这次的误差和上次的误差相等
        if abs(loss - old_loss) < 1e-15:
            break
    # 返回最后一次的b和w
    return [b, w]


def main():
    plt.title("demo")
    plt.xlabel("x")
    plt.ylabel("y")
    data = ready()
    start_b = 0
    start_w = 0
    lr = 0.01
    [b, w] = gradient_descent(data, start_b, start_w, lr)
    print(f"最终得出w为{w},b为{b}")
    x = np.arange(-10, 10, 0.1)
    y = w * x + b
    plt.plot(x, y)
    # 画出数据集的坐标系
    plt.show()


if __name__ == '__main__':
    main()
