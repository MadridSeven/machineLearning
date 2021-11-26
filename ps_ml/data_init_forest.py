import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
import time

parent_path = 'D:/dataInit/'
file_path = 'price_avg_marke.csv'


def analysis():
    ax = plt.axes(projection='3d')
    zdata1 = []
    xdata1 = []
    ydata1 = []
    zdata2 = []
    xdata2 = []
    ydata2 = []
    X_train = []
    for data in pd.read_csv(f'{parent_path}{file_path}', chunksize=5000, encoding='ISO-8859-1'):
        for index, row in data.iterrows():
            hand_price = row['mid_avg_price']
            price_effective_time = row['price_effective_time']
            vip_price = row['mid_avg_vip_price']
            mid = row['goods_id']
            price_effective_time = time.strptime(price_effective_time, "%Y/%m/%d %H:%M")
            X_train.append([time.mktime(price_effective_time), hand_price/vip_price, vip_price, mid])
    clf = IsolationForest(max_samples='auto', contamination='auto')
    X_train = pd.DataFrame(X_train, columns=['x1', 'x2', 'x3', 'x4'])
    clf.fit(X_train)
    outlier_label = clf.predict(X_train)
    # 将array 类型的标签数据转成 DataFrame
    outlier_pd = pd.DataFrame(outlier_label, columns=['outlier_label'])
    # 将异常和数据集合并
    data_merge = pd.concat((X_train, outlier_pd), axis=1)
    for index, row in data_merge.iterrows():
        if row["outlier_label"] == -1:
            zdata2.append(row["x1"])
            xdata2.append(row["x2"])
            ydata2.append(row["x3"])
            print(row["x4"])
        else:
            zdata1.append(row["x1"])
            xdata1.append(row["x2"])
            ydata1.append(row["x3"])
    print(data_merge)
    print(outlier_label)
    ax.scatter(xdata2, ydata2, zdata2, marker='o', c='r')
    ax.scatter(xdata1, ydata1, zdata1, marker='o', c='b')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')




def main():
    plt.title("demo")
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.xticks(range(0, 365, 5))
    plt.xticks(rotation=30),
    analysis()

    plt.show()


if __name__ == '__main__':
    main()
