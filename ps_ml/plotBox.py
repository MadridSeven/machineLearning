import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parent_path = 'D:/dataInit/'
file_path = 'forest_price.csv'

def analysis():
    price_arr_diff = []
    for data in pd.read_csv(f'{parent_path}{file_path}', chunksize=5000, encoding='ISO-8859-1'):
        for index, row in data.iterrows():
            hand_price = row['mid_avg_price']
            vip_price = row['mid_avg_vip_price']
            if hand_price != vip_price:
                price_arr_diff.append(hand_price / vip_price)
    BoxFeature(price_arr_diff)
    price_arr_diff = {'price_diff': price_arr_diff}
    price_arr_diff_pd = pd.DataFrame(price_arr_diff)
    price_arr_diff_pd.plot.box(title="puma")
    print(price_arr_diff_pd.describe())
    plt.grid(linestyle="--", alpha=0.3)
    plt.show()

def BoxFeature(input_list):
    # 获取箱体图特征
    percentile = np.percentile(input_list, (25, 50, 75), interpolation='linear')
    # 以下为箱线图的五个特征值
    Q1 = percentile[0]  # 上四分位数
    Q2 = percentile[1]
    Q3 = percentile[2]  # 下四分位数
    IQR = Q3 - Q1  # 四分位距
    ulim = Q3 + 1.5 * IQR  # 上限 非异常范围内的最大值
    llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值
    # 统计异常点个数
    # 正常数据列表
    right_list = []
    Error_Point_num = 0
    value_total = 0
    average_num = 0
    for item in input_list:
        if item < llim or item > ulim:
            Error_Point_num += 1
        else:
            right_list.append(item)
            value_total += item
            average_num += 1
    average_value = value_total / average_num
    # 特征值保留一位小数
    out_list = [average_value, min(right_list), Q1, Q2, Q3, max(right_list)]
    # print(out_list)
    print(out_list)
    print(Error_Point_num)


def main():
    analysis()

if __name__ == '__main__':
    main()