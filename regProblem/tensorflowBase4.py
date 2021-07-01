#袖型数据初始化
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import requests
import cv2
import pandas as pd

parent_path = 'D:/dataInit/'
headers = {'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}

def download_img(img_url, dir_path, spu):
	file_name = f'{dir_path}{spu}_1.jpg'
	response = requests.get(url=img_url,headers=headers)
	if response.status_code != 200:
		print(img_url)
		return
	img_file = open(file_name, 'wb')
	img_file.write(response.content)
	img_file.close()
	pic = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
	pic_2 = cv2.resize(pic,(224,224))
	cv2.imwrite(file_name, pic_2)

def get_img(file_path):
    i = 1
    for data in pd.read_csv(f'{parent_path}{file_path}',chunksize=5000, encoding='ISO-8859-1'):
        for index,row in data.iterrows():
            if i < 0:
                i+=1
                continue
            print(i)
            img = row['image_1']
            spu = row['spu_id']
            type = row['options']
            print(img)
            if pd.isnull(img) or img == '(NULL)':
                continue
            dir_path = f'{parent_path}'+type+'/'
            try:
                download_img(f'https://a.vpimg2.com{img}', dir_path, spu)
            except BaseException:
                try:
                    download_img(f'https://a.vpimg2.com{img}', dir_path, spu)
                except BaseException:
                    download_img(f'https://a.vpimg2.com{img}', dir_path, spu)
            i=i+1

get_img('qiuxing.csv')