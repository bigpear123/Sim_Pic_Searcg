#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/18
# @Author  : ally
# 传统方法收集数据集的使用，获得与目标图片相似的图片
# 输入mid串，以及相关需要获取的图像例子文件夹，会对比出这些文件中
# 原理：将输入的图片转化为32*32的黑白缩略图，通过和平均值对比得到哈希值，最后再比较哈希值的差距
# --auc 参数是比对的差异程度，指定识别的精确度，0为最高，随着数字变大，精确度降低
# 测试 auc值为5的时候，两张图片差不多一样了
# 效果并不好：）
import os
import numpy as np
import cv2
import argparse

def main():
    remove_simillar_picture_by_perception_hash()

def process_arguments():
    parser = argparse.ArgumentParser(description = '图片去重')
    
    parser.add_argument('--path',action='store',default='image',
                       help="请选择图片所在位置，默认为当前目录下image文件夹")
    
    parser.add_argument('--delete',action = 'store_true',
                       help="带上delete参数即为直接删除，建议先大概查看")

    parser.add_argument('--acc',action = 'store',default=5,
                       help="指定检测的敏感度，默认为5（一般都够用）")
    
    return parser.parse_args()

def cal_similar():
    return 0 

def remove_simillar_picture_by_perception_hash():
    # 获取图片的位置
    path = process_arguments().path
    img_list = os.listdir(path)
    hash_dic = {}
    hash_list = []
    hash_name = []
    count_num = 0
    # 这里加载图片
    for img_name in img_list:
        # 将图片转化为黑白
        try:
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            count_num += 1
        except:
            continue
        # 将图片变成32*32的缩略图并且与平均值比较求和，得到哈希值
        img = cv2.resize(img,(32,32))
        avg_np = np.mean(img)
        img = np.where(img>avg_np,1,0)
        hash_dic[img_name] = img
        if len(hash_list)<1:
            hash_list.append(img)
            hash_name.append(img_name)
        else:
            for index,i in enumerate(hash_list):
                flag = True
                # 获取两张图片的哈希值差距矩阵，异或运算
                dis = np.bitwise_xor(i,img)
                # 当哈希值差距 auc 位不同的时候就是不同，auc 越大 图像
                if np.sum(dis) < int(process_arguments().accu):
                    flag = False
                    if process_arguments().delete == True:
                        os.remove(os.path.join(path, img_name))
                    print(img_name + " is similar to " + hash_name[index])
                    break
            if flag:
                hash_list.append(img)
                hash_name.append(img_name)

if __name__ == '__main__':
    main()
