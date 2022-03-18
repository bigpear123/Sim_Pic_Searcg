#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/18
# @Author  : haoli1
# resnet + hnswlib 相似图片
import json
import os
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils
import joblib
import hnswlib
import numpy as np

def resnet_emb(path_img):
    model_name = 'resnet18' # could be fbresnet152 or inceptionresnetv2
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.eval()
    load_img = utils.LoadImage()
    # transformations depending on the model
    # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
    tf_img = utils.TransformImage(model)
    input_img = load_img(path_img)
    input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
    input = torch.autograd.Variable(input_tensor,requires_grad=False)

    output_logits = model(input) # 1x1000

    return output_logits.detach().numpy()[0]

def store_as_bin(path,embedding_bin_path,key_bin_path):
    if path !='':
        all_pic= [ os.path.join(path,i) for i in os.listdir(path)]
        all_emb= [list(resnet_emb(j)) for j in all_pic]
        print("all_emb:",len(all_emb[0]))
        joblib.dump(all_emb, embedding_bin_path)
        joblib.dump(all_pic, key_bin_path)
        index = hnswlib.Index(space="cosine", dim=1000)
        #M: 表示在构建期间，每个元素创建的双向链表的数量。M合理的范围是2-100。M值较高的时候在高召回率数据集上效果好，M值较低在低召回率数据集上效果好。M值决定了算法内存消耗
        # ef_construction:控制了索引时间和索引准确度，和ef参数具有相同的意义。
        # ef_constraction越大，构建时间越长，但是索引质量更好。在某种程度上提高ef_construction并不能提高index的质量
        index.init_index(max_elements=len(all_emb), ef_construction=400, M=8)
        #index.add_items(all_emb)
        index.add_items(all_emb,list(range(len(all_emb))))
        index.save_index(index_path)
        return True
    return False


def get_topn_id(embedding1,index_path):
    try:
        #embedding1=data[1] embedding格式：[[0.12323....],[0.2333,....]]
        ###
        index = hnswlib.Index(space="cosine", dim=768)
        index.load_index(index_path)
        index.set_ef(200)
        labels,distance=index.knn_query(embedding1, k=50)
        return labels, distance
    except Exception as e:
        print (e)
        return [], []

def predict(index_path,emb_index,pic_embedding):
    try:

        index = hnswlib.Index(space="cosine", dim=1000)
        print("index:",index_path)
        index.load_index(index_path)
        index.set_ef(4)
        # pic_embedding=resnet_emb(image_path)
        labels, distances = index.knn_query(pic_embedding, k=4)
        return labels, distances

    except Exception as e:
        print (e)
        return [], []

if __name__=='__main__':
    import time
    #相关参数
    filename='./non-handraw/'
    candicate_pic='./candidate'
    embedding_bin_path='./pic_embedding.bin'
    key_bin_path='./pic_key.bin'
    index_path='./kmeans_index.index'
    #1、存储原始 字典的图片的embedding
    print("############### 开始处理字典图片 ##################")
    store_as_bin(filename,embedding_bin_path,key_bin_path)
    print("############### 字典图片处理完毕 #################")
    #2、候选图像数据集进行检索
    print("############### 处理候选图片 ####################")
    data=joblib.load(embedding_bin_path)
    key_index=joblib.load(key_bin_path)
    start_time=time.time()
    result=[]
    #候选集都存成 embedding
    print("############### 候选图片处理完毕 ##################")
    can_pic= [ os.path.join(candicate_pic,i) for i in os.listdir(candicate_pic)]
    can_emb= [list(resnet_emb(j)) for j in can_pic]
    print("############# 开始检索 #################")
    #3、进行检索
    all_can_label=[]
    all_can_score=[]
    labels, distance =predict(index_path,embedding_bin_path,can_emb[0:len(can_emb)-1])

    for i in range(len(distance)):
        if distance[i][0]< 0.2:
            all_can_label.append(labels[i][0])
            all_can_score.append(distance[i][0])

    # print(all_can_label, all_can_score)
    # fianl_dict=zip(all_can_label,all_can_score)
    # print("fianl_dict:",fianl_dict)

    ##labels和distance格式：[[],[],[]]
    with open("result.txt","a+") as f:
        for i, result_content in enumerate(labels):
            key_id=can_pic[i]
            predictions_out = []
            for j, prediction in enumerate(result_content):
                huati_results = key_index[int(prediction)]
                if distance[i][j] < 0.2  :
                    predictions_out.append(huati_results + "@" + str(distance[i][j])[:5])
            f.write('%s\t%s' % (key_id, predictions_out))
            f.write("\n")
    end_time=time.time()
