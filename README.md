# Sim_Pic_Searcg
use resnet embedding to get similar picture
这个是简单的开发了一个相似图像检索的代码，网上搜集了一些基于传统算法开发的相似图像检索，但是对阈值要求比较高，感觉都不太符合自己的需要，所以简单开发了一个居于向量进行的图片相似性检索的代码。

主要是用深度网络来产生图片的表示向量，利用这个向量来进行相似检索, 代码直接使用的pytorch 开源的resnet预训练好的模型，使用bin 保存图片向量，使用hnswlib进行大规模向量检索，批量返回每张图片的相似图片
## 依赖

```
torch
hnswlib
pretrainedmodels
```

## 运行代码

```
python3 resnet_hnswlib_method.py
```
