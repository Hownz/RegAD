import torch
from torch.functional import norm
import torch.nn.functional as F
import torch.nn as nn
from math import exp
from functools import partial


def L2Loss(data1, data2):
    # data shape: BCHW
    
    norm_data = torch.norm(data1-data2, p=2, dim=1)
    # norm_data shape: BHW

    loss = norm_data.mean()
    # print("loss: ", loss)
    return loss


def CosLoss(data1, data2, Mean=True):
    data2 = data2.detach()
    cos = nn.CosineSimilarity(dim=1) # 方法是计算两个输入张量之间的余弦相似度 值越接近1表示相似度越高，值越接近-1表示相似度越低。
    if Mean:
        return 1-cos(data1, data2).mean()
    else:
        return 1-cos(data1, data2)



