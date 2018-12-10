# -*- encoding: utf-8 -*-
# Author: MingCrash

import numpy as np
from Sorting import QuickSort
# from matplotlib import pyplot

testdata = [172,42]

def createDataSet():
    group = [[179,42],
             [178,43],
             [165,36],
             [177,42],
             [160,35],
             [168,41]]
    label = ['M','M','F','M','F','M']
    return np.mat(group),label


#数据归一化 (极值归一法)
#缺陷就是当有新数据加入时，可能导致max和min的变化，需要重新定义。
def MinMaxNormalization(gp):
    gp0 = (gp[:,0]-np.min(gp[:,0]))/(np.max(gp[:,0])-np.min(gp[:,0]))
    gp1 = (gp[:,1]-np.min(gp[:,1]))/(np.max(gp[:,1])-np.min(gp[:,1]))
    return np.hstack([gp0,gp1]) #水平矩阵拼接

#x* = (x - μ ) / σ
def Z_ScoreNormalization(gp):
    µ0 = np.mean(gp[:,0])  #µ是均值
    µ1 = np.mean(gp[:,1])
    ç0 = np.std(gp[:,0])   #ç是标准差
    ç1 = np.std(gp[:,1])
    gp0 = (gp[:,0]-µ0)/ç0
    gp1 = (gp[:,1]-µ1)/ç1
    return np.hstack([gp0,gp1])

#欧氏距离
def dist (vec1,vec2):
    return np.sqrt(np.sum(np.square(vec1-vec2),axis=1))

def knn_classify(inx,dataset,label,k):
    dataset_size = dataset.shape[0]
    # dataset = np.insert(arr=dataset,obj=0,values=inx,axis=0)
    dataset = np.append(arr=dataset,values=inx,axis=0)
    nrmgp_tmp = MinMaxNormalization(dataset)
    nrmInx = np.tile(nrmgp_tmp[-1],(dataset_size,1))
    nrmgp = np.delete(arr=nrmgp_tmp,obj=-1,axis=0)
    distSet = dist(nrmInx,nrmgp)
    return distSet
    # QuickSort().startQuickSort(rsu)

# plt.scatter(nrmgp)
# plt.show()

gp,lb = createDataSet()
print(knn_classify(inx=[[170,40]],dataset=gp,label=lb,k=3))

# aa = np.array([[1,2,3],[4,5,6],[7,8,9]])
# bb = np.array([[1,2,3],[4,5,6],[7,8,9]])
# dd = np.square(aa,bb)
# ff = np.sum(dd,axis=1)
# gg = np.sqrt(ff)
# print(dist(aa,bb))