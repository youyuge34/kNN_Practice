#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import operator


def createDataSet():
    group = np.array([[1, 1.1], [1, 1], [0, 0.1], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
        :parameter
        inX:输入测试向量
        dataSet:训练集
        labels:标签向量
        k:选择最邻近点的数目
    """
    # 1、计算输入向量与其他的距离
    size = dataSet.shape[0]
    diffMat = np.tile(inX, (size, 1)) - dataSet  # 距离差值矩阵
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5  # 算出了所有相差的距离矩阵
    sortedDistances = distances.argsort()

    # 2、选择距离最小的k个点
    countDict = {}
    for i in range(k):
        label = labels[sortedDistances[i]]  # 取出对应点的目标标签
        countDict[label] = countDict.get(label, 0) + 1  # 是某个目标标签的个数+1

    # 3、排序，字典中最大的标签数则为测试向量的类别
    ans = sorted(countDict.items(), key=lambda x: x[1], reverse=True)
    return ans[0][0]  # 返回类别
