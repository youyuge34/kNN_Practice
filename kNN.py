#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import operator
from os import listdir


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


def file2matrix(filename):
    """
    文件数据转换为矩阵格式
    :param filename: 文件名
    :return: 返回数据矩阵，类别标签list
    """
    fr = open(filename)
    mLines = fr.readlines()
    mCount = len(mLines)
    mat = np.zeros((mCount, 3))  # 建立一个零矩阵
    mLabelList = []
    index = 0

    # 循环解析文件
    for line in mLines:
        line = line.strip().split('\t')  # 分割行
        mat[index, :] = line[0:3]
        index += 1
        mLabelList.append(int(line[-1]))  # 要强转换为int，否则变为str
    return mat, mLabelList


def autoNum(dataSet):
    """
    归一化特征值
    :param dataSet:数据集
    :return:归一化后的二维array，每列的最大差值，每列最小值
    """
    minValue = dataSet.min(axis=0)  # 对每一列进行min
    maxValue = dataSet.max(axis=0)
    mRange = maxValue - minValue
    m = dataSet.shape[0]  # 数据行数
    normDataSet = dataSet - np.tile(minValue, (m, 1))
    normDataSet = normDataSet / np.tile(mRange, (m, 1))  # numpy中矩阵除法要用solve，这里就是普通的除法
    return normDataSet, mRange, minValue


def img2vector(filename):
    """
    将该文件内的所有数字32x32个，转化为1x1024的向量
    :param filename: 文件名
    :return: 转化为1x1024的array
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(line[j])
    return returnVect


def handWritingTest():
    """
    识别手写数字，数据在文件夹内
    :return:
    """
    labels = []
    trainDirs = listdir('trainingDigits')
    trainCount = len(trainDirs)
    trainMat = np.zeros((trainCount, 1024))
    for i in range(trainCount):
        filename = trainDirs[i]
        fileLabel = filename.strip().split('_')[0]
        labels.append(int(fileLabel))  # 从文件名中取出对应的分类类别，存入labels
        reVector = img2vector('trainingDigits/%s' % filename)
        trainMat[int(i), :] = reVector[0, :]

    testDirs = listdir('testDigits')
    mError = 0
    testCount = len(testDirs)
    for i in range(testCount):
        filename = testDirs[i]
        testLabel = int(filename.strip().split('_')[0])
        testVector = img2vector('testDigits/%s' % filename)
        testAns = int(classify0(testVector, trainMat, labels, 5))
        print 'the %s number is %s and the test comes out %d' % ((i), (testLabel), testAns)
        if testAns != testLabel:
            mError += 1
    print 'Error cases number is', mError
    print 'Error rate is', mError * 1.0 / len(testDirs)
