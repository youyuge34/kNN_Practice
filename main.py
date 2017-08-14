#!/usr/bin/python
# -*- coding: UTF-8 -*-

import kNN
import matplotlib.pyplot as plt


def test():
    """
    测试KNN算法
    :return:
    """
    group, labels = kNN.createDataSet()
    print kNN.classify0([0, 0], group, labels, 3)


def showDatingTestData():
    """
    测试约会案例，文件数据转换成矩阵数据,使用10%数据作为测试集
    :return:
    """
    mRatio = 0.1
    mat, labels = kNN.file2matrix('datingTestSet2.txt')
    normMat, mRange, mMin = kNN.autoNum(mat)  # 数据归一化
    mCount = mat.shape[0]  # 数据行数
    mTestCount = int(mRatio * mCount);  # 测试集数目
    mError = 0  # 错误数
    for i in range(mTestCount):
        mResult = kNN.classify0(normMat[i, :], normMat[mTestCount:mCount, :], labels[mTestCount:mCount], 5)
        if (mResult != labels[i]):
            mError += 1
    print 'The error rate is: %f' % (mError * 1.0 / mTestCount)
    print 'The total test count is %d and the error count is %d' % (mTestCount, mError)


def showDatingFigure(mat, labels):
    """
    显示无类别标签的散点图
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 画布分为1x1，画在第1个上
    ax.scatter(mat[:, 0], mat[:, 1], s=5.0 * kNN.np.array(labels), c=kNN.np.array(labels))
    # plt.axis([-0.5, 25, -0.2, 2])
    plt.show()


def showDatingInput():
    # 输入测试数据
    resultList = ['not at all', 'in small doses', 'in large doses']
    mPercentGame = float(raw_input('the percentange of time spent playing vedio games:'))
    mPercentMiles = float(raw_input('the miles earned every year:'))
    mpercentIce = float(raw_input('the ice cream consumed per year:'))
    testArray = [mPercentMiles, mPercentGame, mpercentIce]

    mat, labels = kNN.file2matrix('datingTestSet2.txt')
    normMat, mRange, mMin = kNN.autoNum(mat)
    ansType = kNN.classify0((testArray - mMin) / mRange, normMat, labels, 5)
    print 'This guy is mostly', resultList[int(ansType) - 1]


if __name__ == '__main__':
    # test()
    # showDatingTestData()
    showDatingInput()
