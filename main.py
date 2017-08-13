#!/usr/bin/python
# -*- coding: UTF-8 -*-

import kNN


def test():
    group, labels = kNN.createDataSet()
    print kNN.classify0([0, 0], group, labels, 3)


if __name__ == '__main__':
    test()
