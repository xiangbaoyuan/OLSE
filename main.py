# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from OrdinaryLeastSquaresEstimation import *
from RestrictedLeastSquares import *
from RidgeRegressionMethod import *
from CompareTheMSE import *
import numpy as np
import scipy.stats as sta
import matplotlib.pyplot as plt




def RandomSampleGeneration(Beta, N):
    '''
        随机样本生成
        y = X Beta + error
        参数：
        Beta     系数
        N        样本量
        返回值：
        X        数据矩阵
        y        真实y
        error    随机扰动项
    '''

    np.random.seed(123)

    # 生成样本X
    X_ones = np.mat(np.ones(N)).T
    X_array = np.random.randn(N, len(Beta)-1)  # 生成一个指定大小的标准正态分布多维数组
    X_array = np.append(X_ones, X_array, axis=1)
    X = np.mat(X_array)  # 将数组转换为矩阵

    # 生成随机扰动项error
    error_array = np.random.randn(N, 1)
    error = np.mat(error_array)

    # 生成指定表达式的y值
    y = X * Beta + error

    return X, y, error



# Press the green button in the gutter to run the script.
'''
    主函数main
    用于调用各函数模块执行
'''


def main():
    # 生成随机模拟变量
    N = 100
    Beta = np.mat([3, 6, 7]).T  # 列向量
    X, y, error = RandomSampleGeneration(Beta, N)

    # 约束
    X_res = np.mat([1, 2, 3])
    y_res = np.mat([4])

    CompareTheMSE(Beta, X, y, X_res, y_res)
    TheResultsShow(X, y)

if __name__ == "__main__":
    # 具有多重共线性的模拟数据
    X = np.mat([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1.1, 1.4, 1.7, 1.7, 1.8, 1.8, 1.9, 2.0, 2.3, 2.4],
                [1.1, 1.5, 1.8, 1.7, 1.9, 1.8, 1.8, 2.1, 2.4, 2.5]]).T
    error = np.mat([[0.8, -0.5, 0.4, -0.5, 0.2, 1.9, 1.9, 0.6, -1.5, -0.3]]).T
    Beta = np.mat([10, 2, 3]).T
    y = X * Beta + error
    # 带约束的最小二乘
    X_res = np.mat([1, 2, 3])
    y_res = np.mat([4])
    CompareTheMSE(Beta, X, y, X_res, y_res)
    TheResultsShow(X, y)

    # 随机模拟数据实现
    main()




