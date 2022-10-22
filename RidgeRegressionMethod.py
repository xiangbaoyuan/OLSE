import numpy as np


def VarianceExpansionFactor(X, y):
    '''
        方差扩大因子法计算岭参数
        参数：
        X    自变量矩阵
        y    因变量
        返回值：
        k    岭参数
    '''

    # 通过方差扩大因子法选取岭参数
    for i in np.arange(0, 3, 0.05):
        # 计算各方差扩大因子
        temp = X.T * X + i * np.identity(X.shape[1])
        c = temp.I * X.T * X * temp.I
        Cjj = np.diagonal(c)  # diagonal取矩阵对角线元素

        # 判断是否所有方差扩大因子都小于等于10
        # if sum(Cjj <= 10) == len(Cjj):
        if (Cjj <= 10).all():
            k = i
            break

    return k




def RidgeRegressionMethod(X, y, k):
    '''
        岭回归实现
        参数：
        参数：
        X    自变量矩阵
        y    因变量
        k    岭参数
        返回值：
        Beta_hat     系数估计值
        y_hat        y估计值
        error_hat    残差
    '''

    mat_temp = X.T * X + k * np.identity(X.shape[1])  # np.identity创建一个I矩阵
    Beta_hat = mat_temp.I * X.T * y

    y_hat = X * Beta_hat
    error_hat = y - y_hat

    return Beta_hat, y_hat, error_hat