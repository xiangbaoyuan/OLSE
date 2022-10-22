# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:27:32 2022

@author: 23618
"""

# from numpy import * # 导入numpy的库函数(可直接使用mat方法进行各种运算)
import numpy as np
import scipy.stats as sta
import matplotlib.pyplot as plt


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
def RandomSampleGeneration(Beta, N):
    np.random.seed(123)
    
    # 生成样本X
    X_ones = np.mat(np.ones(N)).T
    X_array = np.random.randn(N, len(Beta)-1)  # 生成一个指定大小的标准正态分布多维数组
    X_array = np.append(X_ones, X_array, axis=1)
    X = np.mat(X_array)  # 将数组转换为矩阵
    
    # 生成随机扰动项error
    error_array = np.random.randn(N,1)
    error = np.mat(error_array)

    # 生成指定表达式的y值
    y = X * Beta + error
    
    return X, y, error



def LeastSquareMethod(X, y):
    '''
        最小二乘法实现
        参数：
        X    自变量矩阵
        y    因变量
        返回值：
        Beta_hat     系数估计值
        y_hat        y估计值
        error_hat    残差
    '''

    mat_temp = X.T * X # .T表示转置
    Beta_hat = mat_temp.I * X.T * y # .I表示矩阵求逆
    
    y_hat = X * Beta_hat
    error_hat = y - y_hat
    
    return Beta_hat, y_hat, error_hat


'''
    方差扩大因子法计算岭参数
    参数：
    X    自变量矩阵
    y    因变量
    返回值：
    k    岭参数
'''
def VarianceExpansionFactor(X, y):
    # 通过方差扩大因子法选取岭参数
    for i in np.arange(0,3,0.05):
        # 计算各方差扩大因子
        temp = X.T * X + i * np.identity(X.shape[1])
        c = temp.I * X.T * X * temp.I
        Cjj = np.diagonal(c) # diagonal取矩阵对角线元素
        
        # 判断是否所有方差扩大因子都小于等于10
        # if sum(Cjj <= 10) == len(Cjj):
        if (Cjj <= 10).all():
            k = i
            break
        
    return k


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
def RidgeRegressionMethod(X, y, k):
    mat_temp = X.T * X + k * np.identity(X.shape[1]) # np.identity创建一个I矩阵
    Beta_hat = mat_temp.I * X.T * y 
    
    y_hat = X * Beta_hat
    error_hat = y - y_hat
    
    return Beta_hat, y_hat, error_hat


'''
    带约束的最小二乘方法
    参数：
    X            自变量矩阵
    y            因变量
    X_res        约束系数矩阵
    y_res        约束因变量
    返回值：
    Beta_hat     系数估计值
    y_hat        y估计值
    error_hat    残差
'''
def RestrictedLeastSquares(X, y, X_res, y_res):
    Nbb = X.T * X 
    Ncc = X_res * Nbb.I * X_res.T
    temp1 = Nbb.I - Nbb.I*X_res.T*Ncc.I*X_res*Nbb.I
    temp2 = Nbb.I*X_res.T*Ncc.I
    Beta_hat = temp1*(X.T*y) + temp2*y_res
    
    y_hat = X * Beta_hat
    error_hat = y - y_hat
    
    return Beta_hat, y_hat, error_hat


'''
    绘制虚线表对几种参数估计方法的结果进行比较
    print('|{x:^{y}}|'.format(x='天气',y=15 - len('天气'.encode('GBK')) + len('天气')))
    x表示要输出的内容: 号后面带填充的字符，只能是一个字符，不指定则默认是用空格填充。
    ^表示居中对其{}进行转义，里面的y代表总占位长度
    .encode('GBK')是因为中文默认一个字符，GBK改为两个字符
'''
def CompareTheMSE(Beta, X, y, X_res, y_res):
    # 计算
    # 最小二乘法实现参数估计
    Beta_hat1, y_hat, error_hat = LeastSquareMethod(X, y)
    MSE1 = MeanSquareError(Beta, Beta_hat1)
    # 岭回归实现参数估计
    # 计算岭参数
    k = VarianceExpansionFactor(X, y)
    # 岭回归
    Beta_hat2, y_hat, error_hat = RidgeRegressionMethod(X, y, k)
    MSE2 = MeanSquareError(Beta, Beta_hat2)
    # 带约束的最小二乘
    Beta_hat3, y_hat, error_hat = RestrictedLeastSquares(X, y, X_res, y_res)
    MSE3 = MeanSquareError(Beta, Beta_hat3)
    
    # 打印
    print('+{x:-^{y}}+'.format(x='',y=6+len('MSE')) + 
          '{x:-^{y}}+'.format(x='',y=6+len('OLSE')) + 
          '{x:-^{y}}+'.format(x='',y=6+len('RE')) + 
          '{x:-^{y}}+'.format(x='',y=6+len('RLSE')))
    print('|{x:^{y}}|'.format(x='',y=6+len('MSE')) + 
          '{x:^{y}}|'.format(x='OLSE',y=6+len('OLSE')) + 
          '{x:^{y}}|'.format(x='RE',y=6+len('RE')) + 
          '{x:^{y}}|'.format(x='RLSE',y=6+len('RLSE')))
    print('+{x:-^{y}}+'.format(x='',y=6+len('MSE')) + 
          '{x:-^{y}}+'.format(x='',y=6+len('OLSE')) + 
          '{x:-^{y}}+'.format(x='',y=6+len('RE')) + 
          '{x:-^{y}}+'.format(x='',y=6+len('RLSE')))

    for i in range(len(Beta)):
        print('|{x:^{y}}|'.format(x='Beta_%d'%(i+1),y=6+len('MSE')) + 
              '{x:^{y}}|'.format(x='%.3f'%Beta_hat1[i],y=6+len('OLSE')) + 
              '{x:^{y}}|'.format(x='%.3f'%Beta_hat2[i],y=6+len('RE')) + 
              '{x:^{y}}|'.format(x='%.3f'%Beta_hat3[i],y=6+len('RLSE')))
              #%(i+1, Beta_hat1[i], Beta_hat2[i], Beta_hat3[i]))
        print('+{x:-^{y}}+'.format(x='',y=6+len('MSE')) + 
              '{x:-^{y}}+'.format(x='',y=6+len('OLSE')) + 
              '{x:-^{y}}+'.format(x='',y=6+len('RE')) + 
              '{x:-^{y}}+'.format(x='',y=6+len('RLSE')))
        # print('| Beta_%d  |  %.3f  | %.3f |     %.3f      |'
        #       %(i+1, Beta_hat1[i], Beta_hat2[i], Beta_hat3[i]))
        # print('+---------+---------+-------+----------------+')
        
    # print('|   MSE   |  %.3f  | %.3f |     %.3f      |'%(MSE1, MSE2, MSE3))  
    print('|{x:^{y}}|'.format(x='MSE',y=6+len('MSE')) + 
          '{x:^{y}}|'.format(x='%.3f'%MSE1,y=6+len('OLSE')) + 
          '{x:^{y}}|'.format(x='%.3f'%MSE2,y=6+len('RE')) + 
          '{x:^{y}}|'.format(x='%.3f'%MSE3,y=6+len('RLSE')))
          # %(MSE1, MSE2, MSE3))
    print('+{x:-^{y}}+'.format(x='',y=6+len('MSE')) + 
          '{x:-^{y}}+'.format(x='',y=6+len('OLSE')) + 
          '{x:-^{y}}+'.format(x='',y=6+len('RE')) + 
          '{x:-^{y}}+'.format(x='',y=6+len('RLSE')))
    
    return


'''
    拟合优度计算
    参数：
    y          真实的y
    error_hat  残差
    N          样本量
    K          参数个数
    返回值：
    R_square            拟合优度
    Adjusted_R_square   叫做后的拟合优度
'''
def GoodnessOfFit(y, error_hat, N, K):
    SSE = sum(np.multiply(error_hat,error_hat))
    SST = sum(np.multiply(y-y.mean(),y-y.mean()))
    
    R_square = 1 - SSE / SST
    Adjusted_R_square = 1 - (SSE/(N-K)) / (SST/(N-1))
    return R_square, Adjusted_R_square


'''
    MSE计算
    参数：
    Beta       参数真实值
    Beta_hat   参数估计值
    返回值：
    MSE        
'''
def MeanSquareError(Beta, Beta_hat):
    MSE = (Beta - Beta_hat).T * (Beta - Beta_hat) / len(Beta)
    
    return MSE


'''
    单个参数检验
    参数：
    X           自变量矩阵
    Beta_hat    系数估计值
    error_hat   残差
    Beta_test   用于原假设的Beta值，一般为0向量
    返回值：
    P           各参数检验的P值 
'''
def TTest(X, Beta_hat, error_hat, Beta_test):
    # 计算Beat_hat的方差
    mat_temp = np.diagonal((X.T * X).I)
    freedom = X.shape[0] - X.shape[1]
    Beta_var_hat = error_hat.T * error_hat / freedom * mat_temp
    
    # 通过t值计算检验的P值
    t = (Beta_hat - Beta_test) / np.sqrt(Beta_var_hat)
    P = 2 * sta.t.sf(t,freedom)
    
    return P


'''
    F检验
    参数：
    y          真实的y
    error_hat  残差
    N          样本量
    K          参数个数
    返回值：
    P           各参数检验的P值 
'''
def FTest(y, error_hat, N, K):
    SSE = sum(np.multiply(error_hat,error_hat))
    SST = sum(np.multiply(y-y.mean(),y-y.mean()))
    SSR = SST - SSE
    
    # 计算自由度
    freedom_SSR = K - 1
    freedom_SSE = N - K
    
    # 通过F值计算检验的P值
    F = (SSR/freedom_SSR) / (SSE/freedom_SSE)
    P = sta.f.sf(F,freedom_SSR,freedom_SSE)
    
    return P


'''
    主函数main
    用于调用各函数模块执行
'''
def main():
    # 生成随机模拟变量
    N = 100
    Beta = np.mat([3,6,7]).T # 列向量
    X, y, error = RandomSampleGeneration(Beta, N)
    
    # 最小二乘法实现参数估计
    Beta_hat, y_hat, error_hat = LeastSquareMethod(X, y)
    MSE = MeanSquareError(Beta, Beta_hat)
    # 计算拟合优度
    R_square, Adjusted_R_square = GoodnessOfFit(y, error_hat, N, len(Beta))
    
    # 岭回归实现参数估计
    # 计算岭参数
    k = VarianceExpansionFactor(X, y)
    # 岭回归
    Beta_hat, y_hat, error_hat = RidgeRegressionMethod(X, y, k)
    MSE = MeanSquareError(Beta, Beta_hat)
    
    # 带约束的最小二乘
    X_res = np.mat([1,2,3])
    y_res = np.mat([4])
    #Beta_hat, y_hat, error_hat = RestrictedLeastSquares(X, y, X_res, y_res)
    
if __name__ == "__main__":
    
    main()


# X = np.mat([[1,1,1,1,1,1,1,1,1,1],
#           [1.1,1.4,1.7,1.7,1.8,1.8,1.9,2.0,2.3,2.4],
#           [1.1,1.5,1.8,1.7,1.9,1.8,1.8,2.1,2.4,2.5]]).T
# error = np.mat([[0.8,-0.5,0.4,-0.5,0.2,1.9,1.9,0.6,-1.5,-0.3]]).T
# Beta = np.mat([10,2,3]).T
# y = X*Beta+error
# # 带约束的最小二乘
# X_res = np.mat([1,2,3])
# y_res = np.mat([4])
# CompareTheMSE(Beta, X, y, X_res, y_res)
# k = VarianceExpansionFactor(X, y)
# # 带约束的最小二乘
# X_res = np.mat([1,2,3])
# y_res = np.mat([4])
# Beta_hat, y_hat, error_hat = RestrictedLeastSquares(X, y, X_res, y_res)







