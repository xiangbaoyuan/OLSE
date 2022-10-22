from OrdinaryLeastSquaresEstimation import *
from RestrictedLeastSquares import *
from RidgeRegressionMethod import *



def MeanSquareError(Beta, Beta_hat):
    '''
        MSE计算
        参数：
        Beta       参数真实值
        Beta_hat   参数估计值
        返回值：
        MSE_Beta   用Beta计算的MSE
    '''

    MSE_Beta = (Beta - Beta_hat).T * (Beta - Beta_hat) / len(Beta)

    return MSE_Beta

# def MeanSquareError(Beta, Beta_hat, y, y_hat):
#     '''
#         MSE计算
#         参数：
#         Beta       参数真实值
#         Beta_hat   参数估计值
#         y          因变量真实值
#         y_hat      因变量估计值
#         返回值：
#         MSE_Beta   用Beta计算的MSE
#         MSE_y      用y计算的MSE
#     '''
#
#     MSE_Beta = (Beta - Beta_hat).T * (Beta - Beta_hat) / len(Beta)
#     MSE_y = (y - y_hat).T * (y - y_hat) / len(y)
#
#     return MSE_Beta, MSE_y

def CompareTheMSE(Beta, X, y, X_res, y_res):
    # 计算
    # 最小二乘法实现参数估计
    Beta_hat1, y_hat, error_hat = LeastSquareMethod(X, y)
    # MSE_Beta1, MSE_y1 = MeanSquareError(Beta, Beta_hat1, y, y_hat)
    MSE1 = MeanSquareError(Beta, Beta_hat1)
    # 岭回归实现参数估计
    # 计算岭参数
    k = VarianceExpansionFactor(X, y)
    # 岭回归
    Beta_hat2, y_hat, error_hat = RidgeRegressionMethod(X, y, k)
    # MSE_Beta2, MSE_y2 = MeanSquareError(Beta, Beta_hat2, y, y_hat)
    MSE2 = MeanSquareError(Beta, Beta_hat2)
    # 带约束的最小二乘
    Beta_hat3, y_hat, error_hat = RestrictedLeastSquares(X, y, X_res, y_res)
    # MSE_Beta3, MSE_y3 = MeanSquareError(Beta, Beta_hat3, y, y_hat)
    MSE3 = MeanSquareError(Beta, Beta_hat3)

    '''
        绘制虚线表对几种参数估计方法的结果进行比较
        print('|{x:^{y}}|'.format(x='天气',y=15 - len('天气'.encode('GBK')) + len('天气')))
        x表示要输出的内容: 号后面带填充的字符，只能是一个字符，不指定则默认是用空格填充。
        ^表示居中对其{}进行转义，里面的y代表总占位长度
        .encode('GBK')是因为中文默认一个字符，GBK改为两个字符
    '''
    # 打印
    print('+{x:-^{y}}+'.format(x='', y=6 + len('MSE_Beta')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('OLSE')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('RE')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('RLSE')))
    print('|{x:^{y}}|'.format(x='', y=6 + len('MSE_Beta')) +
          '{x:^{y}}|'.format(x='OLSE', y=6 + len('OLSE')) +
          '{x:^{y}}|'.format(x='RE', y=6 + len('RE')) +
          '{x:^{y}}|'.format(x='RLSE', y=6 + len('RLSE')))
    print('+{x:-^{y}}+'.format(x='', y=6 + len('MSE_Beta')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('OLSE')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('RE')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('RLSE')))

    for i in range(len(Beta)):
        print('|{x:^{y}}|'.format(x='Beta_%d' % i, y=6 + len('MSE_Beta')) +
              '{x:^{y}}|'.format(x='%.3f' % Beta_hat1[i], y=6 + len('OLSE')) +
              '{x:^{y}}|'.format(x='%.3f' % Beta_hat2[i], y=6 + len('RE')) +
              '{x:^{y}}|'.format(x='%.3f' % Beta_hat3[i], y=6 + len('RLSE')))
        # %(i+1, Beta_hat1[i], Beta_hat2[i], Beta_hat3[i]))
        print('+{x:-^{y}}+'.format(x='', y=6 + len('MSE_Beta')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('OLSE')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('RE')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('RLSE')))
        # print('| Beta_%d  |  %.3f  | %.3f |     %.3f      |'
        #       %(i+1, Beta_hat1[i], Beta_hat2[i], Beta_hat3[i]))
        # print('+---------+---------+-------+----------------+')

    # print('|   MSE   |  %.3f  | %.3f |     %.3f      |'%(MSE1, MSE2, MSE3))
    # print('|{x:^{y}}|'.format(x='MSE_Beta', y=6 + len('MSE_Beta')) +
    #       '{x:^{y}}|'.format(x='%.3f' % MSE_Beta1, y=6 + len('OLSE')) +
    #       '{x:^{y}}|'.format(x='%.3f' % MSE_Beta2, y=6 + len('RE')) +
    #       '{x:^{y}}|'.format(x='%.3f' % MSE_Beta3, y=6 + len('RLSE')))
    print('|{x:^{y}}|'.format(x='MSE', y=6 + len('MSE_Beta')) +
          '{x:^{y}}|'.format(x='%.3f' % MSE1, y=6 + len('OLSE')) +
          '{x:^{y}}|'.format(x='%.3f' % MSE2, y=6 + len('RE')) +
          '{x:^{y}}|'.format(x='%.3f' % MSE3, y=6 + len('RLSE')))
    # %(MSE1, MSE2, MSE3))
    print('+{x:-^{y}}+'.format(x='', y=6 + len('MSE_Beta')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('OLSE')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('RE')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('RLSE')))

    # print('|{x:^{y}}|'.format(x='MSE_y', y=6 + len('MSE_Beta')) +
    #       '{x:^{y}}|'.format(x='%.3f' % MSE_y1, y=6 + len('OLSE')) +
    #       '{x:^{y}}|'.format(x='%.3f' % MSE_y2, y=6 + len('RE')) +
    #       '{x:^{y}}|'.format(x='%.3f' % MSE_y3, y=6 + len('RLSE')))
    # # %(MSE1, MSE2, MSE3))
    # print('+{x:-^{y}}+'.format(x='', y=6 + len('MSE_Beta')) +
    #       '{x:-^{y}}+'.format(x='', y=6 + len('OLSE')) +
    #       '{x:-^{y}}+'.format(x='', y=6 + len('RE')) +
    #       '{x:-^{y}}+'.format(x='', y=6 + len('RLSE')))

    return
