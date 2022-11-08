import numpy as np
import pandas as pd
import scipy.stats as sta
import matplotlib.pyplot as plt


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

    mat_temp = X.T * X  # .T表示转置
    Beta_hat = mat_temp.I * X.T * y  # .I表示矩阵求逆

    y_hat = X * Beta_hat
    error_hat = y - y_hat

    return Beta_hat, y_hat, error_hat


def GoodnessOfFit(y, error_hat, N, K):
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

    SSE = sum(np.multiply(error_hat, error_hat))
    SST = sum(np.multiply(y - y.mean(), y - y.mean()))

    R_square = 1 - SSE / SST
    Adjusted_R_square = 1 - (SSE / (N - K)) / (SST / (N - 1))

    return R_square, Adjusted_R_square


def TTest(X, Beta_hat, error_hat, Beta_test):
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

    # 计算Beat_hat的方差
    mat_temp = np.diagonal((X.T * X).I)
    freedom = X.shape[0] - X.shape[1]
    Beta_var_hat = ((error_hat.T * error_hat / freedom) * mat_temp).T

    # 通过t值计算检验的P值
    t = (Beta_hat - Beta_test) / np.sqrt(Beta_var_hat)
    P = 2 * sta.t.sf(t, freedom)

    return Beta_var_hat, P


def FTest(y, error_hat, N, K):
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

    SSE = sum(np.multiply(error_hat, error_hat))
    SST = sum(np.multiply(y - y.mean(), y - y.mean()))
    SSR = SST - SSE

    # 计算自由度
    freedom_SSR = K - 1
    freedom_SSE = N - K

    # 通过F值计算检验的P值
    F = (SSR / freedom_SSR) / (SSE / freedom_SSE)
    P = sta.f.sf(F, freedom_SSR, freedom_SSE)

    return P


def ErrorRatio(y, y_hat):
    '''
        输入：
        y     真实值
        y_hat 估计值
        输出：
        ErrorRatio 误差百分比
    '''
    temp = abs(y - y_hat) / y * 100

    return temp.sum() / len(y)


def RegressionDiagnosis(X, y_hat, error_hat):
    '''
        回归诊断
        参数：
        X           自变量矩阵
        error_hat   残差
        返回值：
    '''

    # 绘制qq图
    sta.probplot(np.array(error_hat).reshape(1,len(error_hat))[0], dist="norm", plot=plt)
    plt.show()

    mu = error_hat.mean()
    sigma = error_hat.std()
    # 绘制残差直方图观察残差是否服从正态分布
    if len(error_hat) > 100:
        num_bins = 30  # 直方图柱子的数量
    else:
        num_bins = 10
    n, bins, patches = plt.hist(error_hat, num_bins, density=1, facecolor='blue', alpha=0.5)
    # n, bins, patches = plt.hist(error_hat, density=1)
    y = sta.norm.pdf(bins, mu, sigma)  # 拟合一条最佳正态分布曲线y
    plt.plot(bins, y, 'r--')  # 绘制y的曲线
    plt.xlabel('sepal-length')  # 绘制x轴
    plt.ylabel('Probability')  # 绘制y轴
    plt.title(r'Histogram : $\mu=%.3f$,$\sigma=%.3f$' % (mu, sigma))  # 中文标题 u'xxx'
    plt.show()
    # 绘制残差与y_hat的散点图
    plt.scatter(np.array(y_hat), np.array(error_hat))
    plt.axhline(0, c="red")
    plt.show()

    # 计算标准化残差
    freedom = X.shape[0] - X.shape[1]
    theta = np.sqrt(error_hat.T * error_hat / freedom)
    Std_error = error_hat / theta

    # 计算学生化残差
    mat_temp = X.T * X
    Hii = np.diagonal(X * mat_temp.I * X.T)
    Stu_error = error_hat / (theta * np.sqrt(1-Hii)).T

    # 计算DFFITS
    MSE = error_hat.T * error_hat / freedom
    DFFITSi = error_hat / np.sqrt(MSE * Hii).T

    # 计算Cook distances
    k = X.shape[1]
    temp1 = np.multiply(error_hat, error_hat) / (k*MSE)
    temp2 = np.mat(Hii / np.multiply(1-Hii, 1-Hii))
    Di = np.multiply(temp1, temp2.T)

    # 打印
    print('+{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')))
    print('|{x:^{y}}|'.format(x='', y=6 + len('error_hat')) +
          '{x:^{y}}|'.format(x='error_hat', y=6 + len('error_hat')) +
          '{x:^{y}}|'.format(x='Std_error', y=6 + len('error_hat')) +
          '{x:^{y}}|'.format(x='Stu_error', y=6 + len('error_hat')) +
          '{x:^{y}}|'.format(x='DFFITSi', y=6 + len('error_hat')) +
          '{x:^{y}}|'.format(x='Di', y=6 + len('error_hat')))
    print('+{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')))
    for i in range(len(y_hat)):
        print('|{x:^{y}}|'.format(x='y_hat_%d' % (i+1), y=6 + len('error_hat')) +
              '{x:^{y}}|'.format(x='%.3f' % error_hat[i], y=6 + len('error_hat')) +
              '{x:^{y}}|'.format(x='%.3f' % Std_error[i], y=6 + len('error_hat')) +
              '{x:^{y}}|'.format(x='%.3f' % Stu_error[i], y=6 + len('error_hat')) +
              '{x:^{y}}|'.format(x='%.3f' % DFFITSi[i], y=6 + len('error_hat')) +
              '{x:^{y}}|'.format(x='%.3f' % Di[i], y=6 + len('error_hat')))
        print('+{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('error_hat')))

    return

def TheResultsShow(X, y):
    N = X.shape[0]
    K = X.shape[1]
    freedom = N - K
    Beta_test = np.mat(np.zeros(K)).T

    Beta_hat, y_hat, error_hat = LeastSquareMethod(X, y)
    R_square, Adjusted_R_square = GoodnessOfFit(y, error_hat, N, K)
    Beta_var_hat, Pt = TTest(X, Beta_hat, error_hat, Beta_test)
    Pf = FTest(y, error_hat, N, K)
    MSE = error_hat.T * error_hat / freedom
    error_ratio = ErrorRatio(y, y_hat)

    # 打印
    print('+{x:-^{y}}+'.format(x='', y=6 + len('Number of obs')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Prob > F')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('R-squared')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Adj R-squared')) +
          '{x:-^{y}}+'.format(x='', y=8 + len('MSE')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('ErrorRatio')))
    print('|{x:^{y}}|'.format(x='Number of obs', y=6 + len('Number of obs')) +
          '{x:^{y}}|'.format(x='Prob > F', y=6 + len('Prob > F')) +
          '{x:^{y}}|'.format(x='R-squared', y=6 + len('R-squared')) +
          '{x:^{y}}|'.format(x='Adj R-squared', y=6 + len('Adj R-squared')) +
          '{x:^{y}}|'.format(x='MSE', y=8 + len('MSE')) +
          '{x:^{y}}|'.format(x='ErrorRatio', y=6 + len('ErrorRatio')))
    print('+{x:-^{y}}+'.format(x='', y=6 + len('Number of obs')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Prob > F')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('R-squared')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Adj R-squared')) +
          '{x:-^{y}}+'.format(x='', y=8 + len('MSE')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('ErrorRatio')))
    print('|{x:^{y}}|'.format(x='%d' % N, y=6 + len('Number of obs')) +
          '{x:^{y}}|'.format(x='%.3f' % Pf, y=6 + len('Prob > F')) +
          '{x:^{y}}|'.format(x='%.3f' % R_square, y=6 + len('R-squared')) +
          '{x:^{y}}|'.format(x='%.3f' % Adjusted_R_square, y=6 + len('Adj R-squared')) +
          '{x:^{y}}|'.format(x='%.3f' % MSE, y=8 + len('MSE')) +
          '{x:^{y}}|'.format(x='%.3f' % error_ratio, y=6 + len('ErrorRatio')))
    print('+{x:-^{y}}+'.format(x='', y=6 + len('Number of obs')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Prob > F')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('R-squared')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Adj R-squared')) +
          '{x:-^{y}}+'.format(x='', y=8 + len('MSE')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('ErrorRatio')))

    print('+{x:-^{y}}+'.format(x='', y=6 + len('Beta_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Beta_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Std. err.')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('P>|t|')))
    print('|{x:^{y}}|'.format(x='', y=6 + len('Beta_hat')) +
          '{x:^{y}}|'.format(x='Beta_hat', y=6 + len('Beta_hat')) +
          '{x:^{y}}|'.format(x='Std. err.', y=6 + len('Std. err.')) +
          '{x:^{y}}|'.format(x='P>|t|', y=6 + len('P>|t|')))
    print('+{x:-^{y}}+'.format(x='', y=6 + len('Beta_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Beta_hat')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('Std. err.')) +
          '{x:-^{y}}+'.format(x='', y=6 + len('P>|t|')))
    for i in range(len(Beta_hat)):
        print('|{x:^{y}}|'.format(x='Beta_%d' % i, y=6 + len('Beta_hat')) +
              '{x:^{y}}|'.format(x='%.3f' % Beta_hat[i], y=6 + len('Beta_hat')) +
              '{x:^{y}}|'.format(x='%.3f' % Beta_var_hat[i], y=6 + len('Std. err.')) +
              '{x:^{y}}|'.format(x='%.3f' % Pt[i], y=6 + len('P>|t|')))
        # %(i+1, Beta_hat1[i], Beta_hat2[i], Beta_hat3[i]))
        print('+{x:-^{y}}+'.format(x='', y=6 + len('Beta_hat')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('Beta_hat')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('Std. err.')) +
              '{x:-^{y}}+'.format(x='', y=6 + len('P>|t|')))

    # 如果数据为2维或3维则绘图展示
    if K == 2:
        plt.scatter(np.array(X[:,1]), np.array(y))
        plt.plot(np.array(X[:,1]), np.array(y_hat), c='red')
        plt.show()
    elif K == 3:
        ax = plt.axes(projection='3d')  # 绘制3d图形
        ax.scatter(np.array(X[:, 1]), np.array(X[:, 2]), np.array(y), c='r')
        arr_X, arr_Y = np.meshgrid(np.array(X[:,1]), np.array(X[:,2]))
        Z = float(Beta_hat[0]) + float(Beta_hat[1])*arr_X + float(Beta_hat[2])*arr_Y
        ax.plot_surface(arr_X, arr_Y, Z, alpha=0.5, color = "g")
        plt.show()
    else:
        print('维数太高无法绘图展示')

    RegressionDiagnosis(X, y_hat, error_hat)

    return