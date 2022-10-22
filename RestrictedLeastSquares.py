


def RestrictedLeastSquares(X, y, X_res, y_res):
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

    Nbb = X.T * X
    Ncc = X_res * Nbb.I * X_res.T
    temp1 = Nbb.I - Nbb.I * X_res.T * Ncc.I * X_res * Nbb.I
    temp2 = Nbb.I * X_res.T * Ncc.I
    Beta_hat = temp1 * (X.T * y) + temp2 * y_res

    y_hat = X * Beta_hat
    error_hat = y - y_hat

    return Beta_hat, y_hat, error_hat