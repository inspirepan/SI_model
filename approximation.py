import numpy as np
import matplotlib.pyplot as plt
from interpolation import func


N = 1000
I0 = 1
r_beta = 1
e = np.e


def get_p_func(k):
    """
    返回k阶的勒让德多项式计算函数
    """
    if k == 0:
        def p(x):
            return 1

        return p
    elif k == 1:
        def p(x):
            return x

        return p
    else:
        pk_1 = get_p_func(k - 1)
        pk_2 = get_p_func(k - 2)

        def p(x):
            return ((2 * k - 1) * x * pk_1(x) - (k - 1) * pk_2(x)) / k

        return p


def get_t(n):
    """
    返回 e^(5x)*x^n 在[-1,1]上积分的结果
    """
    if n == 0:
        return (e ** 5 - e ** (-5)) / 5
    else:
        return (e ** 5 - (-1) ** n * e ** (-5)) / 5 - n / 5 * get_t(n - 1)


def get_p_coeffs(k):
    """
    返回Pk(x)各阶次的系数
    :return 返回一个函数，可以获取各阶次的系数
    """
    if k == 0:
        def p0_coeffs(n):
            if n == 0:
                return 1
            else:
                return 0

        return p0_coeffs
    elif k == 1:
        def p1_coeffs(n):
            if n == 1:
                return 1
            else:
                return 0

        return p1_coeffs
    else:
        pk_1_coeffs = get_p_coeffs(k - 1)
        pk_2_coeffs = get_p_coeffs(k - 2)

        def pk_coeffs(n):
            if n == 0:
                return -(k - 1) / k * pk_2_coeffs(0)
            else:
                return -(k - 1) / k * pk_2_coeffs(n) + (2 * k - 1) / k * pk_1_coeffs(n - 1)

        return pk_coeffs


def get_fp(k):
    """
    返回 e^(5x)*Pk(x) 在[-1,1]上积分的结果
    """
    ans = 0
    pk_coeffs = get_p_coeffs(k)
    for i in range(k + 1):
        ans += pk_coeffs(i) * get_t(i)
    return ans


def get_delta_func():
    """
    获取方法误差计算函数
    """
    # 为了避免重复计算，用一个map记录已经算过的delta
    delta_dict = dict()

    def delta(k):
        """
        返回k阶次的方法误差
        """
        if k in delta_dict:
            return delta_dict[k]
        else:
            if k == 0:
                ans = 5 * (e ** 10) * ((e ** 20 - 1) / (10 * e ** 10) - 0.5 * (get_fp(0)) ** 2)
                delta_dict[0] = ans
                return ans
            else:
                ans = delta(k - 1) - 5 * e ** 10 * (2 * k + 1) / 2 * (get_fp(k)) ** 2
                delta_dict[k] = ans
                return ans

    return delta


def get_s_coeffs(n):
    """
    返回n阶次的勒让德逼近系数
    """
    s_coeffs = dict()

    for i in range(n + 1):
        pi_coeffs = get_p_coeffs(i)
        for k in range(i + 1):
            if k not in s_coeffs:
                s_coeffs[k] = (2 * i + 1) / 2 * get_fp(k) * pi_coeffs(k)
            else:
                s_coeffs[k] += (2 * i + 1) / 2 * get_fp(k) * pi_coeffs(k)
    return s_coeffs


def get_approx_func(coeffs):
    def x(t):
        """
        区间转换
        """
        return t / 5 - 1

    def S(t):
        return t

    # 确定阶次
    n = 0
    delta = get_delta_func()
    d = delta(n)
    while d > coeffs:
        print("阶次{}   方法误差{}".format(n, d))
        n += 1
        d = delta(n)
    print("最终采用阶次 {}     （）方法误差 {}".format(n, d))
    print("S(x) 的系数")
    print(get_s_coeffs(n))

    def func(t):
        def x(t):
            """
            区间变换，自变量变成x
            """
            return t / 5 - 1

        def S(x):
            """
            用最高n阶的勒让德多项式逼近
            """
            ans = 0
            for i in range(n + 1):
                pi = get_p_func(i)
                fp = get_fp(i)
                # 最高阶次为n
                ans += (2 * i + 1) / 2 * fp * pi(x)
            return ans

        return 1000 / (1 + 999 / (e ** 5 * S(x(t))))

    return func


if __name__ == "__main__":
    bound = input("Please input the error bound:")
    approx_func = get_approx_func(float(bound))

    f, (ax1, ax2) = plt.subplots(1, 2)
    t = np.arange(0, 10, 0.1)
    I_approx = approx_func(t)
    I = func(t)

    ax1.plot(t, I)
    ax2.plot(t, I_approx, c='r')
    plt.show()
