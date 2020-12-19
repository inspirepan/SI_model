import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N = 1000
I0 = 1
r_beta = 1


def load_table():
    df = pd.read_csv('LUT.csv')
    t = df['t'].values
    I = df['I'].values
    return t, I


def get_hermite_func(t, I):
    # 计算微分I'
    dI = list(map(lambda x: x - x ** 2 / N, I))

    def hermite_func(x_list):
        def H(x):
            # 使用两点的三次Hermite插值
            x_index = np.where(t == x)[0]
            # 如果x是t中已经有的点
            if len(x_index) != 0:
                return I[x_index[0]]
            # 确定x附近的两个插值结点的值
            t0 = np.floor(x)
            t1 = np.ceil(x)
            mid = t0 + 0.5
            if x > mid:
                t0 = mid
            else:
                t1 = mid
            # 获取t0和t1对应的index
            t0_index = np.where(t == t0)[0][0]
            t1_index = np.where(t == t1)[0][0]
            # 获取4个已知条件
            I0 = I[t0_index]
            I1 = I[t1_index]
            dI0 = dI[t0_index]
            dI1 = dI[t1_index]
            # 根据教材式(4.12)求解H(x)
            alpha0 = (1 + 4 * (x - t0)) * 4 * (x - t1) ** 2
            alpha1 = (1 - 4 * (x - t1)) * 4 * (x - t0) ** 2
            beta0 = (x - t0) * 4 * (x - t1) ** 2
            beta1 = (x - t1) * 4 * (x - t0) ** 2
            return alpha0 * I0 + alpha1 * I1 + beta0 * dI0 + beta1 * dI1

        return np.asarray(list(map(H, x_list)))

    return hermite_func


def func(t):
    # calculate I(t)
    It = N * I0 / (I0 + (N - I0) * np.exp(-r_beta * t))
    return It


if __name__ == "__main__":
    tk, I_tk = load_table()
    t = np.arange(0, 15, 0.1)

    hermite_func = get_hermite_func(tk, I_tk)

    # calculate
    I = func(t)
    I_hat = hermite_func(t)
    plt.plot(t, I, label="I")
    plt.plot(t, I_hat, label="I_hat")
    plt.scatter(tk, I_tk, label="I_tk", color="mediumseagreen", s=10)
    plt.legend()
    plt.show()
    print(np.abs(I_hat - I).max())
