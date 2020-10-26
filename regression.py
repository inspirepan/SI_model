import os
import numpy as np
import pandas as pd


def load_data(data_path):
    df = pd.read_csv(data_path)
    t = df['t'].values
    I = df['I'].values
    return t, I


def solve(t, I, r_beta):
    x = np.exp(-r_beta * t)
    y = 1 / I
    phi0_phi0 = t.shape[0]
    phi0_phi1 = sum(x)
    phi1_phi1 = sum(x ** 2)
    phi0_f = sum(y)
    phi1_f = sum(x * y)
    # 得到线性方程组
    # phi0_phi0 * a0 + phi0_phi1 * a1 = phi0_f
    # phi0_phi1 * a0 + phi1_phi1 * a1 = phi1_f
    a0 = (phi0_f / phi0_phi1 * phi1_phi1 - phi1_f) / (phi0_phi0 / phi0_phi1 * phi1_phi1 - phi0_phi1)
    a1 = (phi0_f / phi0_phi0 * phi0_phi1 - phi1_f) / (phi0_phi1 / phi0_phi0 * phi0_phi1 - phi1_phi1)
    # 得到N和I0
    # y = (1/I0-1/N) * x + 1/N
    # 四舍五入取整
    N = np.rint(1 / a0)
    I0 = np.rint(1 / (a1 + a0))
    return N, I0


if __name__ == "__main__":
    r_beta = 1
    data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    t, I = load_data(data_path)
    N, I0 = solve(t, I, r_beta)
    print(N, I0)
