import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_data(data_path):
    df = pd.read_csv(data_path)
    t = df['t'].values
    I = df['I'].values
    return t, I


def solve(t, I, r_beta):
    N, I0 = 0, 0
    plt.plot(t, I)
    plt.show()

    return N, I0


if __name__ == "__main__":
    r_beta = 1
    data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    t, I = load_data(data_path)
    N, I0 = solve(t, I, r_beta)
    print(N, I0)
