import numpy as np
import matplotlib.pyplot as plt
from task1 import correl


def correl_interval_decay(y, threshold=1 / np.e):

    max_tau = 100
    Ryy = np.zeros(max_tau)

    My = np.mean(y)
    for tau in range(max_tau):
        for i in range(len(y) - tau):
            Ryy[tau] += (y[i] - My) * (y[i + tau] - My)
        Ryy[tau] /= (len(y) - tau)

    Ryy_norm = np.abs(Ryy / Ryy[0])

    for tau in range(1, max_tau):
        if Ryy_norm[tau] < threshold:
            return tau

    return max_tau


def autoregression_markov(x, a):
    n = len(x)
    y = np.zeros(n)
    y[0] = x[0]
    for k in range(1, n):
        y[k] = a * y[k - 1] + x[k]

    return y

def cross_correl(x, y, k):
    Rxy = np.zeros(k)
    Mx = np.mean(x)
    My = np.mean(y)
    n = len(x)

    for tau in range(k):
        for i in range(0, n - tau):
            Rxy[tau] += (x[i] - Mx) * (y[i + tau] - My)
        Rxy[tau] /= (n - tau)

    return Rxy

if __name__ == "__main__":
    k = 1000
    Mx = 0
    Dx = 0.25
    sigma = np.sqrt(Dx)
    a = 0.2

    x = np.random.normal(Mx, sigma, k)
    y = autoregression_markov(x, a)

    p_My = np.mean(y)
    p_Dy = np.var(y)
    print(f"Вычисленное мат. ожидание: {p_My}")
    print(f"Погрешность мат. ожидания: {abs(p_My - (Mx / (1 - a)))}")  #0
    print(f"Вычисленная дисперсия: {p_Dy}")
    print(f"Погрешность дисперсии: {abs(p_Dy - (Dx / (1 - a**2)))}")  #(25/36)
    print(f'Интервал корреляции случайного процесса: {1 / (1 - abs(a))}')
    print(f'Интервал корреляции СП  через затухание (1/e): {correl_interval_decay(y)}')

    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.title('Нормально распределенный белый шум')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(y)
    plt.title(f'Авторегрессионная модель Маркова y(k) = {a}y(k-1) + x(k)')
    plt.grid()
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(correl(y, 16))
    plt.title('Корреляционная функция белого шума Ryy')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(cross_correl(x, y, 16))
    plt.title('Взаимная корреляционная функция Rxy')
    plt.grid()
    plt.show()

