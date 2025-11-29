import numpy as np
import matplotlib.pyplot as plt


def correl(x, k):
    Rxx = np.zeros(k)
    Mx = np.mean(x)
    n = len(x)

    for tau in range(k):
        for i in range(0, n - tau):
            Rxx[tau] += (x[i] - Mx) * (x[i + tau] - Mx)
        Rxx[tau] /= (n - tau)

    return Rxx

if __name__ == "__main__":
    k = 1000
    Mx = 0
    Dx = 0.25
    sigma = np.sqrt(Dx)

    x = np.random.normal(Mx, sigma, k)

    p_Mx = np.mean(x)
    p_Dx = np.var(x)
    print(f"Вычисленное мат. ожидание: {p_Mx}")
    print(f"Погрешность мат. ожидания: {abs(p_Mx - Mx)}")
    print(f"Вычисленная дисперсия: {p_Dx}")
    print(f"Погрешность дисперсии: {abs(p_Dx - Dx)}")

    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.title('Нормально распределенный белый шум')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(correl(x, 16))
    plt.title('Корреляционная функция белого шума Rxx')
    plt.grid()
    plt.show()


