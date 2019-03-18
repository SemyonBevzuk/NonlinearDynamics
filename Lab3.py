import math
import matplotlib.pyplot as plt
import os


class Autoregressor(object):
    def __init__(self, alfa, n):
        self.alfa = alfa
        self.n = n

    def function(self, x):
        return x**(self.n + 1) + x - self.alfa

    def derivative(self, x):
        return (self.n + 1) * x**self.n + 1


def ody(x):
    return x


def answer(x0, t):
    return x0*math.exp(t)


def euler_method(f, y0, h, t_max):
    x = []
    y = []
    y_i = y0
    t_i = 0
    while t_i < t_max:
        x.append(t_i)
        y.append(y_i)
        t_i += h
        y_i = y_i + f(y_i)*h
    return x, y


def task1():
    plt.clf()
    t_max = 5;
    h = 0.1;
    x0 = 10;
    x, y = euler_method(ody, x0, h, t_max)
    label = 'Численное решение Эйлером'
    plt.plot(x, y, label=label)

    x = []
    y = []
    t = 0
    while t<t_max:
        x.append(t)
        y.append(answer(x0, t))
        t += h/10.
    label = 'Аналитическое решение'
    plt.plot(x, y, label=label)

    plt.legend()
    plt.title(r'$\dot{x} = x$')
    plt.xlabel(r't')
    plt.ylabel(r'x')
    plt.grid(True)
    #plt.show()
    file_name = 'Решение системы 1 методом Эйлера'
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab3')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=300)


def main():
    task1()


if __name__ == "__main__":
    main()