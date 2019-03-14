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


def newton_method(a, f, f1, eps):
    max_count = 5000
    iter_count = 0
    x0 = a
    x1 = x0 - (f(x0) / f1(x0))

    while math.fabs(f(x1) - f(x0)) >= eps:
        x0 = x1
        x1 = x0 - (f(x0) / f1(x0))
        iter_count += 1
        if iter_count > max_count:
            return None
    return x1


def calculate_lambda_2(beta):
    return (beta**2 - 1)**0.5


def find_stationary_point(alpha, n):
    a = 0
    eps = 10 ** (-6)
    autoregressor = Autoregressor(alpha, n)
    stationary_point = newton_method(a, autoregressor.function, autoregressor.derivative, eps)
    return stationary_point


def calculate_beta(alpha, n):
    x = find_stationary_point(alpha, n)
    beta = (alpha * n * x**(n - 1))/(1 + x**n)**2
    return beta


def calculate_tau(alpha, n):
    beta = calculate_beta(alpha, n)
    if beta > 1:
        lambda_2 = calculate_lambda_2(beta)
        tau = (1./lambda_2)*math.acos(-1./beta)
        return tau
    else:
        return None


# Для разных n строим зависимость tay(alpha)
def task1():
    plt.clf()
    for n in [2, 4, 6]:
        points = []
        alpha = 0.1
        step_alfa = 0.1
        while alpha <= 10:
            tau = calculate_tau(alpha, n)
            if tau != None:
                points.append((alpha, calculate_tau(alpha, n)))
            alpha += step_alfa
        label = 'n = ' + str(n)
        x, y = zip(*points)
        plt.plot(x, y, label=label)

    plt.legend()
    plt.title(r'Зависимость $\tau$ от $\alpha$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\tau(\alpha)$')
    plt.grid(True)
    #plt.show()
    file_name = 'Зависимость тау от альфы при разных n'
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab2')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=300)


def main():
    task1()


if __name__ == "__main__":
    main()