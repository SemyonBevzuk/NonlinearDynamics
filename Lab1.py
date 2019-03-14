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


class GeneSwitch(object):
    def __init__(self, alfa, n):
        self.alfa = alfa
        self.n = n

    def function(self, x):
        return x*(1+x**self.n)**self.n-self.alfa*(1+x**self.n)**self.n+x*self.alfa**self.n

    def derivative(self, x):
        return self.alfa**self.n + (self.n**2)*((x**self.n + 1)**(self.n - 1))*(x**self.n - self.alfa*(x**(self.n - 1))) + (x**self.n + 1)**self.n

    def P2(self, x):
        return x**2-self.alfa*x+1

    def P2_derivative(self, x):
        return 2*x-self.alfa

    def P3(self, x):
        return x ** 3 + x - self.alfa

    def P3_derivative(self, x):
        return 3 * x ** 2 + 1


def bisection_method(a, b, f, eps, additional_information=False):
    if f(a)*f(b) > 0:
        print('Функция должна иметь разные знаки на концах отрезка')
        return None
    else:
        # для графика из второго задания
        a_k = [a]
        b_k = [b]
        difference_between_points = [math.fabs(b - a)]

        #while math.fabs(a - b) >= eps:
        while math.fabs(f(a) - f(b)) >= eps:
            c = (a + b) / 2
            if f(a) * f(c) > 0:
                a, b = c, b
            else:
                a, b = a, c
            a_k.append(a)
            b_k.append(b)
            difference_between_points.append(math.fabs(b-a))

        if additional_information:
            return (a + b) / 2., a_k, b_k, difference_between_points
        else:
            return (a + b) / 2.


def newton_method(a, f, f1, eps, additional_information=False):
    max_count = 5000
    iter_count = 0
    x0 = a
    x1 = x0 - (f(x0) / f1(x0))
    # для графика из второго задания
    x_k = [x1]
    difference_between_points = [math.fabs(x1-x0)]

    #while math.fabs(x1 - x0) >= eps:
    while math.fabs(f(x1) - f(x0)) >= eps:
        x0 = x1
        x1 = x0 - (f(x0) / f1(x0))
        x_k.append(x1)
        difference_between_points.append(math.fabs(x1-x0))
        iter_count += 1
        if iter_count > max_count:
            return None

    if additional_information:
        return x1, x_k, difference_between_points
    else:
        return x1


# Для разных n строим x^*(\alfa) методом Ньютона
def task1_newton():
    a = 0
    eps = 10 ** (-6)
    plt.clf()
    for n in [2, 4, 6]:
        points = []
        alfa = 0
        step_alfa = 0.1
        while alfa <= 10:
            autoregressor = Autoregressor(alfa, n)
            points.append((alfa, newton_method(a, autoregressor.function, autoregressor.derivative, eps)))
            alfa += step_alfa
        label = 'n = ' + str(n)
        x, y = zip(*points)
        plt.plot(x, y, label=label)

    plt.legend()
    plt.title('Метод Ньютона')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$x^*$')
    plt.grid(True)
    # plt.show()
    file_name = 'Зависимость решения от альфы при разных n метод ньютона'
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab1')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=300)


# Для разных n строим x^*(\alfa) методом бисекции
def task1_bisection():
    a = 0
    b = 100
    eps = 10 ** (-6)
    plt.clf()
    for n in [2, 4, 6]:
        points = []
        alfa = 0
        step_alfa = 0.1
        while alfa <= 10:
            autoregressor = Autoregressor(alfa, n)
            points.append((alfa, bisection_method(a, b, autoregressor.function, eps)))
            alfa += step_alfa
        label = 'n = ' + str(n)
        x, y = zip(*points)
        plt.plot(x, y, label=label)

    plt.legend()
    plt.title('Метод бисекции')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$x^*$')
    plt.grid(True)
    # plt.show()
    file_name = 'Зависимость решения от альфы при разных n метод бисекции'
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab1')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=300)


# Для конкретных n и альфа сравнить методы. Построить два графика:
# 1) Зависимость текущей точки от шага метода
# 2) Зависимость ошибки от шага
def task2():
    n = 4
    alfa = 2
    eps = 10 ** (-6)
    a = 0
    b = 10
    autoregressor = Autoregressor(alfa, n)
    plt.clf()

    point, x_k, difference_between_x_newton = newton_method(a, autoregressor.function, autoregressor.derivative, eps, True)
    x = range(0, len(x_k))
    plt.plot(x, x_k, label='Метод Ньютона')

    point, a_k, b_k, difference_between_x_bisection = bisection_method(a, b, autoregressor.function, eps, True)
    x = range(0, len(a_k))
    plt.plot(x, a_k, label='Метод бисекции (левая граница)')
    plt.plot(x, b_k, label='Метод бисекции (правая граница)')

    plt.legend()
    plt.title('n = ' + str(n) + r'$\:\alpha = $' + str(alfa))
    plt.xlabel(r'k')
    plt.ylabel(r'$x_{k}$')
    plt.grid(True)
    # plt.show()
    file_name = 'Зависимость решения от шага метода'
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab1')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=300)

    plt.clf()
    plt.xscale('log')
    plt.yscale('log')
    x = range(0, len(difference_between_x_newton))
    plt.plot(x, difference_between_x_newton, label='Метод Ньютона')
    x = range(0, len(difference_between_x_bisection))
    plt.plot(x, difference_between_x_bisection, label='Метод бисекции')
    plt.legend()
    plt.title('n = ' + str(n) + r'$\:\alpha = $' + str(alfa))
    plt.xlabel(r'$\log (k)$')
    plt.ylabel(r'$\log |x_{k+1}-x_{k}|$')
    plt.grid(True)
    # plt.show()
    file_name = 'Зависимость окрестности решения от шага метода'
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab1')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=300)


# Построить Р5=Р3*Р2
# Построить график зависимости альфы и корней
def task3():
    n = 2
    eps = 10 ** (-6)
    x0 = -100
    alfa = 0
    step_alfa = 0.1
    points_P3 = []
    points_P2 = []
    while alfa <= 10:
        gene_switch = GeneSwitch(alfa, n)
        points_P3.append((alfa, newton_method(x0, gene_switch.P3, gene_switch.P3_derivative, eps)))
        if alfa >= 2: # значит есть один или два корня
            roots = set()
            for x0 in range(-100, 100, 10):
                roots.add(newton_method(x0, gene_switch.P2, gene_switch.P2_derivative, eps))
                for root in roots:
                    points_P2.append((alfa, root))
        alfa += step_alfa
    plt.clf()
    x, y = zip(*points_P3)
    plt.plot(x, y, label='Корни $P_{3}$')
    points_P2.sort(key=lambda i: i[1])
    x, y = zip(*points_P2)
    plt.plot(x, y, label='Корни $P_{2}$')
    plt.legend()
    plt.title('n = ' + str(n))
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$x^*$')
    plt.grid(True)
    #plt.show()
    file_name = 'Зависимость корней от альфы'
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab1')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=300)


def main():
    task1_newton()
    task1_bisection()
    task2()
    task3()


if __name__ == "__main__":
    main()