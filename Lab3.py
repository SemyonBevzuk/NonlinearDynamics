import math
import matplotlib.pyplot as plt
import os
import random
import time

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ~~~~~~~~~~~~~~~~~~~~~~ Part №1 ~~~~~~~~~~~~~~~~~~~~~~
def euler_method(f, x0, h, t_max):
    points = []
    x_i = x0
    t_i = 0
    while t_i < t_max:
        points.append((t_i, x_i))
        t_i += h
        f_i = f(x_i, t_i)
        x_i = [x_k + f_k * h for (x_k, f_k) in zip(x_i, f_i)]
    return points


def runge_kutta_method(f, x0, h, t_max):
    points = []
    x_i = x0
    t_i = 0
    while t_i < t_max:
        points.append((t_i, x_i))
        k1 = f(x_i, t_i)
        x_intermediate = [x_elem + 0.5 * h * k1_elem for (x_elem, k1_elem) in zip(x_i, k1)]
        k2 = f(x_intermediate, t_i + 0.5 * h)
        x_intermediate = [x_elem + 0.5 * h * k2_elem for (x_elem, k2_elem) in zip(x_i, k2)]
        k3 = f(x_intermediate, t_i + 0.5 * h)
        x_intermediate = [x_elem + h * k3_elem for (x_elem, k3_elem) in zip(x_i, k3)]
        k4 = f(x_intermediate, t_i + h)
        x_i = [x_elem + (h / 6.0) * (k1_elem + 2 * k2_elem + 2 * k3_elem + k4_elem)\
               for (x_elem, k1_elem, k2_elem, k3_elem, k4_elem)\
               in zip(x_i, k1, k2, k3, k4)]
        t_i += h
    return points


# x' = x
def stable_system(x, t):
    return [-elem for elem in x]


def stable_system_solution(x0, t):
    return [elem*math.exp(-t) for elem in x0]


# x' = -x
def unstable_system(x, t):
    return [elem for elem in x]


def unstable_system_solution(x0, t):
    return [elem*math.exp(t) for elem in x0]


# x'' + x = 0
def pendulum_system(x, t):
    x1 = x[0]
    x2 = x[1]
    return [x2, -x1]


def pendulum_system_solution(x0, t):
    return [math.sin(t), math.cos(t)]

# Система Рёсслера
def rossler_system(x, t):
    a = b = 0.2
    r = 5.7
    x1 = - x[1] - x[2]
    x2 = x[0] + a*x[1]
    x3 = b + (x[0] - r)*x[2]
    return [x1, x2, x3]


def save_img(_plt, file_name_solution, xlable, ylable):
    _plt.legend()
    _plt.title(file_name_solution)
    _plt.xlabel(xlable)
    _plt.ylabel(ylable)
    _plt.grid(True)
    file_name = file_name_solution
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab3')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    _plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=300)


def solve_test_system_and_save_plots(system, solution, x0, h, t_max, file_name_solution, file_name_errors):
    points = euler_method(system, x0, h, t_max)
    x_euler, y_euler = zip(*points)
    points = runge_kutta_method(system, x0, h, t_max)
    x_runge_kutta, y_runge_kutta = zip(*points)
    points = []
    t = 0
    while t < t_max:
        points.append((t, solution(x0, t)))
        t += h / 10.0
    x, y = zip(*points)

    plt.clf()
    plt.plot(x_euler, y_euler, label='Метод Эйлера')
    plt.plot(x_runge_kutta, y_runge_kutta, label='Метод Рунге-Кутта')
    plt.plot(x, y, '--', label='Аналитическое решение')
    save_img(plt, file_name_solution, r't', r'x')

    points = []
    t = 0
    for euler, runge_kutta in zip(y_euler, y_runge_kutta):
        exact_value = solution(x0, t)
        delta_euler = [math.fabs(e - s) for (e, s) in zip(euler, exact_value)]
        delta_runge_kutta = [math.fabs(rk - s) for (rk, s) in zip(runge_kutta, exact_value)]
        points.append((t, delta_euler, delta_runge_kutta))
        t += h
    time, delta_euler, delta_runge_kutta = zip(*points)
    plt.clf()
    # plt.xscale('log')
    plt.plot(time, delta_euler, label='Ошибка метода Эйлера')
    plt.plot(time, delta_runge_kutta, label='Ошибка метода Рунге-Кутта')
    save_img(plt, file_name_errors, r't', r'$\Delta x = |x(t) - \widetilde{x}(t)|$')


def task1_stable(x0, h, t_max):
    plt.clf()
    file_name_solution = 'Решение устойчивой системы'
    file_name_errors = 'График ошибки методов на устойчивой системе'
    solve_test_system_and_save_plots(stable_system, stable_system_solution, x0, h, t_max,\
                                     file_name_solution, file_name_errors)


def task1_unstable(x0, h, t_max):
    plt.clf()
    t_max = 5.0
    h = 0.1
    x0 = [10.0]
    file_name_solution = 'Решение неустойчивой системы'
    file_name_errors = 'График ошибки методов на неустойчивой системе'
    solve_test_system_and_save_plots(unstable_system, unstable_system_solution, x0, h, t_max, \
                                     file_name_solution, file_name_errors)


# Исследовать сходимость методов Эйлера и РК4
def task1():
    t_max = 5.0
    h = 0.1
    x0 = [10.0]
    task1_stable(x0, h, t_max)
    task1_unstable(x0, h, t_max)


# Решить систему с маятником и посмотреть на график ошибок
def task2_pendulum():
    x0 = [0.0, 1.0]
    h = 0.01
    t_max = 30.0

    points = euler_method(pendulum_system, x0, h, t_max)
    x_euler, y_euler = zip(*points)
    points = runge_kutta_method(pendulum_system, x0, h, t_max)
    x_runge_kutta, y_runge_kutta = zip(*points)
    points = []
    t = 0
    while t < t_max:
        points.append((t, pendulum_system_solution(x0, t)[0]))
        t += h / 10.0
    x, y = zip(*points)

    plt.clf()
    plt.plot(x_euler, [elem[0] for elem in y_euler], label='Метод Эйлера')
    plt.plot(x_runge_kutta, [elem[0] for elem in y_runge_kutta], label='Метод Рунге-Кутта')
    plt.plot(x, y, '--', label='Аналитическое решение')
    save_img(plt, 'Уравнение маятника', r't', r'x')
    #plt.xscale('log')

    points = []
    t = 0
    for euler, runge_kutta in zip(y_euler, y_runge_kutta):
        exact_value = pendulum_system_solution(x0, t)
        delta_euler = [math.fabs(euler[0] - exact_value[0])]
        delta_runge_kutta = [math.fabs(runge_kutta[0] - exact_value[0])]
        points.append((t, delta_euler, delta_runge_kutta))
        t += h
    time, delta_euler, delta_runge_kutta = zip(*points)
    plt.clf()
    plt.plot(time, delta_euler, label='Ошибка метода Эйлера')
    plt.plot(time, delta_runge_kutta, label='Ошибка метода Рунге-Кутта')
    save_img(plt, 'График ошибок методов на системе с маятником', r't', r'$\Delta x = |x(t) - \widetilde{x}(t)|$')


# Исследовать модель Рёсслера и оценить ошибку через меньший шаг
def task3():
    x0 = [0., 0., 0.035]
    h = 0.01
    t_max = 300.0
    points = euler_method(rossler_system, x0, h, t_max)
    x_euler_1, y_euler_1 = zip(*points)
    points = euler_method(rossler_system, x0, h / 2.0, t_max)
    x_euler_2, y_euler_2 = zip(*points)
    plt.clf()
    plt.plot([elem[0] for elem in y_euler_1], [elem[1] for elem in y_euler_1], label='y(x)')
    save_img(plt, 'Решение системы Рёсслера методом Эйлера y(x)', r'x', r'y')
    plt.clf()
    plt.plot([elem[0] for elem in y_euler_1], [elem[2] for elem in y_euler_1], label='z(x)')
    save_img(plt, 'Решение системы Рёсслера методом Эйлера z(x)', r'x', r'z')

    points = runge_kutta_method(rossler_system, x0, h, t_max)
    x_runge_kutta_1, y_runge_kutta_1 = zip(*points)
    points = runge_kutta_method(rossler_system, x0, h / 2.0, t_max)
    x_runge_kutta_2, y_runge_kutta_2 = zip(*points)
    plt.clf()
    plt.plot([elem[0] for elem in y_runge_kutta_1], [elem[1] for elem in y_runge_kutta_1], label='y(x)')
    save_img(plt, 'Решение системы Рёсслера методом Рунге-Кутта y(x)', r'x', r'y')
    plt.clf()
    plt.plot([elem[0] for elem in y_runge_kutta_1], [elem[2] for elem in y_runge_kutta_1], label='z(x)')
    save_img(plt, 'Решение системы Рёсслера методом Рунге-Кутта z(x)', r'x', r'z')

    points = []
    for (i,j) in zip(range(0, len(x_runge_kutta_1)), range(0, len(x_runge_kutta_2), 2)):
        delta_x_euler = math.fabs(y_euler_1[i][0] - y_euler_2[j][0])
        delta_x_runge_kutta = math.fabs(y_runge_kutta_1[i][0] - y_runge_kutta_2[j][0])
        points.append((x_runge_kutta_1[i], delta_x_euler, delta_x_runge_kutta))
    x, y_euler, y_runge_kutta = zip(*points)

    plt.clf()
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x, y_euler, label='Метод Эйлера')
    plt.plot(x, y_runge_kutta, label='Метод Рунге-Кутта')
    save_img(plt, 'График ошибок методов на системе Рёсслера', r'$\ln(t)$', r'$\Delta x = |x_h(t) - x_{\frac{h}{2}}(t)|$')


# Посмотреть на аттрактор - бесценно
def task3_show_XYZ():
    x0 = [0., 0., 0.035]
    h = 0.01
    t_max = 1000.0
    points = runge_kutta_method(rossler_system, x0, h, t_max)
    x_runge_kutta, y_runge_kutta = zip(*points)
    x = np.array([elem[0] for elem in y_runge_kutta])
    y = np.array([elem[1] for elem in y_runge_kutta])
    z = np.array([elem[2] for elem in y_runge_kutta])

    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='Система Рёсслера')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #plt.show()
    file_name = 'Решение системы Рёсслера в фазовом пространстве'
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab3')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=300)


# ~~~~~~~~~~~~~~~~~~~~~~ Part №2 ~~~~~~~~~~~~~~~~~~~~~~
# x' = -x
def test_system(x, t):
    return [elem for elem in x]


def test_system_solution(x0, t):
    return [elem*math.exp(-t) for elem in x0]


# Система с синтезом водорода и его распадом на атомы
def synthesis_decay_system(x, t):
    # x = [x1, x2]
    a = g = 1.0
    x1_derivative = 2*g*x[1] - 2*a*(x[0]*x[0])
    x2_derivative = a*(x[0]*x[0]) - g*x[1]
    return [x1_derivative, x2_derivative]


# Алгоритм прямого моделирования реакции
def naive_algorithm(a, x0, reaction_coefficients_function, reaction_functions,  delta_t, t_max):
    # a - к-т реакции
    # reaction_coefficients_function - список функций для промежуточного вычисления вероятностей // aX, aX^2...
    # reaction_functions - список функций для пересчета количества вещества после реакции // X->X-1, X->X+2...
    x = list(x0)
    t = 0.0
    points = [(t, list(x))]
    while t < t_max:
        # p равномерная св из [0, 1]
        p = random.random()
        # curr_right_border - текущая правая граница для поиска интервала, в которое попала p
        curr_right_border = 0
        # reaction_coefficients - к-ты реакции при текущем количестве вещества
        reaction_coefficients = [f(a, x) for f in reaction_coefficients_function]
        # по всем к-там реакции
        for i in range(0, len(a)):
            # обновляем текущую правую границу
            curr_right_border += reaction_coefficients[i]*delta_t
            # если попали в интервал - вызываем нужную формулу для пересчета количества вещества
            if p <= curr_right_border:
                x = reaction_functions[i](x)
                break
        t += delta_t
        points.append((t, list(x)))
    return points


# Алгоритм моделирования реакции через пуассоновское распределение
def gillespie_algorithm(a, x0, reaction_coefficients_function, reaction_functions, t_max):
    # a - к-т реакции
    # reaction_coefficients_function - список функций для промежуточного вычисления вероятностей // aX, aX^2...
    # reaction_functions - список функций для пересчета количества вещества после реакции // X->X-1, X->X+2...
    x = list(x0)
    t = 0
    points = [(t, list(x))]
    while t < t_max:
        # p1 равномерная св из [0, 1]
        p1 = random.random()
        # reaction_coefficients - к-ты реакции при текущем количестве вещества
        reaction_coefficients = [f(a, x) for f in reaction_coefficients_function]
        # a0 - для нормировки, может быть 0 -> реакция прекрастилась, нет вещества?
        a0 = sum(reaction_coefficients)
        if a0 != 0:
            # tau - пуассоновская св. Показывает промежуток времени, через которое событие случилось
            tau = (1.0/a0)*math.log(1.0/p1)
            # p2 равномерная св из [0, 1]. По ней поймем какое именно событие случилось
            p2 = random.random()
            # curr_right_border - текущая правая граница для поиска интервала, в которое попала p2
            curr_right_border = 0
            for i in range(0, len(a)):
                # обновляем текущую правую границу
                curr_right_border += reaction_coefficients[i]/a0
                # если попали в интервал - вызываем нужную формулу для пересчета количества вещества
                if p2 <= curr_right_border:
                    x = reaction_functions[i](x)
                    break
            # тут t меняется не на шаг, а на время tau
            t += tau
        else:
            # если a0 = 0, то вещества нет, наверное реакция все
            t = t_max
        points.append((t, list(x)))
    return points


# Смоделировать процесс распада вещества
def task4():
    a = [1.0]
    x0 = [100.0]
    t_max = 20.0
    delta_t = 0.01
    reaction_coefficients_function = [lambda a, x: a[0] * x[0]] # aX
    reaction_functions = [lambda x: [x[0] - 1]]                 # X->X-1
    naive_algorithm_time = 0
    gillespie_algorithm_time = 0
    plt.clf()
    plt.xscale('log')
    for i in range(0, 3):
        start_time = time.process_time()
        points = naive_algorithm(a, x0, reaction_coefficients_function, reaction_functions, delta_t, t_max)
        naive_algorithm_time += time.process_time() - start_time
        x_naive_algorithm, y_naive_algorithm = zip(*points)

        start_time = time.process_time()
        points = gillespie_algorithm(a, x0, reaction_coefficients_function, reaction_functions, t_max)
        gillespie_algorithm_time += time.process_time() - start_time
        x_gillespie_algorithm, y_gillespie_algorithm = zip(*points)

        plt.plot(x_gillespie_algorithm, [elem[0] for elem in y_gillespie_algorithm], 'r', label='Алгоритм Gillespie')
        plt.plot(x_naive_algorithm, [elem[0] for elem in y_naive_algorithm], 'g', label='Наивный алгоритм')
    print("Average time of the naive algorithm: {0} s".format(naive_algorithm_time/3.0))
    print("Average time of the gillespie algorithm: {0} s".format(gillespie_algorithm_time / 3.0))
    points = []
    t = 0
    while t < t_max:
        points.append((t, test_system_solution(x0, t)[0]))
        t += delta_t
    x, y = zip(*points)
    plt.plot(x, y, '--', label='Аналитическое решение')
    save_img(plt, 'Решение системы c распадом вещества', r'$\ln(t)$', r'x')


# Рисует те же графики, но с одной реализацией
def task4_example():
    a = [1.0]
    x0 = [100.0]
    t_max = 5.0
    delta_t = 0.01
    reaction_coefficients_function = [lambda a, x: a[0]*x[0]]
    reaction_functions = [lambda x: [x[0] - 1]]

    points = naive_algorithm(a, x0, reaction_coefficients_function, reaction_functions, delta_t, t_max)
    x_naive_algorithm, y_naive_algorithm = zip(*points)
    points = gillespie_algorithm(a, x0, reaction_coefficients_function, reaction_functions, t_max)
    x_gillespie_algorithm, y_gillespie_algorithm = zip(*points)
    points = []
    t = 0
    while t < t_max:
        points.append((t, test_system_solution(x0, t)[0]))
        t += delta_t
    x, y = zip(*points)

    plt.clf()
    plt.plot(x_naive_algorithm, [elem[0] for elem in y_naive_algorithm], 'g', label='Наивный алгоритм')
    plt.plot(x_gillespie_algorithm, [elem[0] for elem in y_gillespie_algorithm], 'r', label='Алгоритм Gillespie')
    plt.plot(x, y, '--', label='Аналитическое решение')
    save_img(plt, 'Пример решения задачи с распадом двумя методами', r't', r'x')


# Смоделировать процесс синтеза и распада вещества
def task5():
    t_max = 0.25
    delta_t = 0.0001
    x0 = [100.0, 0.0]
    a = [1.0, -1.0]
    reaction_coefficients_function = [lambda a, x: a[0] * x[0] * x[0],  # a*(X1)^2,
                                      lambda a, x: -a[1] * x[1]]        # -g*(X2)
    reaction_functions = [lambda x: [x[0] - 2, x[1] + 1],               # X1->X1-2, X2->X2+1
                          lambda x: [x[0] + 2, x[1] - 1]]               # X1->X1+2, X2->X2-1

    points = runge_kutta_method(synthesis_decay_system, x0, delta_t, t_max)
    x_runge_kutta, y_runge_kutta = zip(*points)
    points = naive_algorithm(a, x0, reaction_coefficients_function, reaction_functions, delta_t, t_max)
    x_naive_algorithm, y_naive_algorithm = zip(*points)
    points = gillespie_algorithm(a, x0, reaction_coefficients_function, reaction_functions, t_max)
    x_gillespie_algorithm, y_gillespie_algorithm = zip(*points)

    plt.clf()
    plt.xscale('log')
    plt.plot(x_runge_kutta, [elem[0] for elem in y_runge_kutta], label='Метод Рунге-Кутта')
    plt.plot(x_naive_algorithm, [elem[0] for elem in y_naive_algorithm], 'g', label='Наивный алгоритм')
    plt.plot(x_gillespie_algorithm, [elem[0] for elem in y_gillespie_algorithm], 'r', label='Алгоритм Gillespie')
    save_img(plt, 'Количество атомов водорода', r'$\ln(t)$', r'x')

    plt.clf()
    plt.xscale('log')
    plt.plot(x_runge_kutta, [elem[1] for elem in y_runge_kutta], label='Метод Рунге-Кутта')
    plt.plot(x_naive_algorithm, [elem[1] for elem in y_naive_algorithm], 'g', label='Наивный алгоритм')
    plt.plot(x_gillespie_algorithm, [elem[1] for elem in y_gillespie_algorithm], 'r', label='Алгоритм Gillespie')
    save_img(plt, 'Количество молекул водорода', r'$\ln(t)$', r'$x_2$')

    '''
    x = np.array([elem for elem in x_gillespie_algorithm])
    y = np.array([elem[0] for elem in y_gillespie_algorithm])
    z = np.array([elem[1] for elem in y_gillespie_algorithm])
    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='Система c водородом')
    ax.legend()
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$x_2$')
    plt.show()
    '''


def main():
    random.seed()
    task1()
    task2_pendulum()
    task3()
    #task3_show_XYZ() # дополнительно
    task4()
    #task4_example() # дополнительно
    task5()

if __name__ == "__main__":
    main()