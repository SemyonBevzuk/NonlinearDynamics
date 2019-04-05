import os
import matplotlib.pyplot as plt
import numpy as np


def logistic_map(r, x):
    return r * x * (1 - x)


def logistic_map_derivative(r, x):
    return r - 2 * r * x


def save_bifurcation_diagram(fig, axis, r_left, r_right):
    plt.figure(fig.number)
    axis.set_xlim(r_left, r_right)
    axis.set_title("Бифуркационная диаграмма")
    axis.set_xlabel(r'$r$')
    axis.set_ylabel(r'$x^*$')
    plt.tight_layout()

    file_name = 'Bifurcation_diagram_from_' + str(r_left) + '_to_' + str(r_right)
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab5')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=500)


def save_bifurcation_diagram_with_lyapunov_indicator(fig, axis_bif_diagram, axis_lyapunov_indicator,
                                                     r, L, r_left, r_right):
    plt.figure(fig.number)

    axis_bif_diagram.set_xlim(r_left, r_right)
    axis_bif_diagram.set_title("Бифуркационная диаграмма")
    axis_bif_diagram.set_xlabel(r'$r$')
    axis_bif_diagram.set_ylabel(r'$x^*$')

    axis_lyapunov_indicator.axhline(0, color='k', lw=.5)
    axis_lyapunov_indicator.plot(r[L < 0], L[L < 0], '.g', ms=.25)
    axis_lyapunov_indicator.plot(r[L >= 0], L[L >= 0], '.r', ms=.25)
    axis_lyapunov_indicator.set_xlim(r_left, r_right)
    axis_lyapunov_indicator.set_ylim(-1, 1)
    axis_lyapunov_indicator.set_title("Ляпуновский показатель")
    axis_lyapunov_indicator.set_xlabel(r'$r$')
    axis_lyapunov_indicator.set_ylabel(r'$\widetilde{L}$')
    plt.tight_layout()

    file_name = 'Bifurcation_diagram_with_lyapunov_indicator_from_' + str(r_left) + '_to_' + str(r_right)
    file_name_png = file_name + '.png'
    path_to_folder = os.path.join('img', 'Lab5')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    plt.savefig(os.path.join(path_to_folder, file_name_png), dpi=500)


# n -- число траекторий с разным r
# time_max -- длительность процесса
def plot_bifurcation_diagram_with_lyapunov_indicator(r_left, r_right, n, time_max):
    x = 1e-5 * np.ones(n)
    r = np.linspace(r_left, r_right, n)
    L = np.ones(n, dtype=np.float128)  # ляпуновкие показатели
    last_steps = 100  # определяет момент, когда процесс сошёлся к предельным точкам
    fig_bifurcation_diagram, axis_bif_diagram_1 = plt.subplots(1, 1)
    fig_bifurcation_diagram_with_lyapunov_indicator, (axis_bif_diagram_2, axis_lyapunov_indicator) \
        = plt.subplots(2, 1, figsize=(9, 7))
    count = 0
    for time in range(time_max):
        x = logistic_map(r, x)
        L *= np.abs(logistic_map_derivative(r, x))
        # если переходные процессы прошли и мы у предельного значения
        if time >= (time_max - last_steps):
            axis_bif_diagram_1.plot(r, x, ',k', alpha=.5)
            axis_bif_diagram_2.plot(r, x, ',k', alpha=.5)
        if count > time_max/10:
            count = 0
            print('.')
        count += 1

    L = np.log(L)/time_max

    save_bifurcation_diagram(fig_bifurcation_diagram, axis_bif_diagram_1, r_left, r_right)
    save_bifurcation_diagram_with_lyapunov_indicator(fig_bifurcation_diagram_with_lyapunov_indicator,
                                                     axis_bif_diagram_2, axis_lyapunov_indicator,
                                                     r, L, r_left, r_right)
    plt.close('all')


def main():
    #plot_x_n(0.01, 4.2, 8)

    n = int(1e5)  # число траекторий с разным r
    time_max = int(1e4)  # длительность процесса
    r_left = 3.80
    r_right = 4.0
    #for r_left in [3.0, 3.45, 3.54, 3.57, 3.80]:
    #    print('\nr_left = {}'.format(r_left))
    plot_bifurcation_diagram_with_lyapunov_indicator(r_left, r_right, n, time_max)


if __name__ == "__main__":
    main()