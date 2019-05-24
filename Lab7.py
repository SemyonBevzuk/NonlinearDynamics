import os
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


def generate_random_hermitian_matrix(N, M):
    D = np.matrix(np.random.normal(0., 1., (N, M)) + 1j * np.random.normal(0., 1., (N, M)))
    A = (D + D.H) / 2.
    return A


def runge_kutta_method(f, f_0, a, b, step):
    x = f_0
    t = a
    numSteps = int((b - a) / step)
    for i in range(0, numSteps):
        k1 = f(x, t)
        k2 = f(x + 0.5*step * k1, t + 0.5 * step)
        k3 = f(x + 0.5 * step * k2, t + 0.5 * step)
        k4 = f(x + step*k3, t + step)
        x = x + step * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        t = t + step

    return x


def get_propagator(f, T, size):
    step = 0.01
    P = [[]]*size
    for i in range(size):
        psi_0 = np.zeros((size, 1))
        psi_0[i] = 1.
        P[i] = runge_kutta_method(f, psi_0, 0., T, step)

    return np.matrix(np.array(P).reshape(size, size)).T


def get_eigenvalues(size, A):
    H_0 = generate_random_hermitian_matrix(size, size)
    f = lambda x, t: H_0 * (1.0 + A * np.cos(t)) * x * 1j
    T = 2 * np.pi
    Propagator_T = get_propagator(f, T, size)
    eigenvalues, eigenvectors = np.linalg.eig(Propagator_T)
    return eigenvalues


def plot_eigenvalues(eigenvalues, A):
    file_name_png = 'eigenvalues_A=' + str(A) + '.png'
    path_to_folder = os.path.join('img', 'Lab7')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    file_name_png = os.path.join(path_to_folder, file_name_png)

    plt.clf()
    plt.axis('equal')
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticklabels([0, -1.0, -0.75, -0.5, -0.25, '', 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels([0, -1.0,  -0.5, '', 0.50, 1.0])

    circle = plt.Circle((0, 0), 1.0, color='black', fill=False)
    plt.gca().add_patch(circle)
    x = [eigenvalue.real for eigenvalue in eigenvalues]
    y = [eigenvalue.imag for eigenvalue in eigenvalues]
    plt.scatter(x, y, c='r', s=15, alpha=None)

    plt.savefig(file_name_png, dpi=300)
    plt.clf()


def plot_error_eigenvalues(eigenvalues, A):
    file_name_png = 'error_eigenvalues_A=' + str(A) + '.png'
    path_to_folder = os.path.join('img', 'Lab7')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    file_name_png = os.path.join(path_to_folder, file_name_png)

    plt.clf()
    matplotlib.rcParams["axes.formatter.limits"] = (-3, 2)

    x = [i for i in range(0, len(eigenvalues))]
    y = [abs(eigenvalue) - 1.0 for eigenvalue in eigenvalues]
    plt.plot(x, y, marker="o")
    plt.xlabel(r'$k$')
    plt.ylabel(r'$|\lambda_k|$ - 1')

    plt.savefig(file_name_png, dpi=300)
    plt.clf()



def main():
    size = 10
    A = 0
    for A in [0.0, 0.1, 1.0]:
        eigenvalues = get_eigenvalues(size, A)
        plot_eigenvalues(eigenvalues, A)
        plot_error_eigenvalues(eigenvalues, A)

if __name__ == "__main__":
    main()
