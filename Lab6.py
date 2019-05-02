import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def generate_random_hermitian_matrix(N, M):
    D = np.matrix(np.random.normal(0., 1., (N, M)) + 1j * np.random.normal(0., 1., (N, M)));
    A = (D + D.H) / 2.
    return A


def generate_statistics_for_eigenvalue(matrix_size, num_of_matrix):
    eigenvalues = []
    for i in range(0, num_of_matrix):
        A = generate_random_hermitian_matrix(matrix_size, matrix_size)
        eigenvalues.append(np.real(np.linalg.eigvalsh(A)))
    return eigenvalues


def calculate_distance_between_levels(data):
    all_distance = []
    for eigenvalues in data:
        distance = [eigenvalues[i + 1] - eigenvalues[i] for i in range(len(eigenvalues) - 1)]
        distance = np.array(distance) / np.mean(distance)
        all_distance.append(distance)
    all_distance = np.array(all_distance).reshape(len(data) * len(all_distance[0]))
    return all_distance


def save_statistics(all_eigenvalues, all_distance):
    path_to_folder = 'log'
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)

    file_name = os.path.join(path_to_folder, 'Lab6_eigenvalues_' + str(len(all_eigenvalues)))
    np.savetxt(file_name, all_eigenvalues, delimiter='\t')

    file_name = os.path.join(path_to_folder, 'Lab6_distance_' + str(len(all_eigenvalues)))
    np.savetxt(file_name, all_distance, delimiter='\t')


def load_statistics(size):
    path_to_folder = 'log'

    file_name = os.path.join(path_to_folder, 'Lab6_eigenvalues_' + str(size))
    all_eigenvalues = np.loadtxt(file_name, dtype=np.float, delimiter='\t')

    file_name = os.path.join(path_to_folder, 'Lab6_distance_' + str(size))
    all_distance = np.loadtxt(file_name, dtype=np.float, delimiter='\t')

    return all_eigenvalues, all_distance


def plot_histogram_of_eigenvalues(data, xlabel, ylabel, file_name):
    file_name_png = str(file_name) + '.png'
    path_to_folder = os.path.join('img', 'Lab6')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    file_name_png = os.path.join(path_to_folder, file_name_png)

    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    n, bins, patches = plt.hist(data, 100, density=True, label='Эмпирическое распределение',\
                                edgecolor='black', linewidth=0.5)

    c = np.max(n) / 2.
    x = np.linspace(-2., 2., 100)
    y = [c * np.sqrt(4. - x ** 2) for x in x]
    plt.plot(x, y, '', label='Теоретическое распределение')
    plt.legend(loc='lower right')
    #plt.grid(True)
    plt.savefig(file_name_png, dpi=300)
    plt.clf()


def plot_histogram_of_distance(data, xlabel, ylabel, file_name):
    file_name_png = str(file_name) + '.png'
    path_to_folder = os.path.join('img', 'Lab6')
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    file_name_png = os.path.join(path_to_folder, file_name_png)

    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    n, bins, patches = plt.hist(data, 100, density=True, label='Эмпирическое распределение',\
                                edgecolor='black', linewidth=0.5)

    WignerDyson = lambda x: x ** 2 * np.exp(-4. / np.pi * x ** 2)
    c = np.max(n) / WignerDyson(np.sqrt(np.pi) / 2.)
    x = np.linspace(np.min(bins), np.max(bins), 100)
    y = [c * WignerDyson(x) for x in x]
    plt.plot(x, y, '', label='Распределение Вигнера-Дайсона')
    plt.legend(loc='upper right')
    #plt.grid(True)
    plt.savefig(file_name_png, dpi=300)
    plt.clf()


def main():
    size = 1000
    num_of_matrix = 1000

    '''
    t1 = datetime.now()
    print(t1)
    eigenvalues = generate_statistics_for_eigenvalue(size, num_of_matrix)
    all_eigenvalues = np.array(eigenvalues).reshape(size * num_of_matrix)/np.sqrt(size)
    all_distance = calculate_distance_between_levels(eigenvalues)
    t2 = datetime.now()
    print(t2)
    print(t2 - t1)
    save_statistics(all_eigenvalues, all_distance)
    '''
    all_eigenvalues, all_distance = load_statistics(size*num_of_matrix)

    plot_histogram_of_eigenvalues(all_eigenvalues, r'$\lambda$', r'$W(\lambda)$', 'histogram_of_eigenvalues')
    plot_histogram_of_distance(all_distance, r'$\bar{s}$', r'$W(\bar{s})$', 'histogram_of_distance')


if __name__ == "__main__":
    main()
    exit()