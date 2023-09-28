from quspin.operators import hamiltonian # operators
from quspin.basis import spinless_fermion_basis_general # spin basis constructor
import numpy as np # general math functions
import matplotlib.pyplot as plt
from scipy import linalg
import time

pi = np.pi
e = np.e

def chern(num):
    # num represents the number of plaquettes in one direction
    chern_number = 0.0 + 0.0j

    array = [[] for _ in range(num + 1)]
    # we need num + 1 coordinates

    # here we iterate over the twisted parameter space from 0 to 2pi
    for i in range(0, num + 1):
        for j in range(0, num + 1):

            theta_x = 2 * pi * i / num
            theta_y = 2 * pi * j / num
            # define complex hopping parameters
            x1 = e ** (1.0j * theta_x)
            x2 = e ** (-1.0j * theta_x)
            y1 = e ** (1.0j * theta_y)
            y2 = e ** (-1.0j * theta_y)

            # define model parameters
            N_2d = 12  # number of sites for spin 1
            t = 1.0  # hopping matrix element
            t2 = 0.4 / (3 * np.sqrt(3)) * 1.0j  # complex hopping element
            M = 0.2  # onsite potential
            # setting up bases
            basis_2d = spinless_fermion_basis_general(N_2d)

            # setting up hamiltonian
            real_left = [[t, 0, 1], [t, 1, 2], [t, 2, 3], [t, 3, 4], [t, 4, 5], [t * x1, 5, 0], [t, 6, 7], [t, 7, 8],
                         [t, 8, 9], [t, 9, 10], [t, 10, 11], [t * x1, 11, 6], [t, 1, 7], [t, 3, 9], [t, 5, 11],
                         [t * y1, 6, 0], [t * y1, 8, 2], [t * y1, 10, 4]]
            real_right = [[t, 1, 0], [t, 2, 1], [t, 3, 2], [t, 4, 3], [t, 5, 4], [t * x2, 0, 5], [t, 7, 6], [t, 8, 7],
                          [t, 9, 8], [t, 10, 9], [t, 11, 10], [t * x2, 6, 11], [t, 7, 1], [t, 9, 3], [t, 11, 5],
                          [t * y2, 0, 6], [t * y2, 2, 8], [t * y2, 4, 10]]
            potential_a = [[M, 1], [M, 3], [M, 5], [M, 6], [M, 8], [M, 10]]
            potential_b = [[-M, 0], [-M, 2], [-M, 4], [-M, 7], [-M, 9], [-M, 11]]
            potential = potential_a + potential_b
            complex_left1 = [[t2 * x1, 5, 1], [t2, 1, 6], [t2 * x2, 6, 5], [t2, 0, 7], [t2 * x2, 7, 11],
                             [t2 * x1, 11, 0]]
            complex_right1 = [[-t2 * x2, 1, 5], [-t2, 6, 1], [-t2 * x1, 5, 6], [-t2, 7, 0], [-t2 * x1, 11, 7],
                              [-t2 * x2, 0, 11]]
            complex_left2 = [[t2, 1, 3], [t2, 3, 8], [t2, 8, 1], [t2, 2, 9], [t2, 9, 7], [t2, 7, 2]]
            complex_right2 = [[-t2, 3, 1], [-t2, 8, 3], [-t2, 1, 8], [-t2, 9, 2], [-t2, 7, 9], [-t2, 2, 7]]
            complex_left3 = [[t2, 3, 5], [t2, 5, 10], [t2, 10, 3], [t2, 4, 11], [t2, 11, 9], [t2, 9, 4]]
            complex_right3 = [[-t2, 5, 3], [-t2, 10, 5], [-t2, 3, 10], [-t2, 11, 4], [-t2, 9, 11], [-t2, 4, 9]]
            complex_left4 = [[t2, 6, 8], [t2 * y1, 8, 1], [t2 * y2, 1, 6], [t2 * y1, 7, 2], [t2, 2, 0], [t2 * y2, 0, 7]]
            complex_right4 = [[-t2, 8, 6], [-t2 * y2, 1, 8], [-t2 * y1, 6, 1], [-t2 * y2, 2, 7], [-t2, 0, 2],
                              [-t2 * y1, 7, 0]]
            complex_left5 = [[t2, 8, 10], [t2 * y1, 10, 3], [t2 * y2, 3, 8], [t2 * y1, 9, 4], [t2, 4, 2],
                             [t2 * y2, 2, 9]]
            complex_right5 = [[-t2, 10, 8], [-t2 * y2, 3, 10], [-t2 * y1, 8, 3], [-t2 * y2, 4, 9], [-t2, 2, 4],
                              [-t2 * y1, 9, 2]]
            complex_left6 = [[t2 * x1, 10, 6], [t2 * y1 * x2, 6, 5], [t2 * y2, 5, 10], [t2 * x1 * y1, 11, 0],
                             [t2 * x2, 0, 4], [t2 * y2, 4, 11]]
            complex_right6 = [[-t2 * x2, 6, 10], [-t2 * y2 * x1, 5, 6], [-t2 * y1, 10, 5], [-t2 * x2 * y2, 0, 11],
                              [-t2 * x1, 4, 0], [-t2 * y1, 11, 4]]
            hopping_left = real_left + complex_left1 + complex_left2 + complex_left3 + complex_left4 + complex_left5 + complex_left6
            hopping_right = real_right + complex_right1 + complex_right2 + complex_right3 + complex_right4 + complex_right4 + complex_right5 + complex_right6

            static = [["+-", hopping_left], ["+-", hopping_right], ["n", potential]]
            # build hamiltonian
            H = hamiltonian(static, [], basis=basis_2d, dtype=np.complex128)
            # diagonalise H

            E1, V = H.eigsh(k=1, which='SA')

            array[i].append(np.ravel(V))

    for i in range(0, num):
        for j in range(0, num):
            U1 = np.conjugate(array[i][j]).dot(array[i][j + 1]) / linalg.norm(
                np.conjugate(array[i][j]).dot(array[i][j + 1]))
            U2 = np.conjugate(array[i][j + 1]).dot(array[i + 1][j + 1]) / linalg.norm(
                np.conjugate(array[i][j + 1]).dot(array[i + 1][j + 1]))
            U3 = np.conjugate(array[i + 1][j + 1]).dot(array[i + 1][j]) / linalg.norm(
                np.conjugate(array[i + 1][j + 1]).dot(array[i + 1][j]))
            U4 = np.conjugate(array[i + 1][j]).dot(array[i][j]) / linalg.norm(
                np.conjugate(array[i + 1][j]).dot(array[i][j]))
            F1 = np.log(U1 * U2 * U3 * U4)
            chern_number += F1

    chern_number /= (2 * pi * 1j)
    print("Chern number = ", round(np.real(chern_number), 2))

chern(3)
