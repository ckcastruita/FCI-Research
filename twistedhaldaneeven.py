from quspin.operators import hamiltonian # operators
from quspin.basis import spinless_fermion_basis_general # spin basis constructor
import numpy as np # general math functions
import matplotlib.pyplot as plt
from scipy import linalg
import time

# define pi and e for simplicity
pi = np.pi
e = np.e

# define chern number calculation
def chern(num):
    # num represents the number of plaquettes in each direction
    chern_number = 0.0 + 0.0j

    array = [[] for _ in range(num + 1)]
    # we need num + 1 coordinates

    # here we iterate over the twisted parameter space from 0 to 2pi
    for i in range(0, num + 1):
        for j in range(0, num + 1):

            theta_x = 2 * pi * i / num
            theta_y = 2 * pi * j / num

            # define complex hopping parameters
            p1 = e ** (1.0j * theta_y / 3)
            p2 = e ** (1.0j * (theta_x + theta_y) / 6)
            p3 = e ** (1.0j * (theta_x - theta_y) / 6)
            q1 = e ** (-1.0j * theta_y / 3)
            q2 = e ** (-1.0j * (theta_x + theta_y) / 6)
            q3 = e ** (-1.0j * (theta_x - theta_y) / 6)
            r1 = e ** (1.0j * theta_x / 3)
            s1 = e ** (-1.0j * theta_x / 3)
            r2 = e ** (1.0j * (theta_x / 6 + theta_y / 2))
            s2 = e ** (-1.0j * (theta_x / 6 + theta_y / 2))
            r3 = e ** (1.0j * (theta_x / 6 - theta_y / 2))
            s3 = e ** (-1.0j * (theta_x / 6 - theta_y / 2))

            # define model parameters
            N_2d = 12  # number of sites for spin 1
            t1 = 1.0  # hopping matrix element
            t2 = 0.4 / (3 * np.sqrt(3)) * 1.0j  # complex hopping element
            M = 0.2  # onsite potential
            # setting up bases
            basis_2d = spinless_fermion_basis_general(N_2d)

            # setting up hamiltonian
            real_left = [[t1 * p2, 0, 1], [t1 * p3, 1, 2], [t1 * p2, 2, 3], [t1 * p3, 3, 4], [t1 * p2, 4, 5], [t1 * p3, 5, 0], [t1 * p3, 6, 7], [t1 * p2, 7, 8],
                         [t1 * p3, 8, 9], [t1 * p2, 9, 10], [t1 * p3, 10, 11], [t1 * p2, 11, 6], [t1 * p1, 1, 7], [t1 * p1, 3, 9], [t1 * p1, 5, 11],
                         [t1 * p1, 6, 0], [t1 * p1, 8, 2], [t1 * p1, 10, 4]]
            real_right = [[t1 * q2, 1, 0], [t1 * q3, 2, 1], [t1 * q2, 3, 2], [t1 * q3, 4, 3], [t1 * q2, 5, 4], [t1 * q3, 0, 5], [t1 * q3, 7, 6], [t1 * q2, 8, 7],
                          [t1 * q3, 9, 8], [t1 * q2, 10, 9], [t1 * q3, 11, 10], [t1 * q2, 6, 11], [t1 * q1, 7, 1], [t1 * q1, 9, 3], [t1 * q1, 11, 5],
                          [t1 * q1, 0, 6], [t1 * q1, 2, 8], [t1 * q1, 4, 10]]

            potential_a = [[M, 1], [M, 3], [M, 5], [M, 6], [M, 8], [M, 10]]
            potential_b = [[-M, 0], [-M, 2], [-M, 4], [-M, 7], [-M, 9], [-M, 11]]
            potential = potential_a + potential_b

            complex_left1 = [[t2 * r1, 5, 1], [t2 * s3, 1, 6], [t2 * s2, 6, 5], [t2 * r2, 0, 7], [t2 * s1, 7, 11],
                             [t2 * r3, 11, 0]]
            complex_right1 = [[-t2 * s1, 1, 5], [-t2 * r3, 6, 1], [-t2 * r2, 5, 6], [-t2 * s2, 7, 0], [-t2 * r1, 11, 7],
                              [-t2 * s3, 0, 11]]

            complex_left2 = [[t2 * r1, 1, 3], [t2 * s3, 3, 8], [t2 * s2, 8, 1], [t2 * r2, 2, 9], [t2 * s1, 9, 7], [t2 * r3, 7, 2]]
            complex_right2 = [[-t2 * s1, 3, 1], [-t2 * r3, 8, 3], [-t2 * r2, 1, 8], [-t2 * s2, 9, 2], [-t2 * r1, 7, 9], [-t2 * s3, 2, 7]]
           
            complex_left3 = [[t2 * r1, 3, 5], [t2 * s3, 5, 10], [t2 * s2, 10, 3], [t2 * r2, 4, 11], [t2 * s1, 11, 9], [t2 * r3, 9, 4]]
            complex_right3 = [[-t2 * s1, 5, 3], [-t2 * r3, 10, 5], [-t2 * r2, 3, 10], [-t2 * s2, 11, 4], [-t2 * r1, 9, 11], [-t2 * s3, 4, 9]]

            complex_left4 = [[t2 * r1, 6, 8], [t2 * s3, 8, 1], [t2 * s2, 1, 6], [t2 * r2, 7, 2], [t2 * s1, 2, 0], [t2 * r3, 0, 7]]
            complex_right4 = [[-t2 * s1, 8, 6], [-t2 * r3, 1, 8], [-t2 * r2, 6, 1], [-t2 * s2, 2, 7], [-t2 * r1, 0, 2],
                              [-t2 * s3, 7, 0]]

            complex_left5 = [[t2 * r1, 8, 10], [t2 * s3, 10, 3], [t2 * s2, 3, 8], [t2 * r2, 9, 4], [t2 * s1, 4, 2],
                             [t2 * r3, 2, 9]]
            complex_right5 = [[-t2 * s1, 10, 8], [-t2 * r3, 3, 10], [-t2 * r2, 8, 3], [-t2 * s2, 4, 9], [-t2 * r1, 2, 4],
                              [-t2 * s3, 9, 2]]
           
            complex_left6 = [[t2 * r1, 10, 6], [t2 * s3, 6, 5], [t2 * s2, 5, 10], [t2 * r2, 11, 0],
                             [t2 * s1, 0, 4], [t2 * r3, 4, 11]]
            complex_right6 = [[-t2 * s1, 6, 10], [-t2 * r3, 5, 6], [-t2 * r2, 10, 5], [-t2 * s2, 0, 11],
                              [-t2 * r1, 4, 0], [-t2 * s3, 11, 4]]
           
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
