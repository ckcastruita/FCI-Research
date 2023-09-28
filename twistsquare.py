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
            p1 = e ** (1.0j * pi / 4)
            p2 = e ** (-1.0j * pi /4)

            # define model parameters
            N_2d = 16 # number of sites for spin 1
            t1 = np.sqrt(2) # NN hopping term
            t2 = 1.0 # NNN hoppping term

            # setting up bases
            basis_2d = spinless_fermion_basis_general(N_2d)

            # setting up hamiltonian
            NN_hoppings_left = [[t1 * p1, 0, 1], [t1 * p2, 1, 2], [t1 * p1, 2, 3], [t1 * p2 * x1, 3, 0], [t1 * p2, 0, 4], [t1 * p1, 1, 5], [t1 * p2, 2, 6], [t1 * p1, 3, 7], [t1 * p2, 4, 5],
                                [t1 * p1, 5, 6], [t1 * p2, 6, 7], [t1 * p1 * x1, 7, 4], [t1 * p1, 4, 8], [t1 * p2, 5, 9], [t1 * p1, 6, 10], [t1 * p2, 7, 11], [t1 * p1, 8, 9], [t1 * p2, 9, 10],
                                [t1 * p1, 10, 11], [t1 * p2 * x1, 11, 8], [t1 * p2, 8, 12], [t1 * p1, 9 ,13], [t1 * p2, 10, 14], [t1 * p1, 11, 15], [t1 * p2, 12, 14], [t1 * p1, 13, 14], [t1 * p2, 14, 15],                                [t1 * p1 * x1, 15, 12], [t1 * p1 * y1, 12, 0], [t1 * p2 * y1, 13, 1], [t1 * p1 * y1, 14, 2], [t1 * p2 * y1, 15, 3]]
            NN_hoppings_right = [[t1 * p2, 1, 0], [t1 * p1, 2, 1], [t1 * p2, 3, 2], [t1 * p1 * x2, 0, 3], [t1 * p1, 4, 0], [t1 * p2, 5, 1], [t1 * p1, 6, 2], [t1 * p2, 7, 3], [t1 * p1, 5, 4],
                                 [t1 * p2, 6, 5], [t1 * p1, 7, 6], [t1 * p2 * x2, 4, 7], [t1 * p2, 8, 4], [t1 * p1, 9, 5], [t1 * p2, 10, 6], [t1 * p1, 11, 7], [t1 * p2, 9, 8], [t1 * p1, 10, 9],
                                 [t1 * p2, 11, 10], [t1 * p1 * x2, 8, 11], [t1 * p1, 12, 8], [t1 * p2, 13, 9], [t1 * p1, 14, 10], [t1 * p2, 15, 11], [t1 * p1, 14, 12], [t1 * p2, 14, 13],
                                 [t1 * p1, 15, 14], [t1 * p2 * x2, 12, 15], [t1 * p2 * y2, 0, 12], [t1 * p1 * y2, 1, 13], [t1 * p2 * y2, 2, 14], [t1 * p1 * y2, 3, 15]]
            NNN_hoppings_left = [[t2, 0, 5], [t2, 1, 4], [-t2, 1, 6], [-t2, 2, 5], [t2, 2, 7], [t2, 3, 6], [-t2 * x1, 3, 4], [-t2 * x1, 7, 0], [-t2, 4, 9], [-t2, 5, 8], [t1, 5, 10], [t1, 6, 9],
                                 [-t2, 6, 11], [-t2, 7, 10], [t2 * x1, 7, 8], [t2 * x1, 11, 4], [t2, 8, 13], [t2, 9, 12], [-t2, 9, 14], [-t2, 10, 13], [t1, 10 ,15], [t1, 11, 14], [-t2 * x1, 11, 12],
                                 [-t2 * x1, 15, 8], [-t2 * y1, 12, 1], [-t2 * y1, 13, 0], [t2 * y1, 13, 2], [t2 * y1, 14, 1], [-t2 * y1, 14, 3], [-t2 * y1, 15, 2], [t2 * x1 * y1, 15, 0],
                                 [t1 * x1 * y2, 3, 12]]
            NNN_hoppings_right = [[t2, 5, 0], [t2, 4, 1], [-t2, 6, 1], [-t2, 5, 2], [t2, 7, 2], [t2, 6, 3], [-t2 * x2, 4, 3], [-t2 * x2, 0, 7], [-t2, 9, 4], [-t2, 8, 5], [t1, 10, 5], [t1, 9, 6],
                                 [-t2, 11, 6], [-t2, 10, 7], [t2 * x2, 8, 7], [t2 * x2, 4, 11], [t2, 13, 8], [t2, 12, 9], [-t2, 14, 9], [-t2, 13, 10], [t1, 15, 10], [t1, 14, 11], [-t2 * x2, 12, 11],
                                 [-t2 * x2, 8, 15], [-t2 * y2, 1, 12], [-t2 * y2, 0, 13], [t2 * y2, 2, 13], [t2 * y2, 1, 14], [-t2 * y2, 3, 14], [-t2 * y2, 2, 15], [t2 * x2 * y2, 0, 15],
                                 [t1 * x2 * y1, 12, 3]]

            hopping_left = NN_hoppings_left + NNN_hoppings_left
            hopping_right = NN_hoppings_right + NNN_hoppings_right

            static = [["+-", hopping_left], ["+-", hopping_right]]
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
