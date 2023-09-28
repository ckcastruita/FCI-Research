from quspin.operators import hamiltonian # operators
from quspin.basis import spinless_fermion_basis_general # spin basis constructor
import numpy as np # general math functions
import matplotlib.pyplot as plt
from scipy import linalg
import time

# define constants
pi = np.pi
e = np.e
k = complex(0,1)

# define model parameters
Lx = 6 # sites in x direction
Ly = 4 # sites in y direction
N_2d = Lx*Ly # number of sites for spin 1
t1 = np.sqrt(2) # NN hopping term
t2 = 1.0 # NNN hoppping term

n = np.arange(N_2d) # site indices
x = n%Lx # x positions
y = n//Lx # y positions
T_x = (x+1)%Lx+Lx*y # x translation
T_y = x+Lx*((y+1)%Ly) # y translation
s = [(-1)**(y[i]+i) for i in range(N_2d)] # sign of unit cell i
a = [e**(+s[i]*(pi/4)*k) for i in range(N_2d)] # + flux phase
b = [e**(-s[i]*(pi/4)*k) for i in range(N_2d)] # - flux phase

# setting up bases
basis_2d = spinless_fermion_basis_general(N_2d)

# define chern number calculation

def chern(num):
    # num represents the number of plaquettes in one direction
    chern_number = 0.0 + 0.0j
    
    # array for groundstate eigenvectors to be stored from each (i,j) point in parameter space
    array = [[] for _ in range(num + 1)]

    # here we iterate over the twisted parameter space from 0 to 2pi
    for i in range(0, num + 1):
        for j in range(0, num + 1):

            # twist phases at a given (i,j) in parameter space
            theta_x = 2*pi*i/num
            theta_y = 2*pi*j/num

            # smeared twist phase at a given (i,j) in parameter space
            px = theta_x/Lx*k
            py = theta_y/Ly*k
           
            # defining hoppings
            # nearest neighbor hoppings with pi flux and twist phase
            NN_N = [[t1*b[i]*e**(+py),i,T_y[i]] for i in range(N_2d)]
            NN_E = [[t1*a[i]*e**(+px),i,T_x[i]] for i in range(N_2d)]
            NN_S = [[t1*a[i]*e**(-py),T_y[i],i] for i in range(N_2d)]
            NN_W = [[t1*b[i]*e**(-px),T_x[i],i] for i in range(N_2d)]

            NN = NN_N + NN_E + NN_S + NN_W

            # next nearest neighbor hoppings with twist phase only
            NNN_NE = [[t2*s[i]*e**(px+py), i, T_y[T_x[i]]] for i in range(N_2d)]
            NNN_SE = [[t2*s[i]*e**(px-py), T_y[i], T_x[i]] for i in range(N_2d)]
            NNN_SW = [[t2*s[i]*e**(-px-py), T_y[T_x[i]], i] for i in range(N_2d)]
            NNN_NW = [[t2*s[i]*e**(-px+py), T_x[i], T_y[i]] for i in range(N_2d)]

            NNN = NNN_NE + NNN_SE + NNN_SW + NNN_NW

            static = [["+-", NN + NNN]]
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
