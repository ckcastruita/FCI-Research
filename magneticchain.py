from quspin.operators import hamiltonian # operators
from quspin.basis import spinless_fermion_basis_general # spin basis constructor
import numpy as np # general math functions
import matplotlib.pyplot as plt
#
pi = np.pi
e = np.e
i = complex(0,1)
theta = pi/2
#
###### define model parameters ######
Lx, Ly = 3, 1 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites for spin 1
#
J=1.0 # hopping matrix element
U=2.0 # onsite interaction
mu=0.5 # chemical potential
#
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
#
###### setting up bases ######
basis_2d=spinless_fermion_basis_general(N_2d)
#
###### setting up hamiltonian ######
# setting up site-coupling lists
hopping_left=[[-J,i,T_x[i]] for i in range(N_2d - 1)] + [[-J*e**(-i*theta), N_2d - 1, 0]]
hopping_right=[[-J,T_x[i],i] for i in range(N_2d - 1)] + [[-J*e**(i*theta), 0, N_2d -1]]
potential=[[-mu,i] for i in range(N_2d)]
interaction=[[U,i,T_x[i]] for i in range(N_2d)]
#
static=[["+-",hopping_left],["+-",hopping_right],["n",potential],["nn",interaction]]
# build hamiltonian
H=hamiltonian(static,[],basis=basis_2d,dtype=np.complex128)
# diagonalise H
E=H.eigvalsh()
#
def format_complex(z, threshold=1e-4):
    re, im = z.real, z.imag

    # Suppress small real or imaginary parts
    if abs(re) < threshold:
        re = 0.0
    if abs(im) < threshold:
        im = 0.0

    return complex(re, im)
#
E1, V = H.eigsh(k=1, which='SA')
#
# print ground state energy and wavefunction
print("Ground state energy and wavefunction")
print(E1, [format_complex(v[0]) for v in V])
#
# print translation operators
print("Translation operator in x direciton =")
print(T_x)
#
# print hoppings
print("Hopping left =")
print(hopping_left)
print("Hopping right =")
print(hopping_right)

# print static
print("static =")
print(static)

# print Hamiltonian
print(H)

# convert Hamiltonian to matrix form
dense_H = H.toarray()
print("Matrix Hamiltonian =")
print(np.round(dense_H,decimals = 3))
# print diagonalized hamiltonian
print("Diagonalized Hamiltonian =")
print(np.round(E, decimals=5))
