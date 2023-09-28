from quspin.operators import hamiltonian # operators
from quspin.basis import spinless_fermion_basis_general # spin basis constructor
import numpy as np # general math functions
import matplotlib.pyplot as plt
#
###### define model parameters ######
Lx, Ly = 3, 3 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites for spin 1
#
J=1.0 # hopping matrix element
U=0 # onsite interaction
mu=1.0 # chemical potential
#
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x + Lx*((y+1)%Ly) # translation along y-direction
P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
#
###### setting up bases ######
basis_2d=spinless_fermion_basis_general(N_2d)
#
###### setting up hamiltonian ######
# setting up site-coupling lists
hopping_left=[[-J,i,T_x[i]] for i in range(N_2d)] + [[-J,i,T_y[i]] for i in range(N_2d)]
hopping_right=[[+J,i,T_x[i]] for i in range(N_2d)] + [[+J,i,T_y[i]] for i in range(N_2d)]
potential=[[-mu,i] for i in range(N_2d)]
#interaction=[[U,i,T_x[i]] for i in range(N_2d)] + [[U,i,T_y[i]] for i in range(N_2d)]
#
static=[["+-",hopping_left],["-+",hopping_right],["n",potential]]
# build hamiltonian
H=hamiltonian(static,[],basis=basis_2d,dtype=np.float64)
# diagonalise H
E=H.eigvalsh()
#
# print site list
print(s)
#
print("potential list")
print(potential)
# print translation operators
print("Translation operator in x direciton =")
print(T_x)
print("Translation operator in y direction =")
print(T_y)

print("T_x part of the left hopping =")
print([[-J,i,T_x[i]] for i in range(N_2d)])
print("T_y part of the left hopping =")
print([[-J,i,T_y[i]] for i in range(N_2d)])
print("T_x part of the right hopping =")
print([[+J,i,T_x[i]] for i in range(N_2d)])
print("T_y part of the right hopping =")
print([[+J,i,T_y[i]] for i in range(N_2d)])

# print hoppings
print("Hopping left =")
print(hopping_left)
print("Hopping right =")
print(hopping_right)

# print static
print("static =")
print(static)

# print Hamiltonian in sparse form 
print("H =")
print(H)

# convert Hamiltonian to matrix form
dense_H = H.toarray()
print("Matrix Hamiltonian =")
print(dense_H)
# print diagonalized hamiltonian
print("Diagonalized Hamiltonian =")
print(np.round(E))

# plot eigenvalues
#plt.plot(range(len(E)), E, 'bo')
#plt.xlabel('Eigenstate index')
#plt.ylabel('Eigenvalue')
#plt.title('Eigenvalue Spectrum')
#plt.show()
