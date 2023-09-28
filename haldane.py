from quspin.operators import hamiltonian # operators
from quspin.basis import spinless_fermion_basis_general # spin basis constructor
import numpy as np # general math functions
import matplotlib.pyplot as plt

###### define model parameters ######
N_2d = 12 # number of sites for spin 1
J=1.0 # hopping matrix element
t2=1.0j # complex hopping element
U=0 # onsite interaction
mu=0 # chemical potential

###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
T_x = np.array([2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 6, 7]) # translation along x-direction
T_y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) # translation along y-directions

###### setting up bases ######
basis_2d = spinless_fermion_basis_general(N_2d)

###### setting up hamiltonian ######

real_left = [[-J, 0, 1], [-J, 1, 2], [-J, 2, 3], [-J, 3, 4], [-J, 4, 5], [-J, 5, 0], [-J, 6, 7], [-J, 7, 8], [-J, 8, 9], [-J, 9, 10], [-J, 10, 11], [-J, 11, 6], [-J, 1, 7], [-J, 3, 9], [-J, 5, 11],
[-J, 6, 0], [-J, 8, 2], [-J, 10, 4]]
real_right = [[-J, 1, 0], [-J, 2, 1], [-J, 3, 2], [-J, 4, 3], [-J, 5, 4], [-J, 0, 5], [-J, 7, 6], [-J, 8, 7], [-J, 9, 8], [-J, 10, 9], [-J, 11, 10], [-J, 6, 11], [-J, 7, 1], [-J, 9, 3], [-J, 11, 5],
[-J, 0, 6], [-J, 2, 8], [-J, 4, 10]]
complex_left1 = [[t2, 5, 1], [t2, 1, 6], [t2, 6, 5], [t2, 0, 7], [t2, 7, 11], [t2, 11, 0]]
complex_right1 = [[-t2, 1,5], [-t2, 6, 1], [-t2, 5, 6], [-t2, 7, 0], [-t2, 11, 7], [-t2, 0, 11]]
complex_left2 = [[t2, 1, 3], [t2, 3, 8], [t2, 8, 1], [t2, 2, 9], [t2, 9, 7], [t2, 7, 2]]
complex_right2 = [[-t2, 3, 1], [-t2, 8, 3], [-t2, 1, 8], [-t2, 9, 2], [-t2, 7 ,9], [-t2, 2, 7]]
complex_left3 = [[t2, 3, 5], [t2, 5, 10], [t2, 10, 3], [t2, 4, 11], [t2, 11, 9], [t2, 9, 4]]
complex_right3 = [[-t2, 5, 3], [-t2, 10, 5], [-t2, 3, 10], [-t2, 11, 4], [-t2, 9, 11], [-t2, 4, 9]]
complex_left4 = [[t2, 6, 8], [t2, 8, 1], [t2, 1, 6], [t2, 7, 2], [t2, 2, 0], [t2, 0 ,7]]
complex_right4 = [[-t2, 8, 6], [-t2, 1, 8], [-t2, 6, 1], [-t2, 2, 7], [-t2, 0, 2], [-t2, 7, 0]]
complex_left5 = [[t2, 8, 10], [t2, 10, 3], [t2, 3, 8], [t2, 9, 4], [t2, 4, 2], [t2, 2, 9]]
complex_right5 = [[-t2, 10, 8], [-t2, 3, 10], [-t2, 8, 3], [-t2, 4, 9], [-t2, 2, 4], [-t2, 9, 2]]
complex_left6 = [[t2, 10, 6], [t2, 6, 5], [t2, 5, 10], [t2, 11, 0], [t2, 0, 4], [t2, 4, 11]]
complex_right6 = [[-t2, 6, 10], [-t2, 5, 6], [-t2, 10, 5], [-t2, 0, 11], [-t2, 4, 0], [-t2, 11, 4]]
hopping_left = real_left + complex_left1 + complex_left2 + complex_left3 + complex_left4 + complex_left5 + complex_left6
hopping_right = real_right + complex_right1 + complex_right2 + complex_right3 + complex_right4 + complex_right4 + complex_right5 + complex_right6

static=[["+-",hopping_left],["+-",hopping_right]]
# build hamiltonian
H=hamiltonian(static,[],basis=basis_2d,dtype=np.complex128)
# diagonalise H
E=H.eigvalsh()


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
E1, V = H.eigsh(k=1, which='SA')
#

# print diagonalized hamiltonian
print("Diagonalized Hamiltonian =")
print(np.round(E, decimals=3))

# plot eigenvalues
#plt.plot(range(len(E)), E, 'bo')
#plt.xlabel('Eigenstate index')
#plt.ylabel('Eigenvalue')
#plt.title('Eigenvalue Spectrum')
#plt.show()
