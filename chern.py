import math
import cmath
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import linalg

pi = math.pi

def main():

	eigenvalue(50, hamil)
	chern(3, hamil)

def hamil(kx, ky):

	# define Pauli matrices
	x = np.array([[0, 1], [1, 0]])
	y = np.array([[0, -complex(0, 1)], [complex(0, 1), 0]])
	z = np.array([[1, 0], [0, -1]])
	
	# define lattice vectors

	a1 = np.array([1, 0])
	a2 = np.array([-1/2, math.sqrt(3)/2])
	a3 = np.array([-1/2, -math.sqrt(3)/2])

	b1 = a2 - a3
	b2 = a3 - a1
	b3 = a1 - a2	

	# define model parameters
	t1 = 1.0
	m = 0
	t2 = 0.0/(3 * math.sqrt(3))
	k = np.array([kx, ky])

	# define Hamiltonian
	h0 = t1 * x * (math.cos(np.dot(a1, k)) + math.cos(np.dot(a2, k)) + math.cos(np.dot(a3, k))) - t1 * y * (math.sin(np.dot(a1, k)) + math.sin(np.dot(a2, k)) + math.sin(np.dot(a3, k)))
	h = h0 + z * (m + 2 * t2 * (math.sin(np.dot(b1, k)) + math.sin(np.dot(b2, k)) + math.sin(np.dot(b3, k))))

	return h

def eigenvalue(num, h):
	# num: k-space resolution
	# Hamiltonian as dense matrix
	kx_array = np.linspace(- 4 * pi / (3 * math.sqrt(3)), 4 * pi / (3 * math.sqrt(3)), num)
	ky_array = np.linspace(- 4 * pi / (3 * math.sqrt(3)), 4 * pi / (3 * math.sqrt(3)), num)
	x, y = np.meshgrid(kx_array, ky_array)
	n = np.shape(h(0, 0))[0]  # number of bands
	e = np.zeros([num, num, n])
	for ix in range(num):
        	for iy in range(num):
	        	e_value, e_vector = np.linalg.eig(h(kx_array[ix], ky_array[iy]))
        		e[ix, iy, :] = np.real(e_value)[np.argsort(e_value)]

	fig = plt.figure()  # Begin figure
	ax = fig.add_subplot(111, projection='3d')

	for i in range(n):
        	ax.plot_surface(x, y, np.real(e[:, :, i]), cmap=cm.coolwarm, linewidth=0, antialiased=False)  # plot all bands

	ax.set_xlabel(r'$k_x$')
	ax.set_ylabel(r'$k_y$')
	ax.set_zlabel(r'$E(k)$')
	plt.savefig('Chern.png', dpi=400)
	plt.show()

	return e

def chern(num, h):
    n = np.shape(h(0, 0))[0]
    chern_number = np.zeros(n, dtype=np.complex128)
    v1 = np.array([2*pi/3, -2*pi/np.sqrt(3)])
    v2 = np.array([2*pi/3, 2*pi/np.sqrt(3)])
    d = 1 / num
    for E in range(2):
        for i in np.arange(0, 1, d):
            for j in np.arange(0, 1, d):
                eig1 = np.linalg.eig(hamil(i * v1[0] + j * v2[0], i * v1[1] + j * v2[1]))
                V1 = eig1[1][:, np.argsort(eig1[0])[E]]
                eig2 = np.linalg.eig(hamil((i + d) * v1[0] + j * v2[0], (i + d) * v1[1] + j * v2[1]))
                V2 = eig2[1][:, np.argsort(eig2[0])[E]]
                eig3 = np.linalg.eig(hamil((i + d) * v1[0] + (j + d) * v2[0], (i + d) * v1[1] + (j + d) * v2[1]))
                V3 = eig3[1][:, np.argsort(eig3[0])[E]]
                eig4 = np.linalg.eig(hamil(i * v1[0] + (j + d) * v2[0], i * v1[1] + (j + d) * v2[1]))
                V4 = eig4[1][:, np.argsort(eig4[0])[E]]

                U1 = np.conjugate(V1).dot(V2) / linalg.norm(np.conjugate(V1).dot(V2))
                U2 = np.conjugate(V2).dot(V3) / linalg.norm(np.conjugate(V2).dot(V3))
                U3 = np.conjugate(V3).dot(V4) / linalg.norm(np.conjugate(V3).dot(V4))
                U4 = np.conjugate(V4).dot(V1) / linalg.norm(np.conjugate(V4).dot(V1))

                F12 = np.log(U1 * U2 * U3 * U4)
                chern_number[E] += F12

        chern_number[E] /= (2 * pi * 1j)
        print(f'Band {E + 1}, Chern Number = {round(np.real(chern_number[E]),2)}')	


if __name__ == '__main__':

	main()
