# file to implement wormhole teleportation protocol

import numpy as np
import math
import matplotlib.pyplot as plt

# define pauli matrices
Sx = np.array([[0, 1], [1, 0]]) 
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

# get SYK Hamiltonian
# helper function to return product of pauli matrices given a list of indices
def get_prod(indices, N):
    '''Returns the uncoupled product of majorana fermions in qubit basis'''
    main_prod = np.eye(2**N)
    for ind in indices:
        part_prod = np.eye(2**N)
        for m in range(N-1): # loop over all qubits, except the last one; move the location of sigma_Z within tensor product
            prod= np.array([1])
            for n in range(N): # do the tensor product
                if n != m:
                    prod = np.kron(prod, np.eye(2))
                else:
                    prod = np.kron(prod, Sz)
            part_prod = part_prod @ prod
        if ind %2 == 0:
            # build giant tensor product with sigma_X at end and identity everywhere else
            prod= np.array([1])
            for n in range(N): # do the tensor product
                if n < N-1:
                    prod = np.kron(prod, np.eye(2))
                else:
                    prod = np.kron(prod, Sx)
        else: # same as above, but we use a Sy instead of Sx
            prod= np.array([1])
            for n in range(N): # do the tensor product
                if n < N-1:
                    prod = np.kron(prod, np.eye(2))
                else:
                    prod = np.kron(prod, Sy)
        part_prod = part_prod @ prod
        main_prod = main_prod @ part_prod 
    return main_prod

def anti_commutator(A, B):
    '''Returns the anti-commutator of A and B'''
    return A @ B + B @ A

def is_hermitian(A):
    '''Returns True if A is Hermitian, False otherwise'''
    return np.allclose(A, A.conj().T)

def get_H(N=10, J2=2):
    '''Returns the SYK Hamiltonian for N qubits.
    NOTE: wormhole paper used N = 10, J^2 = 2, beta= 4, mu = -12
    '''
    # initialize H
    H = np.zeros((2**N, 2**N), dtype=np.complex128)
    # get all possible combinations of N qubits
    # Li et al says i < j < k < l
    # if we sum over all combinations, then get non-Hermitian !!
    for i in range(N-3):
        for j in range(i, N-2, 1):
            for l in range(j, N-1, 1):
                for k in range(l, N,1):
                    # print(i, j, l, k)
                    # determine coupling constant for this term
                    c = np.random.normal(loc=0.0, scale=math.factorial(3)*J2/(2**(N)))
                    # print(c)
                    # c=1
                    # assign matrices to each fermion in qubit basis by checking whether the index is odd or even
                    # get product of majorana fermions
                    prod_tot = get_prod([i, j, l, k], N)
                    # fig, ax = plt.subplots(2, 1)
                    # ax[0].imshow(np.real(prod_tot))
                    # ax[1].imshow(np.imag(prod_tot))
                    # plt.show()
                    # mutliply by c and add to H
                    H += c*prod_tot
                    # print(H)
    return H


if __name__ == '__main__':
    H = get_H(N=4)
    print('is H hermitian?', is_hermitian(H))
    # fig, ax = plt.subplots(2, 1)
    # ax[0].imshow(np.real(H))
    # ax[1].imshow(np.imag(H))
    # plt.show()

    # print(H)
    # # confirm that H is hermitian
    # print(np.allclose(H, H.conj().T))
    # psi1 = get_prod([1], 4)
    # psi2 = get_prod([2], 4)
    # print(anti_commutator(psi1, psi2))
    # print(anti_commutator(psi1, psi1))
    # mat = get_prod([0, 2, 3, 4], 4)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].imshow(np.real(mat))
    # ax[1].imshow(np.imag(mat))
    # plt.show()


                    
            