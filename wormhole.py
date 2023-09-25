# file to implement wormhole teleportation protocol

import numpy as np
import math

# define pauli matrices
Sx = np.array([[0, 1], [1, 0]]) 
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

# get SYK Hamiltonian
def get_H(N=10, J2=2):
    '''Returns the SYK Hamiltonian for N qubits.
    NOTE: wormhole paper used N = 10, J^2 = 2, beta= 4, mu = -12
    '''

    # helper function to return product of pauli matrices given a list of indices
    def get_prod(indices):
        '''Returns the uncoupled product of majorana fermions in qubit basis'''
        main_prod = np.eye(2**N)
        for ind in indices:
            for m in range(N-1): # loop over all qubits, except the last one; move the location of sigma_Z within tensor product
                prod= np.array([1])
                for n in range(N): # do the tensor product
                    if n != m:
                        prod = np.kron(prod, np.eye(2))
                    else:
                        prod = np.kron(prod, Sz)
            main_prod = np.dot(main_prod, prod)
            if ind %2 == 0:
                # build giant tensor product with sigma_X at end and identity everywhere else
                prod= np.array([1])
                for n in range(N): # do the tensor product
                    if n < N-1:
                        prod = np.kron(prod, np.eye(2))
                    else:
                        prod = np.kron(prod, Sy)
            else: # same as above, but we use a Sy instead of Sx
                prod= np.array([1])
                for n in range(N): # do the tensor product
                    if n < N-1:
                        prod = np.kron(prod, np.eye(2))
                    else:
                        prod = np.kron(prod, Sy)
            # append last result to main_prod
            main_prod = np.dot(main_prod, prod)
        return main_prod

    # initialize H
    H = np.zeros((2**N, 2**N), dtype=np.complex128)
    # get all possible combinations of N qubits
    for i in range(N):
        for j in range(i, N, 1):
            for l in range(j, N, 1):
                for k in range(j, N, 1):
                    print(i, j, l, k)
                    # determine coupling constant for this term
                    c = np.random.normal(loc=0.0, scale=math.factorial(3)*J2/(2**(N)))
                    # c=1
                    # assign matrices to each fermion in qubit basis by checking whether the index is odd or even
                    # get product of majorana fermions
                    prod_tot = get_prod([i, j, l, k])
                    # mutliply by c and add to H
                    H += c*prod_tot
                    print(H)
    return H


if __name__ == '__main__':
    H = get_H(N=4)
    print(H)
    # confirm that H is hermitian
    print(np.allclose(H, H.conj().T))

                    
            