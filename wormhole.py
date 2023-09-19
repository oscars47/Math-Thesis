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
    # initialize H
    H = np.zeros((2**N, 2**N), dtype=np.complex128)
    # helper function to return product of pauli matrices given a list of indices
    def get_prod(indices):
        '''Returns the uncoupled product of majorana fermions in qubit basis'''
        # initialize product
        prod_tot = np.eye(2**N)
        for ind in indices:
            for i in range(N-1): # big product of all the terms
                # need to take tensor product with Sz at index ind and everywhere else identity
                prod_ind = np.eye(2**N)
                for j in range(N): # building the tensor product term in the big product
                    if j == 0:
                        if i == 0: # initialize the product for this particular index
                            prod = Sz
                        else:
                            prod = np.eye(2)
                    elif j != ind:
                        prod = np.kron(prod, np.eye(2))
                    else:
                        prod = np.kron(prod, Sz)
                prod_ind = np.dot(prod_ind, prod)
            if ind %2 != 0:
                prod_ind = np.dot(prod_ind, np.kron(np.eye(2**(N-1)), Sx))
            else:
                prod_ind = np.dot(prod_ind, np.kron(np.eye(2**(N-1)), Sy))
        
            prod_tot = np.dot(prod_tot, prod_ind)
        return prod_tot

    # get all possible combinations of N qubits
    for i in range(N):
        for j in range(i, N, 1):
            for l in range(max(i, j), N, 1):
                for k in range(max(i, j, l), N, 1):
                    # determine coupling constant for this term
                    c = np.random.normal(loc=0.0, scale=math.factorial(3)*J2/(2**(N)), size=1)
                    # assign matrices to each fermion in qubit basis by checking whether the index is odd or even
                    # get product of majorana fermions
                    prod_tot = get_prod([i, j, l, k])
                    # mutliply by c and add to H
                    H += c*prod_tot
    return H


if __name__ == '__main__':
    H = get_H(N=4)
    print(H)
    # confirm that H is hermitian
    print(np.allclose(H, H.conj().T))
    print('eigenvalues')
    print(np.linalg.eigvals(H))

                    
            