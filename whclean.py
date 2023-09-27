import numpy as np
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from numba import jit
from itertools import combinations
from functools import partial

# define pauli matrices
Sx = np.array([[0, 1], [1, 0]], dtype=np.complex128) 
Sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

def is_hermitian(A):
    '''Returns True if A is Hermitian, False otherwise'''
    return np.allclose(A, A.conj().T)

def print_matrix(A, N=10, display=False):
    '''Prints the matrix A'''
    print('Matrix A:')
    print(A)

    fig, ax = plt.subplots(2, 1, figsize=(7, 10))
    # add colorbar
    im = ax[0].imshow(np.real(A), cmap='RdBu_r', vmin=-max(0, np.real(A)[0][0]), vmax =max(0, np.real(A)[0][0]))
    fig.colorbar(im, ax=ax[0])
    
# Display imaginary part and set color limits
    im2 = ax[1].imshow(np.imag(A), cmap='RdBu_r', vmin=-max(0, np.imag(A)[0][0]), vmax = max(0, np.imag(A)[0][0]))
    fig.colorbar(im2, ax=ax[1])

    ax[0].set_title('Real part')
    ax[1].set_title('Imaginary part')

    plt.suptitle('N = ' + str(N) + ' SYK Hamiltonian')
    plt.tight_layout()

    plt.savefig(f'ham/SYK_{N}.pdf')
    if display:
        plt.show()

# @jit(nopython=True)
# def get_product_matrix(i, N):
#     if i % 2 == 0:
#         return np.kron(np.eye(2**(N-1), dtype=np.complex128), Sx)
#     else:
#         return np.kron(np.eye(2**(N-1), dtype=np.complex128), Sy)

@jit(nopython=True)
def majorana(i, N=10):
    '''Returns the i-th Majorana operator for a system of size N.'''
    if i % 2 == 0:
        op = Sx
    else:
        op = Sy
    
    if i == 0:
        return np.kron(op, np.eye(2**(N-1), dtype=np.complex128))
    elif i == N - 1:
        return np.kron(np.eye(2**(N-1), dtype=np.complex128), op)
    else:
        return np.kron(np.kron(np.eye(2**(i//2), dtype=np.complex128), op), np.eye(2**(N-1-i//2), dtype=np.complex128))


@jit(nopython=True)
def get_product_matrices(indices, N):
    matrices  =[]
    for ind in indices:
        matrices.append(majorana(ind, N))
    return matrices

def do_chunk(ind, N = 10, J2=2):
        c = np.random.normal(loc=0.0, scale=math.factorial(3)*J2/(2**(N)))
        product_matrices = get_product_matrices(ind, N)
        prod_tot = np.linalg.multi_dot([product_matrices[k] for k in range(len(product_matrices))])
        return c * prod_tot

def get_H(N=10):
    '''Returns the SYK Hamiltonian for N qubits.'''
    H = np.zeros((2**N, 2**N), dtype=np.complex128)

    # parallelize this ----
    # first get all the combinations of 4 indices each ranging from 0 to N-1
    # if we restrict the indices to i < j < k < l, then we get non-hermitian matrix which contradicts Li et al
    # indices = np.array(list(combinations(range(N), 4)))
    indices = []
    for i in range(N-3):
        for j in range(i, N-2, 1):
            for l in range(j, N-1, 1):
                for k in range(l, N,1):
                    indices.append([i, j, k, l])
    indices = np.array(indices)

    func = partial(do_chunk, N=N, J2=2)

    # now do the parallelization
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(func, indices)
    
    for result in results:
        H += result

    return H

if __name__ == "__main__":
    import time
    # for N in range(4, 11):
    #     t0 = time.time()
    #     H = get_H(N)
    #     t1 = time.time()
    #     print('time taken: ', t1-t0)
    #     np.save('ham/H_' + str(N) + '.npy', H)
    #     print('H_' + str(N) + ' saved')
        
    #     # get image
    #     print_matrix(H, N=N)
    N = 6
    H = get_H(N)
    print_matrix(H, N=N, display=True)

    # load H
    # N=6
    # H = np.load(f'ham/H_{N}.npy')
    # # check if H is hermitian
    print('H is hermitian: ', is_hermitian(H))

    # look at 

