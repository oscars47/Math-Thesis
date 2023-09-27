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

    # Determine the scale for the real part
    max_real = np.abs(np.real(A)).max()
    im = ax[0].imshow(np.real(A), cmap='RdBu_r', vmin=-max_real, vmax=max_real)
    fig.colorbar(im, ax=ax[0])

    # Determine the scale for the imaginary part
    max_imag = np.abs(np.imag(A)).max()
    im2 = ax[1].imshow(np.imag(A), cmap='RdBu_r', vmin=-max_imag, vmax=max_imag)
    fig.colorbar(im2, ax=ax[1])

    ax[0].set_title('Real part')
    ax[1].set_title('Imaginary part')

    plt.suptitle('N = ' + str(N) + ' SYK Hamiltonian')
    plt.tight_layout()

    plt.savefig(f'ham/SYK_{N}.pdf')
    if display:
        plt.show()

# @jit(nopython=True)
def majorana(ind, N):
    ''' Returns the ind-th majorana fermion operator in qubit basis'''
    def big_prod(m):
        '''Returns the m-th product term in the majorana fermion chain in qubit basis'''
        if m == 0:
            return np.kron(Sz, np.eye(2**(N-1)))
        elif m == N-2:
            part= np.kron(np.eye(2**(N-2)), Sz)
            return np.kron(part, np.eye(2))
        else:
            return np.kron(np.kron(np.eye(2**m), Sz), np.eye(2**(N-m-1)))

    # get the product of all the terms in the chain
    prod = np.linalg.multi_dot([big_prod(m) for m in range(N-1)])
    # add the last term which is either Sx or Sy at the end of the big tensor
    if ind % 2 == 0:
        prod = prod @ np.kron(np.eye(2**(N-1)), Sx)
    else:
        prod = prod @ np.kron(np.eye(2**(N-1)), Sy)
    return prod


# @jit(nopython=True)
def get_product_matrices(indices, N):
    return np.array([majorana(i, N) for i in indices])

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
    from tqdm import trange
    
    for N in range(4, 11):
        t0 = time.time()
        H = get_H(N)
        t1 = time.time()
        print('time taken: ', t1-t0)
        np.save('ham/H_' + str(N) + '.npy', H)
        print('H_' + str(N) + ' saved')
        
        # get image
        print_matrix(H, N=N)

    # get a bunch for H_10s:
    for i in trange(100):
        H = get_H(10)
        np.save('ham/H_10/H_10_' + str(i) + '.npy', H)
        print('H_10_' + str(i) + ' saved')




    # N = 4
    # H = get_H(N)
    # print_matrix(H, N=N, display=True)

    # load H
    # N=6
    # H = np.load(f'ham/H_{N}.npy')
    # # check if H is hermitian
    # print('H is hermitian: ', is_hermitian(H))

