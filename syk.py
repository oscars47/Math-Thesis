import numpy as np
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from numba import jit
from itertools import combinations
from functools import partial

# define pauli matrices
Sx = np.array([[0, 1], [1, 0]]) 
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

def is_hermitian(A):
    '''Returns True if A is Hermitian, False otherwise'''
    return np.allclose(A, A.conj().T)

def print_matrix(A, N=10, display=False):
    '''Prints the matrix.'''
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
            return np.kron(Sz, np.eye(2**(N-1), dtype=np.float64))
        elif m == N-2:
            part= np.kron(np.eye(2**(N-2), dtype=np.float64), Sz)
            return np.kron(part, np.eye(2, dtype=np.float64))
        else:
            return np.kron(np.kron(np.eye(2**m, dtype=np.float64), Sz), np.eye(2**(N-m-1), dtype=np.float64))

    # get the product of all the terms in the chain
    # prod = np.linalg.multi_dot([big_prod(m) for m in range(N-1)])
    # can't use multi_dot because it doesn't support jit
    prod = np.eye(2**N, dtype=np.float64)
    for m in range(N):
        # print(prod.dtype, big_prod(m).dtype)
        prod = prod.astype(np.float64)
        big_prod_m = big_prod(m).astype(np.float64)
        prod = prod @ big_prod_m

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
        # prod_tot = np.linalg.multi_dot([product_matrices[k] for k in range(len(product_matrices))])
        # can't use multi_dot because it doesn't support jit
        prod_tot = product_matrices[0]
        for k in range(1, len(product_matrices)):
            prod_tot = prod_tot @ product_matrices[k]
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

    # now add all the results to get the Hamiltonian
    for result in results:
        H += result

    return H

if __name__ == "__main__":
    import time, os
    from tqdm import trange

    def gen_inrange(start, stop):
        '''Generates 1 H matrix for each N in range(start, stop).'''
        for i in range(start, stop):
            t0 = time.time()
            H = get_H(i)
            t1 = time.time()
            print('time taken: ', t1-t0)
            # create directory to store if it doesn't exist
            if not os.path.exists('ham/H_' + str(i)):
                os.makedirs('ham/H_' + str(i))
            np.save('ham/H_' + str(i) + '.npy', H)
            print('H_' + str(i) + ' saved')

    def gen_HN(num, N):
        '''Generate and save num instances of H_N matrices. On a 2019 MacBook Pro, it takes ~10 min/ matrix.'''
        for i in trange(num):
            t0 = time.time()
            H = get_H(N)
            t1 = time.time()
            print('time taken: ', t1-t0)
            np.save('ham/H_10/H_10_' + str(i) + '.npy', H)
            print('H_10_' + str(i) + ' saved')

    def combine_N(N):
        '''Reads in all H_N matrices and makes a histogram of the real and imaginary parts of the eigenvalues.'''
        # initialize the arrays
        real = np.array([])
        imag = np.array([])

        # read in all the H_N matrices
        for file in os.listdir('ham/H_' + str(N)):
            if file.endswith('.npy'):
                H = np.load('ham/H_' + str(N) + '/' + file, allow_pickle=True)
                eigs = np.linalg.eigvals(H)
                real = np.append(real, np.real(eigs))
                imag = np.append(imag, np.imag(eigs))

        # get only unique values
        real = np.unique(real)
        imag = np.unique(imag)
        # remove the negative copies of the imaginary eigenvalues
        imag = imag[imag > 0]

        print('Number of unique real eigenvalues: ', len(real))
        print('Number of unique imaginary eigenvalues: ', len(imag))

        # plot the histogram
        plt.figure(figsize=(10, 10))
        plt.hist(real, bins=20, alpha=0.5, color='orange', label='Real')
        plt.hist(imag, bins=20, alpha=0.5, color='blue', label='Imaginary')
        plt.legend()
        plt.title('Eigenvalues of $H_{{%.3g}}$, Total Num = %.3g'%(N, len(real)))
        plt.savefig('ham/H_' + str(N) + '/hist.pdf')
        plt.show()

    # combine_N(10)

