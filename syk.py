import os
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import datetime

# define pauli matrices
Sx = np.array([[0, 1], [1, 0]]) 
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# helper functions #
def is_hermitian(A):
    '''Returns True if A is Hermitian, False otherwise'''
    return np.allclose(A, A.conj().T)

def is_unitary(A, tol=1e-10):
    '''Returns True if A is unitary, False otherwise'''
    A_dagger = A.conj().T
    if np.allclose(np.eye(len(A)), A @ A_dagger, atol=tol) and np.allclose(np.eye(len(A)), A_dagger @ A, atol=tol):
        return True
    else:
        print('A @ A_dagger: ', A @ A_dagger)
        loss = np.linalg.norm(A @ A_dagger - np.eye(len(A)))
        return 'False. Loss: %.3g'%loss

def commutator(A, B):
    '''Returns the commutator of A and B, i.e. AB - BA'''
    return A @ B - B @ A

def anti_commutator(A, B):
    '''Returns the anti-commutator of A and B, i.e. AB + BA'''
    return A @ B + B @ A

def is_zero(mat):
        '''Returns True if the matrix is all zeros.'''
        return np.isclose(mat, np.zeros((2**N, 2**N)), atol = 1e-10).all()

def print_matrix(A, N=10, is_SYK = True, other_name = None, l_r= None, ts = None, display=False):
    '''Prints the matrix.

    Params:
        A: matrix to print
        N: number of qubits
        is_SYK: True if the matrix is a SYK Hamiltonian, False otherwise
        other_name: name of the matrix if it is not a SYK Hamiltonian
        l_r: 'left' or 'right' or 'li' depending on whether the majorana operators are on the left or right side of the chain or instead using li's notation
        ts: timestamp of when the matrix was saved
        display: if True, displays the matrix
    
    
    '''

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Determine the scale for the real part
    max_real = np.abs(np.real(A)).max()
    im = ax[0].imshow(np.real(A), cmap='RdBu_r', vmin=-max_real, vmax=max_real)

    # Add colorbar for the first subplot
    # cbar_ax0 = fig.add_axes([ax[0].get_position().x1 + 0.02, ax[0].get_position().y0, 0.02, ax[0].get_position().height])
    fig.colorbar(im, ax=ax[0])

    # Determine the scale for the imaginary part
    max_imag = np.abs(np.imag(A)).max()
    im2 = ax[1].imshow(np.imag(A), cmap='RdBu_r', vmin=-max_imag, vmax=max_imag)

    # Add colorbar for the second subplot
    # cbar_ax1 = fig.add_axes([ax[1].get_position().x1 + 0.12, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
    fig.colorbar(im2, ax=ax[1])
    ax[0].set_title('Real part')
    ax[1].set_title('Imaginary part')
    if is_SYK:
        plt.suptitle(f'N = {N}, {l_r} SYK Hamiltonian')
    else:
        plt.suptitle(f'N = {N}, {other_name}')
    # fig.subplots_adjust(wspace=0.1)  # Adjust this value as needed
    plt.tight_layout()

    # if no directory exists, create it
    if not os.path.exists(f'ham/H_{N}'):
        os.makedirs(f'ham/H_{N}')

    if is_SYK:
        plt.savefig(f'ham/H_{N}/SYK_{N}_{l_r}_{ts}.pdf')
    else:
        plt.savefig(f'ham/H_{N}/{other_name}.pdf')
    if display:
        plt.show()

def majorana_li(ind, N):
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
    return prod * 1/np.sqrt(2)

def majorana_left(ind, N):
    ''' Returns the ind-th majorana fermion operator in qubit basis, using Gao and Jafferis for *LEFT* side of chain.

    NOTE: ASSUMES INDEX STARTS AT 1.

    Params:
        ind: index of the majorana operator
        N: number of qubits
    
    '''

    assert ind > 0, 'Index must be greater than 0.'
    assert ind <= N, f'Index must be less than or equal to N. Index = {ind}, N = {N}'
    
    if ind % 2 != 0: # odd: m = 2j - 1
        j = (ind + 1) // 2
        # loop to tensor Z tensor X j-1 times
        if j >= 2:
            for n in range(j - 1):
                if n > 0:
                    prod = np.kron(prod, Sz)
                else:
                    prod = Sz
                prod = np.kron(prod, Sx)
            # tensor with X tensor X
            prod = np.kron(prod, Sx)
        else:
            prod = Sx
        prod = np.kron(prod, Sx)

        # tensor with I tensor I for N/2 - j times
        if j < N//2:
            for _ in range(N//2 - j):
                prod = np.kron(prod, I)
                prod = np.kron(prod, I)

        return prod * 1/np.sqrt(2)


    else: # even: m = 2j
        # loop to tensor Z tensor X j-1 times
        j = ind // 2
        if j >= 2:
            for n in range(j-1):
                if n > 0:
                    prod = np.kron(prod, Sz)
                else:
                    prod = Sz
                prod = np.kron(prod, Sx)
            prod = np.kron(prod, Sy)   
        else:
            prod = Sy
        prod = np.kron(prod, Sx)

        # tensor with I tensor I for N - m times
        if j < N//2:
            for _ in range(N//2 - j):
                prod = np.kron(prod, I)
                prod = np.kron(prod, I)

        return prod * 1/np.sqrt(2)

def majorana_right(ind, N):
    ''' Returns the ind-th majorana fermion operator in qubit basis, using Gao and Jafferis for *RIGHT* side of chain.

    NOTE: ASSUMES INDEX STARTS AT 1.

    Params:
        ind: index of the majorana operator
        N: number of qubits
    
    '''

    assert ind > 0, 'Index must be greater than 0.'
    assert ind <= N, f'Index must be less than or equal to N. Index = {ind}, N = {N}'
    
    if ind % 2 != 0: # odd: m = 2j - 1
        j = (ind + 1) // 2
        # loop to tensor Z tensor X j-1 times
        if j >= 2:
            for n in range(j-1):
                if n > 0:
                    prod = np.kron(prod, Sz)
                else:
                    prod = Sz
                prod = np.kron(prod, Sx)

            prod = np.kron(prod, I)
        else:
            prod = I
        prod = np.kron(prod, Sy)

        # tensor with I tensor I for N/2 - j times
        if j < N//2:
            for _ in range(N//2 - j):
                prod = np.kron(prod, I)
                prod = np.kron(prod, I)
        return prod * 1/np.sqrt(2)

    else: # even: m = 2j
            # loop to tensor Z tensor X j-1 times
        j = ind // 2
        if j >= 2:
            for n in range(j-1):
                if n > 0:
                    prod = np.kron(prod, Sz)
                else:
                    prod = Sz
                prod = np.kron(prod, Sx)
      
            # tensor with Y tensor X
            prod = np.kron(prod, I)
        else:
            prod = I
        prod = np.kron(prod, Sz)

        # tensor with I tensor I for N - m times
        if j < N//2:
            for _ in range(N//2 - j):
                prod = np.kron(prod, I)
                prod = np.kron(prod, I)

        return prod * 1/np.sqrt(2)
    
def get_dirac_left(j, N):
    '''Gets the left dirac operator corresponding to the ind-th majorana operator'''
    assert j > 0, 'Index must be > 0.'
    assert j <= N//2, 'Index must be less than N/2.'

    m_left_0 = majorana_left(j*2-1, N)
    m_left_1 = majorana_left(j*2, N)

    return np.array(1/np.sqrt(2) * (m_left_0 + 1j * m_left_1))

def get_dirac_right(i, N):
    '''Gets the right dirac operator corresponding to the ind-th majorana operator'''
    assert i > 0, 'Index must be > 0.'
    assert i <= N//2, 'Index must be less than N/2.'

    m_right_0 = majorana_right(i*2-1, N)
    m_right_1 = majorana_right(i*2, N)

    return np.array(1/np.sqrt(2) * (m_right_0 + 1j * m_right_1))

def time_ev(H, t):
    '''Calculates time evolution operator by diagonalizing H'''
    hbar = 1
    # decompose H into eigenvalues and eigenvectors
    U = scipy.linalg.expm(-1j * H * t / hbar)
    print('Is U unitary? ', is_unitary(U))
    U = np.array(U)
    return U

def time_ev_op(H, t):
    '''Returns the time evolution operator for the Hamiltonian H at time t.'''
    hbar = 1
    # decompose H into eigenvalues and eigenvectors
    e_vals, e_vecs = np.linalg.eig(H)
    print('is H hermitian? ', is_hermitian(H))
    # are evecs orthonormal?
    print('Are evecs orthonormal? ', np.allclose(np.eye(len(H)), e_vecs @ e_vecs.conj().T))

    # normalize the eigenvectors
    for i in range(len(H)):
        e_vecs[:, i] = e_vecs[:, i] / np.sqrt(np.linalg.norm(e_vecs[:, i].conj().T @ e_vecs[:, i]))
        # e_vecs[:, i] = e_vecs[:, i] / np.sqrt(len(H))
    print('Are evecs normalized? ', np.allclose(np.eye(len(H)), e_vecs.conj().T @ e_vecs, atol=1e-10))

    print_matrix(e_vecs, N=len(H), is_SYK=False, other_name='evecs')

    print(e_vecs.shape)
    print(np.linalg.norm(e_vecs[:, 0]))

    print('Are there non real elements in e_vals? ', np.any(np.imag(e_vals)))
    print(np.linalg.norm(e_vecs[:, 0]))
    print(np.linalg.norm(e_vecs[:, 0].conj().T))


    # check eigenvectors are orthongonal. take rank of matrix where the evecs are columns
    print('Are evecs orthogonal? ', np.linalg.matrix_rank(e_vecs.conj().T @ e_vecs))
    print('Are evecs normalized? ', np.allclose(np.eye(len(e_vecs)), (e_vecs.conj().T @ e_vecs), atol=1e-10))

    print('outer prod mat', e_vecs.conj().T @ e_vecs)
    print_matrix(e_vecs.conj().T @ e_vecs, N=len(H), is_SYK=False, other_name='evecs_outer_prod')
    print(np.trace(e_vecs.conj().T @ e_vecs))

    # get the time evolution operator
    U = np.zeros((len(H), len(H)), dtype=np.complex128)
    for i in range(len(H)):
        # each term is v v^\dagger weighted by e^{-i E_i t / hbar}
        proj = np.outer(e_vecs[:, i], e_vecs[:, i].conj().T)
        U += np.exp(-1j * e_vals[i] * t / hbar) * proj

    print('Is U unitary? ', is_unitary(U))
    return U

def get_product_matrices(indices, N, l_r = 'left'):
    '''Returns the product of the majorana operators corresponding to the indices in the list indices.
    Params:
        indices: list of indices of the majorana operators
        N: number of qubits
        l_r: 'left' or 'right' or 'li' depending on whether the majorana operators are on the left or right side of the chain or instead using li's notation

    Returns:
        a matrix corresponding to the product of the majorana operators
    
    '''
    product = np.eye(2**N, dtype=complex)
    if l_r == 'left':
       for i in indices:
            product = product @  majorana_left(i, N)
    elif l_r == 'right':
        for i in indices:
            try:
                product = product @ majorana_right(i, N)
            except ValueError:
                print('ValueError at i = ', i)
    elif l_r == 'li':
        for i in indices:
            product = product @ majorana_li(i, N)
    return product

def get_H(N=10, J2=2, dirac=False, l_r = 'left'):
    '''Returns the L, R, SYK Hamiltonian for N qubits and the timestamp of creation.

    Params:
        N: number of qubits
        J2: coupling constant
        dirac: If true, saves H_L and H_R using same coupling constant as specified in 2.37 in Jafferis and Gao. If false, only saves either left or right as indicated by l_r.
        l_r: 'left' or 'right' or 'li' depending on whether the majorana operators are on the left or right side of the chain or instead using li's notation

    Returns:
        H: SYK Hamiltonian
        timestamp: timestamp of when the matrix was saved

        Also saves the matrix in the ham/H_N directory.
    
    
    '''
    H = np.zeros((2**N, 2**N), dtype=np.complex128)

    # parallelize this ----
    # first get all the combinations of 4 indices each ranging from 0 to N-1
    # if we restrict the indices to i < j < k < l, then we get non-hermitian matrix which contradicts Li et al
    # indices = np.array(list(combinations(range(N), 4)))
    indices = []
    for i in range(1,N+1):
        for j in range(i+1, N):
            for l in range(j+1, N-1):
                for k in range(l+1, N-2):
                    indices.append([i, j, l, k])
    indices = np.array(indices)

    # precompute the majorana
    if not(dirac):
        product_matrices = np.array([get_product_matrices(index_ls, N, l_r)  for index_ls in indices])
        print(product_matrices.shape)
        c = np.random.normal(loc=0.0, scale=math.factorial(3)*J2/(2**(N)), size=len(product_matrices))
        # scale each product matrix by the corresponding c
        H_terms = np.array([c[i] * product_matrices[i] for i in range(len(indices))])

        # ---------------------
        # sum all the terms
        H = np.zeros((2**N, 2**N), dtype=np.complex128)
        for i in range(len(indices)):
            H += H_terms[i]
        # ---------------------
        # save H with timestamp
        # ---------------------
        # make sure directory exists
        if not os.path.exists(f'ham/H_{N}'):
            os.makedirs(f'ham/H_{N}')
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        np.save(f'ham/H_{N}/H_{N}_{l_r}_{timestamp}.npy', H)
        H = np.array(H)
        H = H.reshape((2**N, 2**N))
        return H, timestamp
    else:
        prod_left = np.array([get_product_matrices(index_ls, N, 'left')  for index_ls in indices])
        prod_right = np.array([get_product_matrices(index_ls, N, 'right')  for index_ls in indices])
        c = np.random.normal(loc=0.0, scale=math.factorial(3)*J2/(2**(N)), size=len(prod_left))
        # scale each product matrix by the corresponding c
        H_terms_l = np.array([c[i] * prod_left[i] for i in range(len(indices))])
        H_terms_r = np.array([c[i] * prod_right[i] for i in range(len(indices))])

        # ---------------------
        # sum all the terms
        H_l = np.zeros((2**N, 2**N), dtype=np.complex128)
        H_r = np.zeros((2**N, 2**N), dtype=np.complex128)
        for i in range(len(indices)):
            H_l += H_terms_l[i]
            H_r += H_terms_r[i]
        # ---------------------
        # save H with timestamp
        # ---------------------
        # make sure directory exists
        if not os.path.exists(f'ham/H_{N}'):
            os.makedirs(f'ham/H_{N}')
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        np.save(f'ham/H_{N}/H_{N}_left_{timestamp}.npy', H_l)
        np.save(f'ham/H_{N}/H_{N}_right_{timestamp}.npy', H_r)
        return [H_l, H_r], timestamp

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

    def gen_HN(num, N, display=False):
        '''Generate and save num instances of H_N matrices. On a 2019 MacBook Pro, it takes ~10 min/ matrix.'''
        for i in trange(num):
            t0 = time.time()
            H = get_H(N)
            print(H.shape)
            if display:
                plt.imshow(np.abs(H))
                plt.show()
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

    ## comparing the majorana operators ##
    N = 10
    # get_H(N, dirac=True)
    # n_l = 3
    # n_r = 3
    # print_matrix(get_dirac_left(n_l, N), N, is_SYK=False, other_name=f'dirac_left_{n_l}')
    # print_matrix(get_dirac_right(n_r, N), N, is_SYK=False, other_name=f'dirac_right_{n_r}')

    # time_ev(get_H(N)[0], 1)
    # mat = get_H(N)[0]
    # time_ev(mat, 1)
    # time_ev_op(mat, 1)
    # print('time evol op: ', time_ev(get_H(N)[0], 1).shape)
    # print('dirac left: ', get_dirac_left(1, N).shape)
    print('prod', time_ev(get_H(N)[0], 1) @ get_dirac_left(1, N))
    # print('Majorana left: ', majorana_left(1, N).shape)
    







    # print('-------')
    # for ind in range(1, N+1):
    #     print(ind)
    #     print('Majorana left: ', majorana_left(ind, N).shape)
    #     print('Majorana right: ', majorana_right(ind, N).shape)
    #     print('-------')
    # print(majorana_left(10, N))
    # print(majorana_left(10, N).shape)
    # print(majorana_right(0, N))
    # print(np.trace(anti_commutator(majorana_right(3, N), majorana_left(1, N))))

    # H_l, ts_l = get_H(N, l_r='left')
    # H_r, ts_r = get_H(N, l_r='right')

    # print('H_l: ')
    # print_matrix(H_l, N=N, l_r = 'Left', ts=ts_l)
    # print('Is H_l hermitian? ', is_hermitian(H_l))
    # print('e vals of H_l: ', np.linalg.eigvals(H_l))
    # print('H_r: ')
    # print_matrix(H_r, N=N, l_r = 'Right', ts=ts_r)
    # print('Is H_r hermitian? ', is_hermitian(H_r))
    # print('e vals of H_r: ', np.linalg.eigvals(H_r))


