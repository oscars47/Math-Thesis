# application of tucker decomp to approximating unitaries

import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot, mode_dot
from scipy.linalg import expm
from datetime import datetime

## create circuits ##
def make_circuit(gate_ls, gate_set='IHSTC'):
    '''Make a circuit from a list of gates.

    Params:
        gate_ls: list of strings representing gates. e.g., 'HIX'
        gate_set (str): set of gates to use. Default is HSTCX: Hadamard, S, T, CNOT
    '''
    # ensure len of each element is the same
    assert len(set([sum(2 if g == 'C' else 1 for g in gate) for gate in gate_ls])) == 1, 'gates must be of the same length'
    # assert all gates are in gate_set
    assert all(g in gate_set for gate in gate_ls for g in gate), 'gates must be in gate set'

    # define universal gate set
    if gate_set == 'IHSTC':
        I = np.eye(2)
        H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        S = np.array([[1, 0], [0, 1j]])
        T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
        CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                        [0, 0, 0, 1], [0, 0, 1, 0]])
        gate_dict = {'I': I, 'H': H, 'S': S, 'T': T, 'C': CNOT}

    def get_gate_tensor(gate_str):
        '''Applies a gate to a circuit, from a string representation of the gates.'''
        # get all separate gates as ls
        gate_ls = [gate_dict[gate] for gate in gate_str]
        # now get the tensor product of all gates
        gate_tensor = gate_ls[0]
        for gate in gate_ls[1:]:
            gate_tensor = np.kron(gate_tensor, gate)
        # print('Gate string', gate_str)
        # print('Gate tensor', gate_tensor)
        # apply gate
        return gate_tensor

    # create circuit
    # len of first element in gate ls defines number of qubits
    n = len(gate_ls[0])
    # initialize circuit as identity
    circuit = np.eye(2**n, dtype=complex)
    # apply gates, starting from the right
    for gate in gate_ls:
        # apply gate
        circuit = circuit @ get_gate_tensor(gate)

    return circuit

## create and validate an arbitrary unitrary ##
def is_unitary(matrix, tol=1e-10):
    '''Checks if a matrix is unitary.'''
    return np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=tol)

def get_u_n_generators(n):
    '''Returns the generators of u(n) as a list of numpy arrays. By definition, need A^dagger = -A and A^dagger = A^T for A in generators.
    
    '''
    generators = []
    # add n-1 diagonal generators
    for i in range(n):
        diag_matrix = np.zeros((n, n), dtype=complex)
        diag_matrix[i, i] = -1
        generators.append(diag_matrix)

    # add off-diagonal generators
    for i in range(n):
        for j in range(i + 1, n):
            real_matrix = np.zeros((n, n), dtype=complex)
            real_matrix[i, j] = real_matrix[j, i] = 1
            generators.append(real_matrix)

            imag_matrix = np.zeros((n, n), dtype=complex)
            imag_matrix[i, j] = -1j
            imag_matrix[j, i] = 1j
            generators.append(imag_matrix)

    # print('Number of generators:', len(generators))
    return generators

def get_U_n(params, generators):
    '''Exponentiates a linear combination of the generators of u(n).'''
    params = np.array(params)
    generator_sum = sum(p * g for p, g in zip(params, generators))
    mat =  expm(1j*generator_sum)
    assert is_unitary(mat), 'U is not unitary'
    return mat

def get_random_unitary(n):
    '''Returns a random unitary matrix of size n x n.'''
    generators = get_u_n_generators(n)
    params = np.random.normal(size=len(generators))
    return get_U_n(params, generators)

## tucker decomposition for qubit system ##
def factorize_tucker(target, d=2):
    '''Factor a d x d matrix into a tensor product of two d x d matrices.'

    Params:
        target (np.ndarray): matrix to be factorized
        d (int): dimension of decomposition matrices
    
    '''
    assert target.shape[0] == target.shape[1], 'target must be a square matrix'
    assert target.shape[0] % d == 0, 'decomposition size must divide target size'

    # reshape target into a tensor of shape d repeated log base d of target size
    n = int(np.log(target.shape[0]*target.shape[1]) / np.log(d))

    rank = [d] * n
    tensor = target.reshape(rank)

    # perform Tucker decomposition on the tensor
    # num factors = n
    core, factors = tucker(tensor, rank=rank)
    # print dimensionality
    print('Core:\n', core)
    print('Core shape:', core.shape)
    for factor in factors:
        # print('------')
        # print(np.round(factor, 10))
        # print('------')
        print('Factor:', factor.shape)

    # reconstruct target from the factors and reshape from [d] * n tensor back to  matrix
    target_recon = multi_mode_dot(core, factors, modes=list(range(n)))
    target_recon = target_recon.reshape(target.shape[0],target.shape[1])

    # try manual reconstruction
    for n in range(len(factors)):
        print('------')
        core = mode_dot(core, factors[n], n)
        print(f'Core after mode {n}:\n', core)
        

    core = core.reshape(target.shape[0], target.shape[1])

    print('Target:\n', target)
    print('Reconstructed target:\n', target_recon)
    print('Manual reconstruction:\n', core)

    norm = np.linalg.norm(target - target_recon)
    print('Norm:', norm)
    assert norm < 1e-10, 'reconstructed matrix does not match target matrix'

    # try to reconstruct using kronecker product
    # kron_recon = np.eye(d**n)
    # for factor in factors:
    #     # multiply by core
    #     kron_recon = np.kron(kron_recon, factor)

        
    # norm = np.linalg.norm(target - kron_recon)


    return factors

def print_matrix(A, title='Title', display=False):
    '''Prints the matrix.

    Params:
        A: matrix to print
        title (str): title of the plot
        display: if True, displays the matrix
    
    '''

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Determine the scale for the real part
    max_real = np.abs(A).max()
    im = ax[0].imshow(np.abs(A), cmap='RdBu_r', vmin=-max_real, vmax=max_real)

    # Add colorbar for the first subplot
    # cbar_ax0 = fig.add_axes([ax[0].get_position().x1 + 0.02, ax[0].get_position().y0, 0.02, ax[0].get_position().height])
    fig.colorbar(im, ax=ax[0])

    # Determine the scale for the imaginary part
    max_imag = np.angle(A).max()
    im2 = ax[1].imshow(np.angle(A), cmap='RdBu_r', vmin=-max_imag, vmax=max_imag)

    # Add colorbar for the second subplot
    # cbar_ax1 = fig.add_axes([ax[1].get_position().x1 + 0.12, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
    fig.colorbar(im2, ax=ax[1])
    ax[0].set_title('Magnitude')
    ax[1].set_title('Angle')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{title}.pdf')
    if display:
        plt.show()

if __name__ == '__main__':
    # get arbitrary unitary
    # d = 2**4
    # U = get_random_unitary(d)
    # factors = factorize_tucker(U, 2)

    circuit = ['HIST', 'CTT', 'HHHH']
    U = make_circuit(circuit)
    print(np.round(factorize_tucker(U, 2), 10))



    ## get intuition for mode_dot ##
    # get sample 3x3x3 array
    # A = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
    # A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(A.shape)
    # # factor = 2*np.eye(3)
    # factor = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]])
    # print(factor)
    # print('-----')
    # print(mode_dot(A, factor, 0))
    # print(mode_dot(A, factor, 1))
    # print('-----')
    # print(factor @ A)
    # print((factor @ A.T).T)

    


    
