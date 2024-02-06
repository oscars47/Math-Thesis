import numpy as np
from scipy.linalg import expm


# ------ TFD and helper functions adapted from Zlokapa's thesis ------- ##
def get_TFD(H, beta=4):
    '''Takes in pauli sum op and returns the TFD state. Assumes time reversal applied first.

    Params:
        H (PauliSumOp): Hamiltonian
        beta (float): inverse temperature
    
    '''
    # get the matrix representation of the Hamiltonian
    H_mat = H.to_matrix()
    N = int(np.log2(H_mat.shape[0]))

    expH = expm(-beta * H_mat/4)

    # apply time reversal
    tfd = time_reverse(expH@get_bell_pair(N), right=True)

    # get parition function to normalize
    Z = np.sqrt(np.vdot(tfd, tfd))

    return tfd / Z

def get_bell_pair(N):
    '''Returns the bell state in N qubit hilbert space'''
    zero =np.array([1, 0])
    one = np.array([0, 1])

    bell_pair = (np.kron(zero, zero) + np.kron(one, one)) * 1/np.sqrt(2)

    # now put inside N qubit hilbert space
    if N==2:
        return bell_pair
    else:
        epr = bell_pair
        for _ in range(N//2-1):
            epr = np.kron(bell_pair, epr)

        return epr
    
def time_reverse(M, right=True):
    '''Calls time_reverse_op on matrix M'''
    N = int(np.log2(M.shape[0]))
    m = time_reverse_op(N, right)
    return m @ np.conjugate(M)

def time_reverse_op(N, right=True):
    '''Returns the time reversal operator for N qubits'''
    Sy = np.array([[0, -1j], [1j, 0]]) # Pauli Y

    mr = np.kron(np.identity(1), -1j*Sy)
    ml = np.kron(-1j*Sy, np.identity(1))

    print(mr, ml)

    if right:
        m = mr
    else:
        m = ml

    for _ in range(N-2):
        if right:
            m = np.kron(m, mr)
        else:
            m = np.kron(m, ml)

    return m
    