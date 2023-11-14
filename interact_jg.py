# file to implement the wormhole protocol using the analytic expression derived in Jafferis and Gao et al
import os
import numpy as np
from syk import * # my code implementing their majorana operators and Hamiltonians

# projectors #
Px_p = (np.eye(2)+Sx)/2
Px_m = (np.eye(2)-Sx)/2
Py_p = (np.eye(2)+Sy)/2
Py_m = (np.eye(2)-Sy)/2

## overview of protocol: ##
# 1. at time -t', apply SWAP between right qubit in Bell state (Q) and left qubit in TFD state (L)
# 2. at time 0, apply U = e^{i mu V} to the state
# 3. at time t, apply SWAP between right qubit of TFD (R) and the final readout qubit T

def adjoint(state):
    ''' Returns the adjoint of a state vector. For a np.matrix, can use .H'''
    return np.conjugate(state).T

def is_valid_rho(rho, verbose=True):
    ''' Checks if a density matrix is valid. 
    params:
        rho: density matrix to check
        verbose: bool, whether to print out what is wrong with rho
    '''
    tolerance = 1e-17
    # make sure not a 0 matrix
    if np.all(np.isclose(rho, np.zeros((rho.shape[0],rho.shape[1])), rtol=tolerance)):
        if verbose: print('rho is 0 matrix')
        return False
    # check if Hermitian
    if not(np.all(np.isclose(rho,adjoint(rho), rtol=tolerance))):
        if verbose: print(f'rho is not Hermitian')
        return False
    # check if trace 1, within tolerance. can use param rtol to change tolerance
    if not(np.isclose(np.trace(rho), 1, tolerance)):
        if verbose: print('rho trace is not 1', np.trace(rho))
        return False
    # check if positive semidefinite
    eig_val = np.linalg.eigvals(rho)
    if not(np.all(np.greater_equal(eig_val,np.zeros(len(eig_val))) | np.isclose(eig_val,np.zeros(len(eig_val)), rtol=tolerance))):
        if verbose: print('rho is not positive semidefinite. eigenvalues:', eig_val)
        return False
    # square root of rho must exist
    if np.isnan(rho).any() or np.isinf(rho).any():
        if verbose: 
            print('rho has infs or nans')
            print('nan', np.isnan(rho))
            print('inf', np.isinf(rho))
            print(rho)
        return False
    return True

# function to return the reduced density matrix for TR given a particular jex for Dirac fermion
def get_Uj(j,N,mu):
    # only sum over odd since otherwise projectors won't commute
    if j % 2 == 0:
        j_term = np.kron(Px_p, np.exp(1j*mu*Sz / 2)) + np.kron(Px_m, np.exp(-1j*mu*Sz / 2))
        
        # put inside tensor product
        j = j // 2
        if j > 1:
            for n in range(j-1):
                if n > 0:
                    prod = np.kron(prod, np.eye(2))
                else:
                    prod = np.eye(2)
                prod = np.kron(prod, np.eye(2))
            # tensor with Y tensor X
            prod = np.kron(prod, j_term)
        else:
            prod = j_term
        # tensor with I tensor I for N - m times
        if j > 0:
            for _ in range(N//2 - j):
                prod = np.kron(prod, np.eye(2))
                prod = np.kron(prod, np.eye(2))
        else:
            for _ in range(N//2 - j - 1):
                prod = np.kron(prod, np.eye(2))
                prod = np.kron(prod, np.eye(2))
        return prod
    else:
        j_term = np.kron(Py_p, np.exp(1j*mu*Sy / 2)) + np.kron(Py_m, np.exp(-1j*mu*Sy / 2))

        j = (j + 1) // 2
        # loop to tensor Z tensor X j-1 times
        if j > 1:
            for n in range(j-1):
                if n > 0:
                    prod = np.kron(prod, np.eye(2))
                else:
                    prod = Sz
                prod = np.kron(prod, np.eye(2))
            # tensor with X tensor X
            prod = np.kron(prod, j_term)
        else:
            prod = j_term
        # tensor with I tensor I for N/2 - j times
        for _ in range(N//2 - j):
            prod = np.kron(prod, np.eye(2))
            prod = np.kron(prod, np.eye(2))
    
        return prod
    
def get_U(N, mu=-12):
    '''Returns the unitary operator U = e^{i mu V}'''
    # get the Hamiltonian
    # get the potential operator
    for j in range(0, N, 2):
        if j == 0:
            U = get_Uj(j, N, mu)
        else:
            U += get_Uj(j, N, mu)
    return U

def get_TFD(H, beta=4):
    '''Returns the density matrix for the thermal state of the Hamiltonian H at inverse temperature beta'''
    
    # get the eigenvalues and eigenvectors of the Hamiltonian
    eig_val, eig_vec = np.linalg.eigh(H)
    # get the partition function
    Z = np.sum(np.exp(-beta * eig_val))
    # get the density matrix
    rho = np.zeros((H.shape[0], H.shape[0]), dtype=complex)
    for i in range(len(eig_val)):
        rho += np.exp(-beta * eig_val[i]) / Z * np.outer(eig_vec[:, i], eig_vec[:, i].conj())
    return rho

def get_rho_TR(t, j, N, H=None, l_r = 'left', J2 = 4,beta=4, nt0 = -2.8):
    '''Define the non-trivial correlation functions and combine to get reduced density matrix.
    Params:
        t: time at which to evaluate the reduced density matrix
        j: index of the Dirac fermion
        N: number of qubits
        H: Hamiltonian
        l_r: left or right fermion
        J2: J^2 coupling strength
        beta: inverse temperature
        nt0: negative of initial time; assuming fixed injection time
    '''

    if H is None:
        H = get_H(N, J2, l_r)

    TFD = get_TFD(H, beta)

    chi_l = time_ev(get_dirac_left(j, N), -nt0) # time-evolve the left fermion by -t0
    chi_l_dagger = chi_l.conj().T
    chi_r = time_ev(get_dirac_right(j, N), t) # time-evolve the right fermion by t
    chi_r_dagger = chi_r.conj().T

    U = get_U(N)
    U_dagger = U.conj().T

    # print('sample operator', chi_l @ chi_l_dagger)

    # chi_l @ chi_l_dagger @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l @ chi_l_dagger

    ## using the direct forms of chi; not sure if this is correct -- (not density matrices!)
    rho_11 = np.trace(chi_l @ chi_l_dagger @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l @ TFD)
    
    rho_14 = np.trace(chi_l @ U_dagger @ chi_r_dagger @ U @ chi_l @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ chi_l @ U_dagger @ chi_r_dagger @ U @ chi_l @ TFD)
    
    rho_22 = np.trace(chi_l @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ chi_l @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l_dagger @ chi_l @ TFD)

    rho_23 = np.trace(chi_l @ chi_l_dagger @ U_dagger @ chi_r_dagger @ U @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ U_dagger @ chi_r_dagger @ U @ chi_l_dagger @ chi_l @ TFD)

    rho_33 = np.trace(chi_l @ chi_l_dagger @ U_dagger @ chi_r_dagger @ chi_r @ U @ chi_l @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ U_dagger @ chi_r_dagger @ chi_r @ U @ chi_l @ TFD)

    rho_44 = np.trace(chi_l @ U_dagger @ chi_r_dagger @ chi_r @ U @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ chi_l @ U_dagger @ chi_r_dagger @ chi_r @ U @ chi_l_dagger @ chi_l @ TFD) 

    # construct the density matrix
    rho_TR = .5 * np.array([[rho_11, 0, 0, rho_14], [0, rho_22, rho_23, 0], [0, rho_23.conj(), rho_33, 0], [rho_14.conj(), 0, 0, rho_44]])
    return rho_TR

def S(rho):
    '''Returns the von Neumann entropy of the density matrix rho'''
    return -np.trace(rho @ np.log(rho))

def get_IRT(t, j, N, H=None, l_r = 'left', J2 = 4,beta=4, nt0 = -2.8):
    '''Computes the mutual information of the RT state'''
    rho_TR = get_rho_TR(t, j, N, H, l_r, J2, beta, nt0)
    print(rho_TR)
    rho_TR = rho_TR.reshape(2, 2, 2, 2)
    rho_T = np.trace(rho_TR, axis1=0, axis2=1)
    rho_R = np.trace(rho_TR, axis1=2, axis2=3)
    return S(rho_T) + S(rho_R) - S(rho_TR)

if __name__ == '__main__':
    H_l = np.load('ham/H_10/H_10_right_20231114-020214.npy', allow_pickle=True)
    H_r = np.load('ham/H_10/H_10_right_20231114-020214.npy', allow_pickle=True)
    H = H_l + H_r
    
    print_matrix(time_ev(H, 1), N = 10, is_SYK=False, other_name='U(t=1)')
    

    # print(get_IRT(1, 1, 10, nt0=2, H=H))
    # print(np.linalg.eigvals(get_rho_TR(1, 1, 10, nt0=2, H=H)))
    # TFD = is_valid_rho(get_TFD(H))
    # print_matrix(get_TFD(H), N=10, is_SYK=False, other_name='TFD')


