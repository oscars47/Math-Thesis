# file to implement the wormhole protocol using the analytic expression derived in Jafferis and Gao et al
import os
import numpy as np
import scipy
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
    # check if trace 1, within tolerance. can use param rtol to change tolerance
    if not(np.isclose(np.trace(rho), 1, tolerance)):
        if verbose: print('rho trace is not 1', np.trace(rho))
        return False
    return True

# function to return the reduced density matrix for TR given a particular jex for Dirac fermion
def get_Uj(j,N,mu):
    # only sum over odd since otherwise projectors won't commute
    if j % 2 == 0:
        j_term = np.kron(Px_p, scipy.linalg.expm(1j*mu*Sz / 2)) + np.kron(Px_m, scipy.linalg.expm(-1j*mu*Sz / 2))
        
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
        j_term = np.kron(Py_p, scipy.linalg.expm(1j*mu*Sy / 2)) + np.kron(Py_m, scipy.linalg.expm(-1j*mu*Sy / 2))

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
    print(H)
    
    # get the eigenvalues and eigenvectors of the Hamiltonian
    eig_val, eig_vec = np.linalg.eigh(H)
    print('len eig_val', len(eig_val))
    # print('first 10 eig val', eig_val[:10])
    N = int(np.log2(H.shape[0]))
    # get the partition function
    Z = np.sqrt(np.sum(np.exp(-beta * eig_val/2)**2))
    # get the density matrix
    TFD = np.exp(-beta * eig_val /2) / Z 
    print(TFD)
    print(np.linalg.norm(TFD))
    TFD = TFD.reshape(2**N, 1)
    rho_TFD = TFD @ TFD.conj().T
    print('shape', rho_TFD.shape)
    print('is valid rho?', is_valid_rho(rho_TFD))
    return rho_TFD

def get_rho_TR(t, j_1 = 1, j_2 = 2, N=10, H=None, J2 = 4,beta=4, nt0 = -2.8):
    '''Define the non-trivial correlation functions and combine to get reduced density matrix.
    Params:
        t: time at which to evaluate the reduced density matrix
        j_1: index of the first majorana
        j_2: index of the second majorana
        N: number of qubits
        H: Hamiltonian
        l_r: left or right fermion
        J2: J^2 coupling strength
        beta: inverse temperature
        nt0: negative of initial time; assuming fixed injection time
    '''

    if H is None:
        H_ls = get_H(N, J2)
        H = np.array(H_ls[0] + H_ls[1])

    TFD = get_TFD(H=H, beta=beta)

    # get time evolution operator
    U_nt0 = time_ev(H,-nt0)
    U_t = time_ev(H,t)

    U = get_U(N)
    U_dagger = adjoint(U)

    # get right fermion
    psi_1_r = U_t @ majorana_right(j_1, N) # time-evolve the right fermion by t
    psi_1_r = psi_1_r.reshape(2**N, 2**N)
    psi_1_r_dagger = adjoint(psi_1_r)
    psi_2_r = U_t @ majorana_right(j_2, N) # time-evolve the right fermion by t
    psi_2_r = psi_2_r.reshape(2**N, 2**N)
    psi_2_r_dagger = adjoint(psi_2_r)

    # get left fermion
    psi_1_l = U_nt0 @ majorana_left(j_1, N),  # time-evolve the left fermion by -t0
    psi_1_l = np.array(psi_1_l[0])
    psi_1_l = psi_1_l.reshape(2**N, 2**N)
    psi_1_l_dagger = adjoint(psi_1_l)
    psi_2_l = U_nt0 @ majorana_left(j_2, N),  # time-evolve the left fermion by -t0
    psi_2_l = np.array(psi_2_l[0])
    psi_2_l = psi_2_l.reshape(2**N, 2**N)
    psi_2_l_dagger = adjoint(psi_2_l)

    # get rho_11
    rho_11 = .5 * (1 - np.trace(
        anti_commutator(psi_1_l, commutator(psi_2_l, psi_1_l @ U_dagger @ psi_1_r @ psi_2_r @ U)
                        ) @ TFD
    ))
    # get rest of diagonals
    rho_22 = 1 - rho_11
    rho_33 = 1 - rho_11
    rho_44 = 1 - rho_22

    # get off diagonal
    rho_14 = .5 * np.trace(
        anti_commutator(psi_1_l, U_dagger @ psi_1_r @ U)  @ TFD
    ) - np.trace(
        anti_commutator(psi_2_l, psi_1_l @ U_dagger @ psi_2_r @ U @ psi_1_l) @ TFD
    )

    # combine to get density matrix
    rho_TR = .5*np.array([[rho_11, 0, 0, rho_14], [0, rho_22, 0, 0], [0, 0, rho_33, 0], [rho_14.conj(), 0, 0, rho_44]])

    print('is rho_TR valid?', is_valid_rho(rho_TR))
    return rho_TR

def get_rho_TR_old(t, j, N, H=None, J2 = 4,beta=4, nt0 = -2.8):
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
        H = get_H(N, J2)[0]
        H = np.array(H)
        print('H', H)

    TFD = get_TFD(H, beta)

    # get time evolution operator
    U_nt0 = time_ev(H,-nt0)
    U_t = time_ev(H,t)

    chi_l = U_nt0 @ get_dirac_left(j, N),  # time-evolve the left fermion by -t0
    chi_l =chi_l[0]
    print('chi_l', chi_l)
    chi_l_dagger = chi_l.conj().T
    chi_r = U_t @ get_dirac_right(j, N) # time-evolve the right fermion by t
    chi_r = chi_r.reshape(2**N, 2**N)
    chi_r_dagger = chi_r.conj().T

    U = get_U(N)
    U_dagger = U.conj().T

    # chi_l @ chi_l_dagger @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l @ chi_l_dagger

    print('-------')

    print('first expec', np.trace(chi_l @ chi_l_dagger @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l @ chi_l_dagger @ TFD))

    print('-------')

    ## using the direct forms of chi; not sure if this is correct -- (not density matrices!)
    rho_11 = np.trace(chi_l @ chi_l_dagger @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l @ TFD)
    
    rho_14 = np.trace(chi_l @ U_dagger @ chi_r_dagger @ U @ chi_l @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ chi_l @ U_dagger @ chi_r_dagger @ U @ chi_l @ TFD)
    
    rho_22 = np.trace(chi_l @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ chi_l @ U_dagger @ chi_r @ chi_r_dagger @ U @ chi_l_dagger @ chi_l @ TFD)

    rho_23 = np.trace(chi_l @ chi_l_dagger @ U_dagger @ chi_r_dagger @ U @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ U_dagger @ chi_r_dagger @ U @ chi_l_dagger @ chi_l @ TFD)

    rho_33 = np.trace(chi_l @ chi_l_dagger @ U_dagger @ chi_r_dagger @ chi_r @ U @ chi_l @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ U_dagger @ chi_r_dagger @ chi_r @ U @ chi_l @ TFD)

    rho_44 = np.trace(chi_l @ U_dagger @ chi_r_dagger @ chi_r @ U @ chi_l_dagger @ TFD) + np.trace(chi_l_dagger @ chi_l @ U_dagger @ chi_r_dagger @ chi_r @ U @ chi_l_dagger @ chi_l @ TFD) 

    # construct the density matrix
    rho_TR = .5 * np.array([[rho_11, 0, 0, rho_14], [0, rho_22, rho_23, 0], [0, rho_23.conj(), rho_33, 0], [rho_14.conj(), 0, 0, rho_44]])

    print('is rho_TR valid?', is_valid_rho(rho_TR))
    return rho_TR

def S(rho):
    print('rho', rho)
    '''Returns the von Neumann entropy of the density matrix rho'''

    # check valid density matrix
    if not(is_valid_rho(rho)):
        raise Exception('rho is not a valid density matrix')

    # diagonalize rho
    eig_val, eig_vec = np.linalg.eigh(rho)
    # get the entropy
    return -np.sum(eig_val * np.log2(eig_val, where=eig_val>0))

def get_IRT(t, j_1=1, j_2=2, N=10, H=None, J2 = 4,beta=4, nt0 = -2.8):
    '''Computes the mutual information of the RT state'''
    rho_TR = get_rho_TR(t=t, j_1=j_1, j_2 = j_2, N=N, H=H, J2 = J2, beta=beta, nt0 = nt0)
    print(rho_TR)
    rho_TR = rho_TR.reshape(2, 2, 2, 2)
    print('reshape', rho_TR)
    rho_T = np.trace(rho_TR, axis1=0, axis2=1)
    rho_R = np.trace(rho_TR, axis1=2, axis2=3)
    return S(rho_T) + S(rho_R) - S(rho_TR)

if __name__ == '__main__':
    H_l = np.load('ham/H_10/H_10_right_20231114-020214.npy', allow_pickle=True)
    H_r = np.load('ham/H_10/H_10_right_20231114-020214.npy', allow_pickle=True)
    H = H_l + H_r

    # print(get_IRT(t=1, H=H))
    # get_TFD(H=H)
    print(get_rho_TR(2, H=H))
    
    # print_matrix(time_ev(H_l, 1), N = 10, is_SYK=False, other_name='U(t=1)')
    # print(time_ev(np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz), 2))
    # print(time_ev(Sx, 1))
    

    # print(get_IRT(1, 1, 10, nt0=2, H=H_l))
    # print(np.linalg.eigvals(get_rho_TR(1, 1, 10, nt0=2, H=H)))
    # TFD = is_valid_rho(get_TFD(H))
    # print_matrix(get_TFD(H), N=10, is_SYK=False, other_name='TFD')


