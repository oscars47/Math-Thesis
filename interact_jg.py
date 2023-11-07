# file to implement the wormhole protocol using the analytic expression derived in Jafferis and Gao et al

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

# function to return the reduced density matrix for TR given a particular jex for Dirac fermion
def get_U_j(j,N,mu):
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
            U = get_U_j(j, N, mu)
        else:
            U += get_U_j(j, N, mu)
    return U

def get_TFD(H, beta=4):
    rho =  np.exp(-1j * H * beta / 2)
    return rho / np.trace(rho)

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
    H = np.load('ham/H_10/H_10_left_20231103-042012.npy', allow_pickle=True)
    print(get_IRT(0, 0, 10, H=H))


