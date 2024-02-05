# use qiskit to simulate SYK hamiltoninans and wormhole protocol
import numpy as np
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.opflow import I, X, Y, Z, PauliSumOp, PauliOp
import matplotlib.pyplot as plt
import os, time
from math import factorial
from scipy.linalg import expm

## ------ visualization ------- ##
def print_matrix(matrix, show=False, save=False, save_name=None):
    '''Nicely prints out matrix with color coded mag and phase.'''
    mag = np.abs(matrix)
    phase = np.angle(matrix)

    # wherever mag is 0, make phase 0
    phase = np.where(mag == 0, 0, phase)

    # plot both matrices with color bar; add colorbar
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im = ax[0].imshow(mag, cmap='RdBu_r')
    fig.colorbar(im, ax=ax[0])
    im2 = ax[1].imshow(phase, cmap='RdBu_r')
    fig.colorbar(im2, ax=ax[1])
    if save:
        # add to 'syk_qiskit' folder
        path = os.path.join(os.getcwd(), 'syk_qiskit')
        if not os.path.exists(path):
            os.makedirs(path)
        # timestamp
        if save_name is None:
            plt.savefig(os.path.join(path, f'{time.time()}.pdf'))
        else:
            plt.savefig(os.path.join(path, save_name+str(time.time())+'.pdf'))
    if show:
        plt.show()
    
## ------ qiskit helpers ------- ##
def num_gates_pso(pauli_sum_op):
    '''Count number of gates for PauliSumOp object.'''
    gate_count = 0
    # Extract the SparsePauliOp
    sparse_pauli_op = pauli_sum_op.primitive

    # Iterate over each Pauli string in SparsePauliOp
    for pauli_string in sparse_pauli_op.paulis:
        # Count non-identity operators in the Pauli string
        gate_count += np.sum(pauli_string != 'I')

    return gate_count

#### ------ WORMHOLE PROTOCOL ------- ####
def majorana_to_qubit_op(l,  num_qubits, left=True):
    '''Performs Jordan Wigner transformation from Majorana to qubit operators, as in Jafferis and Gao et al.

    Params:
        l (int): index of Majorana operator
        num_qubits (int): number of qubits
        left (bool): whether the Majorana operator is on the left or right side 
    
    '''
    j = (l + 1) // 2
    
    # start with identity operator for all qubits
    operator_chain = [I] * num_qubits
    
    # apply (ZX)^(j-1)
    for idx in range(j - 1):
        operator_chain[idx] = Z if idx % 2 == 0 else X
    
    if left:
        # apply the XX or YX at j-th position
        if l % 2 == 0:  # for 2j-1, we apply XX, which is just X because X^2 = I
            operator_chain[j - 1] = X
        else:  # for 2j, we apply YX, which simplifies to iZ
            operator_chain[j - 1] = Y  # Apply Y at the j-th position

         # if not the last qubit, apply X on the next qubit for the 2j-1 case
        if l % 2 == 0 and j < num_qubits:  # check if there's a qubit to apply X on
            operator_chain[j] = X

    else:
        # apply I j-th position
        operator_chain[j - 1] = I

        # if not the last qubit, apply Y if odd or Z if even on the next qubit for the 2j-1 case
        if l % 2 == 0 and j < num_qubits:
            operator_chain[j] = Z
        elif l % 2 == 1 and j < num_qubits:
            operator_chain[j] = Y

    # convert the list of single-qubit operators to a PauliOp
    full_operator = operator_chain[0]
    for op in operator_chain[1:]:
        full_operator = full_operator ^ op

    return full_operator

def get_SYK(n_majorana, J=2, left=True):
    '''Returns the SYK Hamiltonian as a PauliSumOp object in Qiskit.

    Params:
        N (int): number of Majorana fermions
        J (float): effective variance
        save (bool): whether to save the matrix representation of the circuit
        save_name (str): name of the file to save as
    
    '''

    print(f'Generating SYK Hamiltonian for N_m = {n_majorana}...')

    # Parameters
    n_qubits = n_majorana // 2
    var = factorial(3) * J**2 / (n_majorana**3)
    J = np.random.normal(0, np.sqrt(var), (n_majorana, n_majorana, n_majorana, n_majorana))

    # Ensure Jijkl is antisymmetric
    for i in range(n_majorana):
        for j in range(n_majorana):
            for k in range(n_majorana):
                for l in range(n_majorana):
                    if i >= j or j >= k or k >= l:
                        J[i, j, k, l] = 0                     

    # initialize Hamiltonian as all 0s
    H = PauliSumOp.from_list([("I" * n_qubits, 0.0)]) 

    for i in range(n_majorana):
        for j in range(n_majorana):
            for k in range(n_majorana):
                for l in range(n_majorana):
                    if J[i, j, k, l] != 0:
                        term = majorana_to_qubit_op(i, n_qubits, left=left) @ majorana_to_qubit_op(j, n_qubits, left=left) @ majorana_to_qubit_op(k, n_qubits, left=left) @ majorana_to_qubit_op(l, n_qubits, left=left)
                        H += J[i, j, k, l] * term

    return H

## ------ TROTTER-SUZUKI ------- ##
def trotter_suzuki_circuit(pauli_sum_op, time, steps):
    '''Perform the Trotter-Suzuki approximation of the time evolution operator.'''
    n_qubits = pauli_sum_op.num_qubits
    qc = QuantumCircuit(n_qubits)

    # First-order Trotter-Suzuki approximation
    delta_t = time / steps
    for _ in range(steps):
        for term in pauli_sum_op:
            # print('term:', term.primitive.coeffs[0] )
            # need to check if delta_t is real or imaginary
            if np.isreal(delta_t):
                angle = -1j * np.abs(term.primitive.coeffs[0]) * delta_t # sometimes Qiskit will autoconvert YX = iZ thus adding a phase, so we need to take the abs
            else:
                angle = np.abs(term.primitive.coeffs[0]) * delta_t
            angle = np.imag(angle)
            # print('angle:', angle)
            pauli_strings = term.primitive.paulis.to_labels()
            for qubit_idx, gate_coeff in enumerate(pauli_strings):
                    for i, gate in enumerate(gate_coeff):
                        # print('qubit_idx:', i)
                        # print('gate_char:', gate)
                        if gate == 'X':
                            qc.rx(2 * angle, i)
                        elif gate == 'Y':
                            qc.ry(2 * angle, i)
                        elif gate == 'Z':
                            qc.rz(2 * angle, i)
                        # 'I' does nothing
    return qc

def time_evolve(H, tf=1, steps=20, benchmark=False, save=False, save_name=None):
    '''Time evolves the Hamiltonian H from t0 to tf in steps.'''
    print('Time evolving...')
    # get the circuit 
    H_circ = trotter_suzuki_circuit(H, tf, steps)

    # log the total number of gates in the qc
    num_gates_initial_dict = H_circ.count_ops()
    num_gates_initial = sum(num_gates_initial_dict.values())
    print('Number of gates in initial circuit:', num_gates_initial)

    # run transpiler
    H_opt_circ = transpile(H_circ, optimization_level=1)
    print('Transpiled...')

    # log the number of gates in the optimized qc
    num_gates_opt_dict = H_opt_circ.count_ops()
    num_gates_opt = sum(num_gates_opt_dict.values())
    print('Number of gates in optimized circuit:', num_gates_opt)

    if benchmark:
        # compute fidelity
        # get matrix representation of the circuit
        H_circ_matrix = Operator(H_circ).data
        print('Converted to matrix')
        H_opt_circ_matrix = Operator(H_opt_circ).data

        # fidelity
        fidelity = np.linalg.norm(H_circ_matrix - H_opt_circ_matrix)

        # log gate speedup
        gate_speedup = num_gates_initial / num_gates_opt

        print('Fidelity:', fidelity)
        print('Gate speedup:', gate_speedup)

        # print out both matrices
        print_matrix(H_circ_matrix, save=True, save_name='H_circ')
        print_matrix(H_opt_circ_matrix, save=True, save_name='H_opt_circ')

        if save:
            # add to 'syk_qiskit' folder
            path = os.path.join(os.getcwd(), 'syk_qiskit')
            if not os.path.exists(path):
                os.makedirs(path)
            # timestamp
            if save_name is None:
                np.save(os.path.join(path, f'H_circ_{time.time()}'), H_circ_matrix)
                np.save(os.path.join(path, f'H_opt_circ_{time.time()}'), H_opt_circ_matrix)
            else:
                np.save(os.path.join(path, save_name+f'H_circ_{time.time()}'), H_circ_matrix)
                np.save(os.path.join(path, save_name+f'H_opt_circ_{time.time()}'), H_opt_circ_matrix)

        return fidelity, gate_speedup
    else:
        return H_opt_circ # just return the optimized circuit

## ------ TFD and helper functions adapted from Zlokapa's thesis ------- ##
# def get_TFD(H, beta=4):
#     '''Takes in pauli sum op and returns the TFD state. Assumes time reversal applied first.

#     Params:
#         H (PauliSumOp): Hamiltonian
#         beta (float): inverse temperature
    
#     '''
#     # get the matrix representation of the Hamiltonian
#     H_mat = H.to_matrix()
#     N = int(np.log2(H_mat.shape[0]))

#     expH = expm(-beta * H_mat/4)

#     # apply time reversal
#     tfd = time_reverse(expH@get_bell_pair(N), right=True)

#     # get parition function to normalize
#     Z = np.sqrt(np.vdot(tfd, tfd))

#     return tfd / Z

# def get_bell_pair(N):
#     '''Returns the bell state in N qubit hilbert space'''
#     zero =np.array([1, 0])
#     one = np.array([0, 1])

#     bell_pair = (np.kron(zero, zero) + np.kron(one, one)) * 1/np.sqrt(2)

#     # now put inside N qubit hilbert space
#     if N==2:
#         return bell_pair
#     else:
#         epr = bell_pair
#         for _ in range(N//2-1):
#             epr = np.kron(bell_pair, epr)

#         return epr
    
# def time_reverse(M, right=True):
#     '''Calls time_reverse_op on matrix M'''
#     N = int(np.log2(M.shape[0]))
#     m = time_reverse_op(N, right)
#     return m @ np.conjugate(M)

# def time_reverse_op(N, right=True):
#     '''Returns the time reversal operator for N qubits'''
#     Sy = np.array([[0, -1j], [1j, 0]]) # Pauli Y

#     mr = np.kron(np.identity(1), -1j*Sy)
#     ml = np.kron(-1j*Sy, np.identity(1))

#     print(mr, ml)

#     if right:
#         m = mr
#     else:
#         m = ml

#     for _ in range(N-2):
#         if right:
#             m = np.kron(m, mr)
#         else:
#             m = np.kron(m, ml)

#     return m
    
def get_TFD(H, beta=4, steps=20):
    '''Takes in pauli sum op and returns the TFD state. Assumes time reversal applied first. Performs computation as quantum circuit. Applies Trotter-Suzuki with imag time evolve the Hamiltonian.

    Params:
        H (PauliSumOp): Hamiltonian
        beta (float): inverse temperature
        steps (int): number of steps for Trotter-Suzuki
        '''

    # exponentiate using trotter suzuki
    tfd_i = trotter_suzuki_circuit(H, -1j*beta, steps)
    print(tfd_i)

    H_exp = -1j*beta/4 * H

    # use trotter suzuki to get the circuit
    tfd = time_evolve(H_exp, steps)

    return tfd # is already normalized bc quantum circuit

def get_bell_pair(N):
    '''Returns the bell state in N qubit hilbert space using quantum circuit'''
    qc = QuantumCircuit(N)
    qc.h(0)
    qc.cx(0, 1)
    return qc
    
## ------ Apply potential ------- ##
def get_expV(n_majorana, mu =-12, steps=20):
    '''Computes the exponential of the potential V for n_majorana fermions for interaction param mu'''

    n_qubits = n_majorana // 2

    # initialize the potential as pauli sum op
    V = PauliSumOp.from_list([("I" * n_qubits, 0.0)])
    
    for j in range(n_majorana):
        term = majorana_to_qubit_op(j, n_qubits, left=True) @ majorana_to_qubit_op(j, n_qubits, left=False)
        V += 1/(4*n_majorana) * term

    # scale by mu
    V = mu * V

    # compute exponential with trotter suzuki
    expV = time_evolve(V, tf=1, steps=steps)
    return expV

## ------ main function to implment the protocol ------- ##
def implement_protocol(n_majorana, mu=-12, beta=4):
    '''Computes the correlation K for n_majorana fermions at time t, interaction param mu, and inverse temperature beta'''

    # get the Hamiltonian
    H = get_SYK(n_majorana) # pauli sum op

    # make tfd
    tfd = get_TFD(H, beta) # quantum circuit

    # get bell pair P and Q
    bell_pair = get_bell_pair(n_majorana) # quantum circuit














## ------- testing ------- ##
def benchmark_SYK(num, n_majorana):
    '''Simulates num times of n_majorana SYK Hamiltonians and logs the average and sem fidelity and gate speedup.'''
    fidelities = []
    gate_speedups = []
    for _ in range(num):
        fidelity, gate_speedup = time_evolve(get_SYK(n_majorana), benchmark=True)
        fidelities.append(fidelity)
        gate_speedups.append(gate_speedup)
    avg_fidelity = np.mean(fidelities)
    sem_fidelity = np.std(fidelities) / np.sqrt(num)
    avg_gate_speedup = np.mean(gate_speedups)
    sem_gate_speedup = np.std(gate_speedups) / np.sqrt(num)

    print(f'Average fidelity: {avg_fidelity} ± {sem_fidelity}')
    print(f'Average gate speedup: {avg_gate_speedup} ± {sem_gate_speedup}')

if __name__ == "__main__":
    # print(time.time())
    # get_SYK(20, save=True, save_name='SYK_20')
    # benchmark_SYK(10, 20) # N = 10 qubits
    H = get_SYK(10)
    tfd = get_TFD(H)
    print(tfd)