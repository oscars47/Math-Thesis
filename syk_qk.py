# use qiskit to simulate SYK hamiltoninans and wormhole protocol
import numpy as np
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.opflow import I, X, Y, Z, PauliSumOp, PauliOp
import matplotlib.pyplot as plt
import os
from time import time
from math import factorial

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
            plt.savefig(os.path.join(path, f'{time.now()}.pdf'))
        else:
            plt.savefig(os.path.join(path, save_name+str(time.now())+'.pdf'))
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
## ------ simulation ------- ##
def trotter_suzuki_circuit(pauli_sum_op, time, steps):
    '''Perform the Trotter-Suzuki approximation of the time evolution operator.'''
    n_qubits = pauli_sum_op.num_qubits
    qc = QuantumCircuit(n_qubits)

    # First-order Trotter-Suzuki approximation
    delta_t = time / steps
    for _ in range(steps):
        for term in pauli_sum_op:
            # print('term:', term.coeff, )
            angle = -1j * term.primitive.coeffs[0] * delta_t
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

def get_SYK(n_majorana, J=2):
    '''Returns the SYK Hamiltonian as a PauliSumOp object in Qiskit.

    Params:
        N (int): number of Majorana fermions
        J (float): effective variance
    
    '''

    # Parameters
    n_qubits = n_majorana // 2
    var = factorial(3) * J**2 / (n_qubits**3)
    J = np.random.normal(0, np.sqrt(var), (n_majorana, n_majorana, n_majorana, n_majorana))

    # Ensure Jijkl is antisymmetric
    for i in range(n_majorana):
        for j in range(n_majorana):
            for k in range(n_majorana):
                for l in range(n_majorana):
                    if i >= j or j >= k or k >= l:
                        J[i, j, k, l] = 0

    # def majorana_to_qubit_op(i, n_qubits): ## OLD, wrong
    #     '''Map Majorana operator to qubit operator.'''
    #     qubit_index = i // 2
    #     operator = I ^ qubit_index  # Identity on all qubits before the target
    #     operator = operator ^ (X if i % 2 == 0 else Y)  # X or Y on the target qubit
    #     operator = operator ^ (I ^ (n_qubits - qubit_index - 1))  # Identity on all qubits after the target
    #     return operator
                        
    # def majorana_to_qubit_op(l, num_qubits): ## OLD, not accouting for 
    #     '''Performs Jordan Wigner transformation from Majorana to qubit operators, as in Jafferis and Gao et al.

    #     Note: `l` is the index of the Majorana operator in terms of the total majorana number, and `num_qubits` is the number of qubits in the system.
        
        
    #     '''
        
    #     # Initialize the operator as an identity for each qubit
    #     operator_chain = [I] * num_qubits

    #     # Determine the position of the final X or Y
    #     position = l // 2

    #     # Construct the alternating sequence of Z and X up to the position
    #     for i in range(position):
    #         operator_chain[i] = Z if i % 2 == 0 else X

    #     # Set the final X or Y at the position
    #     operator_chain[position] = X if l % 2 == 0 else Y

    #     # Create the full operator
    #     full_operator = operator_chain[0]
    #     for op in operator_chain[1:]:
    #         full_operator = full_operator ^ op

    #     return full_operator
                        
                        
    def majorana_to_qubit_op(l,  num_qubits):
        '''Performs Jordan Wigner transformation from Majorana to qubit operators, as in Jafferis and Gao et al. Assumes only left subsystem expression since one value of H_L,R is what's quoted in Jafferis et al wormhole paper.'''
        j = (l + 1) // 2
        
        # start with identity operator for all qubits
        operator_chain = [I] * num_qubits
        
        # apply (ZX)^(j-1)
        for idx in range(j - 1):
            operator_chain[idx] = Z if idx % 2 == 0 else X
        
        # apply the XX or YX at j-th position
        if l % 2 == 0:  # for 2j-1, we apply XX, which is just X because X^2 = I
            operator_chain[j - 1] = X
        else:  # for 2j, we apply YX, which simplifies to iZ
            operator_chain[j - 1] = Y  # Apply Y at the j-th position

        # if not the last qubit, apply X on the next qubit for the 2j-1 case
        if l % 2 == 0 and j < num_qubits:  # check if there's a qubit to apply X on
            operator_chain[j] = X

        # convert the list of single-qubit operators to a PauliOp
        full_operator = operator_chain[0]
        for op in operator_chain[1:]:
            full_operator = full_operator ^ op

        return full_operator

    # initialize Hamiltonian as all 0s
    H = PauliSumOp.from_list([("I" * n_qubits, 0.0)]) 

    for i in range(n_majorana):
        for j in range(n_majorana):
            for k in range(n_majorana):
                for l in range(n_majorana):
                    if J[i, j, k, l] != 0:
                        term = majorana_to_qubit_op(i, n_qubits) @ majorana_to_qubit_op(j, n_qubits) @ majorana_to_qubit_op(k, n_qubits) @ majorana_to_qubit_op(l, n_qubits)
                        H += J[i, j, k, l] * term
    
    # convert to unitary time evolution
    time = 1.0
    steps = 20
    H_circ = trotter_suzuki_circuit(H, time, steps)

    # print('H_circ:', H_circ)

    # log the total number of gates in the qc
    num_gates_initial_dict = H_circ.count_ops()
    num_gates_initial = sum(num_gates_initial_dict.values())
    print('Number of gates in initial circuit:', num_gates_initial)

    # run transpiler
    H_opt_circ = transpile(H_circ, optimization_level=1)

    # log the number of gates in the optimized qc
    num_gates_opt_dict = H_opt_circ.count_ops()
    num_gates_opt = sum(num_gates_opt_dict.values())
    print('Number of gates in optimized circuit:', num_gates_opt)

    # compute fidelity
    # get matrix representation of the circuit
    H_circ_matrix = Operator(H_circ).data
    H_opt_circ_matrix = Operator(H_opt_circ).data

    # fidelity
    fidelity = np.linalg.norm(H_circ_matrix - H_opt_circ_matrix)

    # log gate speedup
    gate_speedup = num_gates_initial / num_gates_opt

    print('Fidelity:', fidelity)
    print('Gate speedup:', gate_speedup)

    # print out both matrices
    print_matrix(H_circ_matrix)
    print_matrix(H_opt_circ_matrix)

    return fidelity, gate_speedup

## ------- testing ------- ##
def benchmark_SYK(num, n_majorana):
    '''Simulates num times of n_majorana SYK Hamiltonians and logs the average and sem fidelity and gate speedup.'''
    fidelities = []
    gate_speedups = []
    for _ in range(num):
        fidelity, gate_speedup = get_SYK(n_majorana)
        fidelities.append(fidelity)
        gate_speedups.append(gate_speedup)
    avg_fidelity = np.mean(fidelities)
    sem_fidelity = np.std(fidelities) / np.sqrt(num)
    avg_gate_speedup = np.mean(gate_speedups)
    sem_gate_speedup = np.std(gate_speedups) / np.sqrt(num)

    print(f'Average fidelity: {avg_fidelity} ± {sem_fidelity}')
    print(f'Average gate speedup: {avg_gate_speedup} ± {sem_gate_speedup}')

if __name__ == "__main__":
    # get_SYK(12)
    benchmark_SYK(10, 20) # N = 10 qubits
    print('Done!')