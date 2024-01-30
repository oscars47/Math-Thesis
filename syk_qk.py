# use qiskit to simulate SYK hamiltoninans and wormhole protocol
import numpy as np
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.opflow import I, X, Y, Z, PauliSumOp
import matplotlib.pyplot as plt
import os
from time import time
from math import factorial

## ------ visualization ------- ##
def print_matrix(matrix, save=False):
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
        plt.savefig(os.path.join(path, f'{time.now()}.pdf'))
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

    def majorana_to_qubit_op(i, n_qubits): ## NEED TO CHECK THIS WITH PAPER
        '''Map Majorana operator to qubit operator.'''
        qubit_index = i // 2
        operator = I ^ qubit_index  # Identity on all qubits before the target
        operator = operator ^ (X if i % 2 == 0 else Y)  # X or Y on the target qubit
        operator = operator ^ (I ^ (n_qubits - qubit_index - 1))  # Identity on all qubits after the target
        return operator

    # initialize Hamiltonian as all 0s
    H = PauliSumOp.from_list([("I" * n_qubits, 0.0)]) 

    for i in range(n_majorana):
        for j in range(n_majorana):
            for k in range(n_majorana):
                for l in range(n_majorana):
                    if J[i, j, k, l] != 0:
                        term = majorana_to_qubit_op(i, n_qubits) @ majorana_to_qubit_op(j, n_qubits) @ majorana_to_qubit_op(k, n_qubits) @ majorana_to_qubit_op(l, n_qubits)
                        H += J[i, j, k, l] * term


    # print('H:', H)
    
    # convert to unitary time evolution
    time = 1.0
    steps = 10
    H_circ = trotter_suzuki_circuit(H, time, steps)

    print('H_circ:', H_circ)

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
    gate_speedup = (num_gates_initial - num_gates_opt) / num_gates_initial

    print('Fidelity:', fidelity)
    print('Gate speedup:', gate_speedup)

    # print out both matrices
    print_matrix(H_circ_matrix)
    print_matrix(H_opt_circ_matrix)


if __name__ == "__main__":
    get_SYK(20)
    print("Done!")