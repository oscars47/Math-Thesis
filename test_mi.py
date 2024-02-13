# file to test out mutual information measurements

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import DensityMatrix, partial_trace, entropy
from qiskit.compiler import transpile
import matplotlib.pyplot as plt


def main(savename):
    # create the circuit
    total_circuit = QuantumCircuit(8)
    total_circuit.h(0)
    total_circuit.cx(0, 1)
    total_circuit.swap(1, 2)
    # total_circuit.cx(1, 2)
    # total_circuit.cx(1, 3)
    # # total_circuit.cx(1, 3)
    # # total_circuit.rx(0.3, 2)
    # # total_circuit.h(3)
    # total_circuit.cx(2, 4)
    # total_circuit.cx(3, 5)
    # total_circuit.cx(4, 6)
    # starting at qubit 3, entangle to +3
    # total_circuit.ry(.3, 3)
    for i in range(3):
        # total_circuit.cx(i+2, i+3)
        total_circuit.cx(i+2, i+5)
        
    
    final = transpile(total_circuit, optimization_level = 1)

    backend = Aer.get_backend('statevector_simulator')
    job = execute(final, backend)
    result = job.result()

    # Get the statevector
    statevector = result.get_statevector(final)

    # Form the density matrix from the statevector
    density_matrix = DensityMatrix(statevector)

    # check if valid
    # print('is valid:', density_matrix.is_valid())

    # get the reduced density matrices
    rho_P = partial_trace(state = density_matrix, qargs = range(1, total_circuit.num_qubits))
    rho_T = partial_trace(state = density_matrix, qargs = range(total_circuit.num_qubits-1))
    rho_PT = partial_trace(state = density_matrix, qargs = range(1, total_circuit.num_qubits-1))

    # compute the mutual info
    I = entropy(rho_P) + entropy(rho_T) - entropy(rho_PT)
    print(f'Mutual info = {I}')

     # display the circuit
    total_circuit.draw('mpl')
    plt.title(f'Mutual info = {I}')
    plt.savefig(f'mi_ex/{savename}.pdf')
    plt.show()

if __name__ == '__main__':
    main(savename='6')