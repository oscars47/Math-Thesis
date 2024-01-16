# file to test qiskit's decomposition
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.extensions import UnitaryGate
from elegans_adapt2 import random_circuit

N = 3
depth = 5

target = random_circuit(N, depth)

target_op = UnitaryGate(target)


# Create a quantum circuit with the appropriate number of qubits
qc = QuantumCircuit(N)

# Apply the unitary
qc.unitary(target_op, list(range(N)), label='U')

# Decompose the circuit
decomposed_circuit = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'p', 'cx'], optimization_level=3)
print(decomposed_circuit)
gate_counts = decomposed_circuit.count_ops()
print('number of gates with qiskit', gate_counts)

decomposed_circuit = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'p', 'cx'], optimization_level=0)
gate_counts = decomposed_circuit.count_ops()
# print(decomposed_circuit)
print('number of gates with qiskit', gate_counts)

# # use elegans adapt2
# from elegans_adapt2 import find_params
# find_params(target, model=2.1, depth=depth)
