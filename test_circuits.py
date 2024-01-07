# file to test circuits I've learned with IBM simulators and real devices!
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.circuit import Parameter

# Example circuit description
# Format: [("gate", parameter), ...]
# Supported gates: 'h' (Hadamard), 'x', 'y', 'z', 'rx', 'ry', 'rz'
circuit_description = [
    [("h", None), ("rz", 0.5)],
    [("x", None)],
    [("ry", 1.0), ("rz", 0.3)]
]

# create quantum circuit
num_qubits = len(circuit_description)
qc = QuantumCircuit(num_qubits)

# Function to add gate to the circuit based on description
## NEED TO ADAPT THIS
def add_gate(qc, gate, qubit, param):
    if gate == "h":
        qc.h(qubit)
    elif gate == "x":
        qc.x(qubit)
    elif gate == "y":
        qc.y(qubit)
    elif gate == "z":
        qc.z(qubit)
    elif gate == "rx":
        qc.rx(param, qubit)
    elif gate == "ry":
        qc.ry(param, qubit)
    elif gate == "rz":
        qc.rz(param, qubit)

# Construct the circuit
for qubit, gates in enumerate(circuit_description):
    for gate, param in gates:
        add_gate(qc, gate, qubit, param)

# Print the circuit
print(qc)

def simulate(qc):
    # Execute the circuit on a simulator
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    print(counts)

def run_on_ibmq(qc):
    # Load IBM Q account and get the least busy backend device
    IBMQ.load_account()  # Load your IBM Q account
    provider = IBMQ.get_provider(hub='ibm-q')
    qcomp = provider.get_backend('ibmq_5_yorktown')  # Example backend
    job = execute(qc, qcomp, shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    print(counts)

