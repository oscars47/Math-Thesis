# file to generate matrices for sphynx to train on
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.extensions import UnitaryGate
import matplotlib.pyplot as plt
from tqdm import trange
from qiskit import transpile
import tensorly as tl
from tensorly.decomposition import tucker
import h5py, os
from tqdm import trange

GATE_SET = ['rx', 'ry', 'rz', 'p', 'cx']
GATE_SET_RP = ['rx', 'ry', 'rz', 'p']

def gen_random_circuit(N, depth,max_depth = None, N2=None):
    '''Returns random circuit with N qubits and depth.
    
    If N2 is None, then the circuit is on N qubits. Else, if we want to encode the circuit within a larger space, then N2 is the dimension of the larger space. 

    The circuit is returned as a tensor of shape (2**N2, 2**N2, 2) where the first two dimensions are the matrix representation of the circuit and the last dimension is the real and imaginary parts of the matrix.

    The circuit summary is returned as a matrix of shape (N2, depth) where each row corresponds to a qubit and each column corresponds to a layer. The entries are the gate applied to the qubit. The gates are encoded as follows:
        0: no gate
        1: rx
        2: ry
        3: rz
        4: p
        5i: cx control; i is the target
        6j: cx target; j is the control 

    Params:
        N: int, number of qubits
        depth: int, depth of circuit
        max_depth: int, maximum depth of circuit. will pad to get to full depth
        N2: int, dimension of hilbert space
    '''

    # Create a quantum circuit with the appropriate number of qubits
    if N2 is None:
        qc = QuantumCircuit(N)
        N2 = N
    else:
        qc = QuantumCircuit(N2)

    # random simplex probability to choose gates
    p = np.random.dirichlet(np.ones(5))
    p_new = p.copy()[:-1]
    p_new /= np.sum(p_new)

    # go through and apply random gates
    for i in range(N):
        for j in range(depth):
            # choose random set of gate
            if j < depth - 1:
                gate = np.random.choice(GATE_SET, p=p)
            else:
                gate = np.random.choice(GATE_SET_RP, p=p_new)
            if gate in GATE_SET_RP:
                if gate == 'rx':
                    # choose random angle
                    angle = np.random.uniform(0, 2*np.pi)
                    # apply gate
                    qc.rx(angle, i)
                elif gate == 'ry':
                    # choose random angle
                    angle = np.random.uniform(0, 2*np.pi)
                    # apply gate
                    qc.ry(angle, i)
                elif gate == 'rz':
                    # choose random angle
                    angle = np.random.uniform(0, 2*np.pi)
                    # apply gate
                    qc.rz(angle, i)
                elif gate == 'p':
                    # choose random angle
                    angle = np.random.uniform(0, 2*np.pi)
                    # apply gate
                    qc.p(angle, i)
            elif gate == 'cx':
                # choose random target not equal to itself and hasn't been used
                # create a list of eligible elements
                eligible_elements = [x for x in range(N) if x != i]

                # check if the list is empty
                if eligible_elements:
                    target = np.random.choice(eligible_elements)
                    
                    # apply gate
                    qc.cx(i, target)
                else:
                    # choose random gate from RP
                    gate = np.random.choice(['rx', 'ry', 'rz', 'p'], p=p_new)
                    # apply gate
                    if gate == 'rx':
                        # choose random angle
                        angle = np.random.uniform(0, 2*np.pi)
                        # apply gate
                        qc.rx(angle, i)
                    elif gate == 'ry':
                        # choose random angle
                        angle = np.random.uniform(0, 2*np.pi)
                        # apply gate
                        qc.ry(angle, i)
                    elif gate == 'rz':
                        # choose random angle
                        angle = np.random.uniform(0, 2*np.pi)
                        # apply gate
                        qc.rz(angle, i)
                    elif gate == 'p':
                        # choose random angle
                        angle = np.random.uniform(0, 2*np.pi)
                        # apply gate
                        qc.p(angle, i)
    # to print the circuit
    # qc.draw(output='mpl')
    # plt.show()  
                        
    if N2 is not None and N2 > N:
        # apply tensor product with identity to the rest of the qubits
        for i in range(N, N2):
            qc.id(i)

    optimized_circuit = transpile(qc, optimization_level=3, basis_gates=['rx', 'ry', 'rz', 'p', 'cx'])
    
    # create summary from the optimized circuit
    optimized_summary = [[] for _ in range(N)]

    for instruction, qargs, _ in optimized_circuit.data:
        gate = instruction
        gate_name = gate.name

        # check for RX, RY, RZ, P, or CX (CNOT) gates
        if gate_name in GATE_SET:
            # for CX (CNOT) gates, identify control and target qubits using find_bit
            if gate_name == 'cx':
                control_qubit_index = qc.find_bit(qargs[0]).index
                target_qubit_index = qc.find_bit(qargs[1]).index
                
            else:
                try:
                    qubit_index = [qc.find_bit(q).index  for q in qargs][0]
                except:
                    print(f'gate {gate_name} has multiple qubits, {[qc.find_bit(q).index for q in qargs]}')
            # add to summary, but check if it's a CX gate
            if gate_name == 'cx':
                optimized_summary[control_qubit_index].append(int(f'6{target_qubit_index}'))
                optimized_summary[target_qubit_index].append(int(f'7{control_qubit_index}'))
            else:
                # the index in the GATE_SET is what we add to the summary
                # find index
                name_index = GATE_SET.index(gate_name)+1
                optimized_summary[qubit_index].append(name_index)
    
    # convert the optimized summary to a single list using the '7' as the separator
    optimized_summary_new = []
    for row in optimized_summary:
        for r in row:
            optimized_summary_new.append(r)
        optimized_summary_new.append(7)

    # pad to max depth
        # full length is N2 * max_depth
    if max_depth is not None:
        optimized_summary_new += [0] * (N2 * max_depth - len(optimized_summary_new))
        
    optimized_summary = np.array(optimized_summary_new) # convert to numpy array

    # return the circuit as matrix and summary
    matrix = Operator(qc).data

    # compute eigenvalues
    eigenvalues, _ = np.linalg.eig(matrix)
    # since inout matrix is unitary, only need to store the angles
    angles = np.angle(eigenvalues)
    # convert to all positive so angles only 0 to 2pi
    angles = (angles + 2*np.pi) % (2*np.pi)

    # sort angles from smallest to largest
    angles = np.sort(angles)

    return angles, optimized_summary

def get_single_rc(N=10, depth=100, N2=None):
    '''Returns a single random circuit with N qubits and depth encoded in N2 qubit space.'''
    if N2 is None:
        N2 = N
    circuit_tensor, summary = gen_random_circuit(N, depth, N2)
    np.save(f'random_circuit_{N}_{depth}_{N2}.npy', circuit_tensor)
    np.save(f'random_summary_{N}_{depth}_{N2}.npy', summary)

def build_dataset(total_num, N2=10, N0=3, max_depth=100):
    '''Builds dataset of total_num random circuits with randomized N qubits in N2 qubit hilbert space and random depth.

    Params:
        total_num: int, total number of circuits to generate
        N2: int, dimension of hilbert space
        N0: int, minimum number of qubits in circuit
    
    
    '''

    # repeat each N total_num / (N2 - N0) times
    num_repeats = total_num // (N2 - N0)
    # get the vector of Ns
    Ns = np.arange(N0, N2)
    Ns = np.repeat(Ns, num_repeats)
    # shuffle
    np.random.shuffle(Ns)

    # create dataset
    x = []
    y = []
    for i in trange(len(Ns)):
        N = Ns[i]
        # random depth, uniform distribution
        # choose the depth from 1 to max_depth
        depth = np.random.randint(1, max_depth)
        circuit_tensor, summary = gen_random_circuit(N, depth, N2=N2, max_depth=max_depth)
        x.append(circuit_tensor)
        y.append(summary)

    # make sure save directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # save dataset
    x = np.array(x)
    y = np.array(y)
    np.save(os.path.join('data', f'x_{N2}_{max_depth}_{total_num}.npy'), x)
    np.save(os.path.join('data', f'y_{N2}_{max_depth}_{total_num}.npy'), y)

def plot_data_size():
    '''Plots data size in kb vs number of circuits.'''
    # for N2 = 10
    # max_depth = 100
if __name__ == '__main__':
    build_dataset(10)

 