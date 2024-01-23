# file to generate matrices for sphynx to train on
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
# from qiskit.extensions import UnitaryGate
import matplotlib.pyplot as plt
from tqdm import trange
from qiskit import transpile
import os, pickle
from tqdm import trange
from time import time

GATE_SET = ['rx', 'ry', 'rz', 'p', 'cx']
GATE_SET_RP = ['rx', 'ry', 'rz', 'p']

## prepare decomposition ##
def decompose_qc(optimized_circuit, N, max_depth=None, N2=None, target=None):
    '''Decomposes the optimized circuit into normalized vector of angles and summary vector representing the circuit

    Params:
        optimized_circuit: QuantumCircuit, optimized circuit
        N: int, number of qubits
        max_depth: int, maximum depth of circuit
        N2: int, dimension of hilbert space
        target: np.array, (optional) target matrix to compute norm of difference
    '''

     # create summary from the optimized circuit
    optimized_summary = [[] for _ in range(N)]

    for instruction, qargs, _ in optimized_circuit.data:
        gate = instruction
        gate_name = gate.name

        # check for RX, RY, RZ, P, or CX (CNOT) gates
        if gate_name in GATE_SET:
            # for CX (CNOT) gates, identify control and target qubits using find_bit
            if gate_name == 'cx':
                control_qubit_index = optimized_circuit.find_bit(qargs[0]).index
                target_qubit_index = optimized_circuit.find_bit(qargs[1]).index
                
            else:
                try:
                    qubit_index = [optimized_circuit.find_bit(q).index  for q in qargs][0]
                except:
                    print(f'gate {gate_name} has multiple qubits, {[optimized_circuit.find_bit(q).index for q in qargs]}')
            # add to summary, but check if it's a CX gate
            if gate_name == 'cx':
                # need to make sure control and target qubit appear at the same index for their respective qubits
                # figure out which is smaller and append 0s until the indices are the same
                if control_qubit_index < target_qubit_index:
                    optimized_summary[control_qubit_index] += [0] * (target_qubit_index - control_qubit_index)
                    control_qubit_index = target_qubit_index
                elif target_qubit_index < control_qubit_index:
                    optimized_summary[target_qubit_index] += [0] * (control_qubit_index - target_qubit_index)
                    target_qubit_index = control_qubit_index
                optimized_summary[control_qubit_index].append(int(f'5'))
                optimized_summary[target_qubit_index].append(int(f'6'))
            else:
                # the index in the GATE_SET is what we add to the summary
                # find index
                name_index = GATE_SET.index(gate_name)+1
                optimized_summary[qubit_index].append(name_index)

    # make sure sizes are correct
    print(f'optimized_summary before 7s: {optimized_summary}')
    
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
        
    # need to convert to vector of vector form
    optimized_summary_tensor = []
    for elem in optimized_summary_new:
        # create one hot vector
        one_hot = [0] * 8
        one_hot[elem] = 1
        one_hot = np.array(one_hot)
        optimized_summary_tensor.append(one_hot)

    # convert to tensor
    optimized_summary_tensor = np.array(optimized_summary_tensor)

    # return the circuit as matrix and summary
    matrix = Operator(optimized_circuit).data

    # if supplied with target, compute the norm of the difference
    if target is not None:
        # compute the norm of the difference
        norm = np.linalg.norm(matrix - target)
        # normalize by the dimension of the matrix
        norm /= matrix.shape[0]
        # return the norm
        return norm, optimized_summary_tensor

    # compute eigenvalues
    eigenvalues, _ = np.linalg.eig(matrix)
    # since inout matrix is unitary, only need to store the angles
    angles = np.angle(eigenvalues)
    # convert to all positive so angles only 0 to 2pi
    angles = (angles + 2*np.pi) % (2*np.pi)
    # normalize to 0 to 1
    angles /= (2*np.pi)

    # sort angles from smallest to largest
    angles = np.sort(angles)

    return angles, optimized_summary_tensor

## generate random circuit ##
def gen_random_circuit(N, depth, N2=None, plot=False):
    '''Returns random circuit with N qubits and depth.
    
    If N2 is None, then the circuit is on N qubits. Else, if we want to encode the circuit within a larger space, then N2 is the dimension of the larger space. 

    The circuit is returned as a tensor of shape (2**N2, 2**N2, 2) where the first two dimensions are the matrix representation of the circuit and the last dimension is the real and imaginary parts of the matrix.

    The circuit summary is returned as vector of length N2*max_depth. The gates are encoded as follows:
        0: no gate
        1: rx
        2: ry
        3: rz
        4: p
        5: cx control
        6: cx target
        7: end of qubit sequence

    Params:
        N: int, number of qubits
        depth: int, depth of circuit
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
                        
    if N2 is not None and N2 > N:
        # apply tensor product with identity to the rest of the qubits
        for i in range(N, N2):
            qc.id(i)

    optimized_circuit = transpile(qc, optimization_level=3, basis_gates=['rx', 'ry', 'rz', 'p', 'cx'])
    #                                                                      1,   2,    3,   4,    5,6
    
    if plot:
        qc.draw('mpl')
        # get timestamp to save figure
        if not os.path.exists('figures'):
            os.makedirs('figures')
        plt.savefig(os.path.join('figures', f'random_circuit_{N}_{depth}_{N2}_{time()}.png'))
        plt.show()
    
    return optimized_circuit

## decompose random circuit ##
def random_decompose(N, depth, N2=None, max_depth=None, target=None, plot=False):
    '''Runs gen_random_circuit and decompose_qc and returns the result.'''
    # generate random circuit
    optimized_circuit = gen_random_circuit(N, depth, N2=N2, plot=plot)
    # decompose
    return decompose_qc(optimized_circuit, N, max_depth=max_depth, N2=N2, target=target)

## for testing ##
def recompose_qc(summary_vec, N, N2=None, plot=False):
    '''Recomposes a quantum circuit from a summary vector.'''
    if N2 is None:
        N2 = N

    # first convert from vector of vectors to vector
    summary_vec = np.argmax(summary_vec, axis=1)

    # split into rows based on 7. find indices of 7
    indices = np.where(summary_vec == 7)[0]
    # split into rows
    rows = np.split(summary_vec, indices)
    # remove the 7s
    rows = [row[row != 7] for row in rows]
    # remove empty rows
    rows = [row for row in rows if len(row) > 0]
    # remove any rows of all 0
    rows = [row for row in rows if not np.all(row == 0)]
    # convert to list
    rows = list(rows)
    # convert to list of lists
    rows = [list(row) for row in rows]
    print(rows)
    

def get_single_rc(N=5, depth=10, N2=None, plot=True, save=False, show_summary=True):
    '''Returns a single random circuit with N qubits and depth encoded in N2 qubit space.'''
    if N2 is None:
        N2 = N
    circuit_tensor, summary = random_decompose(N, depth, plot=plot)
    if save:
        np.save(f'random_circuit_{N}_{depth}_{N2}.npy', circuit_tensor)
        np.save(f'random_summary_{N}_{depth}_{N2}.npy', summary)
    if show_summary:
        print(list(summary))
    
    # now recompose
    recompose_qc(summary, N, N2=N2, plot=plot)

##### build dataset #####
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

    print(f'Number of circuits to create: {len(Ns)}')

    # create dataset
    x = []
    y = []
    individual_y_len = N2*max_depth
    for i in trange(len(Ns)):
        N = Ns[i]
        # random depth, uniform distribution
        # choose the depth from 1 to max_depth
        depth = np.random.randint(1, max_depth)
        circuit_tensor, summary = gen_random_circuit(N, depth, N2=N2, max_depth=max_depth)
        if len(summary) != individual_y_len:
            print(f'not valid size summary vector. N: {N}, depth: {depth}, len(summary): {len(summary)}')
        else:
            x.append(circuit_tensor)
            y.append(summary)

    # make sure save directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # save dataset
    x = np.array(x)
    y = np.array(y)
    try:
        np.save(os.path.join('data', f'x_{N2}_{max_depth}_{total_num}.npy'), x)
    except ValueError:
        print('Error saving x dataset. Saving as list.')
        x = list(x)
        # save as pickle
        with open(os.path.join('data', f'x_{N2}_{max_depth}_{total_num}.pkl'), 'wb') as f:
            pickle.dump(x, f)
    try:
        np.save(os.path.join('data', f'y_{N2}_{max_depth}_{total_num}.npy'), y)
    except ValueError:
        print('Error saving y dataset. Saving as list.')
        y = list(y)
        # save as pickle
        with open(os.path.join('data', f'y_{N2}_{max_depth}_{total_num}.pkl'), 'wb') as f:
            pickle.dump(y, f)

    # note: to load pickle, use with open('my_list.pkl', 'rb') as file:
    #                               loaded_list = pickle.load(file)

if __name__ == '__main__':

    get_single_rc()

 