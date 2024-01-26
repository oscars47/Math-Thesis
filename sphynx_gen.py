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
def decompose_qc(optimized_circuit, N, max_depth, N2=None, target=None):
    '''Decomposes the optimized circuit into normalized vector of angles and summary vector representing the circuit

    Params:
        optimized_circuit: QuantumCircuit, optimized circuit
        N: int, number of qubits
        max_depth: int, maximum depth of circuit
        N2: int, dimension of hilbert space
        target: np.array, (optional) target matrix to compute norm of difference

    Returns:
        angles, optimized_summary_tensor
    '''

    if N2 is None:
        N2 = N

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
                optimized_summary[control_qubit_index].append(target_qubit_index+len(GATE_SET))
            else:
                # the index in the GATE_SET is what we add to the summary
                # find index
                name_index = GATE_SET.index(gate_name)+1
                optimized_summary[qubit_index].append(name_index)
    
    # convert the optimized summary to a single list of sequences of length max_depth
    optimized_summary_new = []
    for row in optimized_summary:
        # if length is not max_depth, append 0s
        if len(row) < max_depth:
            row += [0] * (max_depth - len(row))
        optimized_summary_new += row

    num_total_classes = len(GATE_SET) + N2 + 1
        
    # need to convert to vector of vector form
    optimized_summary_tensor = []
    for elem in optimized_summary_new:
        # create one hot vector
        one_hot = [0] * num_total_classes
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

    # print('optimized summary\n', optimized_summary)

    return angles, optimized_summary_tensor

## generate random circuit ##
def gen_random_circuit(N, max_depth, N2=None, plot=False, check_fidelity=False):
    '''Returns random circuit with N qubits and depth.
    
    If N2 is None, then the circuit is on N qubits. Else, if we want to encode the circuit within a larger space, then N2 is the dimension of the larger space. 

    The circuit is returned as a tensor of shape (2**N2, 2**N2, 2) where the first two dimensions are the matrix representation of the circuit and the last dimension is the real and imaginary parts of the matrix.

    The circuit summary is returned as vector of length N2*max_depth. The gates are encoded as follows:
        0: no gate
        1: rx
        2: ry
        3: rz
        4: p
        5+: cx, where the target qubit is the index of the gate minus 5


    Params:
        N: int, number of qubits
        max_depth: int, depth of circuit
        N2: int, dimension of total hilbert space
        plot: bool, whether to plot the circuit
        check_fidelity: bool, whether to check the fidelity of transpiled circuit against actual
    '''

    # Create a quantum circuit with the appropriate number of qubits
    if N2 is None:
        qc = QuantumCircuit(N)
        N2 = N
    else:
        qc = QuantumCircuit(N2)

    # random simplex probability to choose gates
    p = np.random.dirichlet(np.ones(len(GATE_SET)))
    p_new = p.copy()[:-1]
    p_new /= np.sum(p_new)

    # go through and apply random gates
    for i in range(N):
        for j in range(max_depth):
            # choose random set of gate
            gate = np.random.choice(GATE_SET, p=p)
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
                if len(eligible_elements)>0:
                    target = np.random.choice(eligible_elements)
                    
                    # apply gate
                    qc.cx(i, target)
                else:
                    # choose random gate from RP
                    gate = np.random.choice(GATE_SET_RP, p=p_new)
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

    optimized_circuit = transpile(qc, optimization_level=3, basis_gates=GATE_SET)
    #                                                                      1,   2,    3,   4,    5,6
    
    if check_fidelity:
        # get the matrix representation of the circuit
        matrix = Operator(optimized_circuit).data
        # get the matrix representation of the target
        target = Operator(qc).data
        # compute the norm of the difference
        norm = np.linalg.norm(matrix - target)
        # normalize by the dimension of the matrix
        norm /= matrix.shape[0]
        # print the norm
        if plot:
            print('norm of difference:', norm)

    if plot:
        optimized_circuit.draw('mpl')
        # get timestamp to save figure
        if not os.path.exists('figures'):
            os.makedirs('figures')
        plt.savefig(os.path.join('figures', f'random_circuit_{N}_{max_depth}_{N2}_{time()}.png'))
        plt.show()
    if not check_fidelity:
        return optimized_circuit
    else:
        return optimized_circuit, norm

## decompose random circuit ##
def random_decompose(N, max_depth, N2=None, target=None, plot=False):
    '''Runs gen_random_circuit and decompose_qc and returns the result.'''
    # generate random circuit
    optimized_circuit = gen_random_circuit(N, max_depth=max_depth, N2=N2, plot=plot)
    # decompose
    angles, summary = decompose_qc(optimized_circuit, N, max_depth=max_depth, N2=N2, target=target)
    if N2 is None:
        N2 = N
    while len(summary) != N2*max_depth:
        # print('not valid size summary vector. N: {N}, depth: {max_depth}, len(summary): {len(summary)}')
        optimized_circuit = gen_random_circuit(N, max_depth=max_depth, N2=N2, plot=plot)
        angles, summary = decompose_qc(optimized_circuit, N, max_depth=max_depth, N2=N2, target=target)
    return angles, summary

## for testing ##
def recompose_qc(summary_vec, N, N2=None, max_depth = 100, plot=False):
    '''Recomposes a quantum circuit from a summary vector.'''
    if N2 is None:
        N2 = N

    # first convert from vector of vectors to vector
    summary_vec = np.argmax(summary_vec, axis=1)

    # convert to list of lists, splitting based on max_depth
    summary = []
    for i in range(0, len(summary_vec), max_depth):
        summary.append(summary_vec[i:i+max_depth])

    print('reconstructed summary\n', summary)

    # now iterate through the summary and apply gates to the circuit
    qc = QuantumCircuit(N2)

    for i in range(N2):
        for j in range(max_depth):
            gate = summary[i][j]
            if gate == 0:
                pass
            elif gate == 1:
                qc.rx(0, i)
            elif gate == 2:
                qc.ry(0, i)
            elif gate == 3:
                qc.rz(0, i)
            elif gate == 4:
                qc.p(0, i)
            elif gate > 4: # this is the target qubit for a CX gate
                qc.cx(i, gate-len(GATE_SET))
            
    if plot:
        qc.draw('mpl')
        if not os.path.exists('figures'):
            os.makedirs('figures')
        plt.savefig(os.path.join('figures', f'reconstruction_{N}_{max_depth}_{N2}_{time()}.png'))
        plt.show()
    
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

def plot_eigenvalues(num = 100, N=5, max_depth=10, N2=None):
    '''Plots histogram of eigenvalues of random circuits.'''
    if N2 is None:
        N2 = N

    eigenvalues = []
    for i in trange(num):
        # generate random circuit
        optimized_circuit = gen_random_circuit(N, max_depth=max_depth, N2=N2)
        # compute eigenvalues
        eigenvalues += list(np.linalg.eig(Operator(optimized_circuit).data)[0])

    plt.hist(eigenvalues, bins=100)
    plt.savefig(f'eigenvalues_{N}_{max_depth}_{N2}.png')
    plt.show()

def benchmark_transpile(num = 100, N0=3, Nf = 10, min_max_depth=10, max_max_depth = 100, depth_steps = 10, N2=None):
    '''Rigorously compute the mean and sem of transpilation accuracy as defined as the norm of the difference between the transpiled and original circuit.'''
    # for all N in N0 to Nf, for all depths in min_max_depth to max_max_depth, compute the mean and sem of the norm of the difference between the transpiled and original circuit
    # store in a dictionary
    # key: (N, depth)
    # value: (mean, sem)
    results = {}

    for N in range(N0, Nf+1):
        for depth in range(min_max_depth, max_max_depth+1, depth_steps):
            # create list of norms
            norms = []
            for i in trange(num):
                # generate random circuit
                if N2 is None:
                    _, norm = gen_random_circuit(N, depth, check_fidelity=True)
                else:
                    _, norm = gen_random_circuit(N, depth, N2=N2, check_fidelity=True)
                # compute eigenvalues
                norms.append(norm)

            # compute mean and sem
            mean = np.mean(norms)
            sem = np.std(norms) / np.sqrt(len(norms))

            print(f'N: {N}, depth: {depth}, mean: {mean}, sem: {sem}')

            # store in dictionary
            results[(N, depth)] = (mean, sem)

    # save results
    if not os.path.exists('results'):
        os.makedirs('results')
    np.save(os.path.join('results', f'benchmark_transpile_{N0}_{Nf}_{min_max_depth}_{max_max_depth}_{depth_steps}_{N2}.npy'), results)

def plot_benchmark_transpile(results_dict):
    '''Reads in dictionary from benchmark_transpile and plots the results.'''
    # plot the results
    # x axis: number of qubits
    # y axis: depth
    # z axis: mean of norm of difference
    # color: sem of norm of difference

    # get the keys
    keys = list(results_dict.keys())
    # get the values
    values = list(results_dict.values())
    # get the mean and sem
    means = [x[0] for x in values]
    sems = [x[1] for x in values]

    # get the Ns and depths
    Ns = [x[0] for x in keys]
    depths = [x[1] for x in keys]

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Ns, depths, means, c=sems, cmap='viridis')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Mean of Norm of Difference')
    plt.show()

##### build dataset #####
def build_dataset(total_num, N2=10, N0=3, max_depth=100):
    '''Builds dataset of total_num random circuits with randomized N qubits in N2 qubit hilbert space and random depth.

    Params:
        total_num: int, total number of circuits to generate
        N2: int, dimension of hilbert space
        N0: int, minimum number of qubits in circuit
    
    
    '''
    if N2 - N0 > 0:
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
            angles, summary_tensor = random_decompose(N, N2=N2, max_depth=max_depth)
            if len(summary_tensor) != individual_y_len:
                print(f'not valid size summary vector. N: {N}, depth: {depth}, len(summary): {len(summary_tensor)}')
            else:
                x.append(angles)
                y.append(summary_tensor)

    else:
        print('doing single!')
        x = []
        y = []
        N = N2
        individual_y_len = N*max_depth
        for i in trange(total_num):
            # random depth, uniform distribution
            # choose the depth from 1 to max_depth
            # depth = np.random.randint(1, max_depth)
            angles, summary_tensor = random_decompose(N, max_depth=max_depth)
            if len(summary_tensor) != individual_y_len:
                print(f'not valid size summary vector. N: {N}, depth: {max_depth}, len(summary): {len(summary_tensor)}')
            else:
                x.append(angles)
                y.append(summary_tensor)


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
    # N = 3
    # max_depth = 20
    # build_dataset(100000, N0=N, N2=N, max_depth=max_depth)
    benchmark_transpile(num=1000)
    