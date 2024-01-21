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

GATE_SET = ['rx', 'ry', 'rz', 'p', 'cx']
GATE_SET_RP = ['rx', 'ry', 'rz', 'p']

def gen_random_circuit(N, depth, N2=None, check_fidelity=False):
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
        N2: int, dimension of hilbert space
        check_fidelity: bool, whether to check the fidelity of the tucker decomposition
    
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

    # circuit summary
    circ_summary = np.zeros((N2, depth))
    
    # go through and apply random gates

    # determine the set of gates for all qubits
    for i in range(N):
        for j in range(depth):
            # choose random set of gate
            if j < depth - 1:
                gate = np.random.choice(GATE_SET, p=p)
            else:
                gate = np.random.choice(GATE_SET_RP, p=p_new)
            if gate in GATE_SET_RP and circ_summary[i, j] == 0:
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
                    # add to summary
                    circ_summary[i, j] += 2
                elif gate == 'rz':
                    # choose random angle
                    angle = np.random.uniform(0, 2*np.pi)
                    # apply gate
                    qc.rz(angle, i)
                    # add to summary
                    circ_summary[i, j] += 3
                elif gate == 'p':
                    # choose random angle
                    angle = np.random.uniform(0, 2*np.pi)
                    # apply gate
                    qc.p(angle, i)
                    # add to summary
                    circ_summary[i, j] += 4
            elif gate == 'cx' and circ_summary[i, j] == 0:
                # choose random target not equal to itself and hasn't been used
                # create a list of eligible elements
                eligible_elements = [x for x in range(N) if x != i and circ_summary[x, j] == 0]

                # check if the list is empty
                if eligible_elements:
                    target = np.random.choice(eligible_elements)
                    
                    # apply gate
                    qc.cx(i, target)
                    # add to summary
                    circ_summary[i, j] += int(f'5{target}')
                    circ_summary[target, j] += int(f'6{i}')
                else:
                    # choose random gate from RP
                    gate = np.random.choice(['rx', 'ry', 'rz', 'p'], p=p_new)
                    # apply gate
                    if gate == 'rx':
                        # choose random angle
                        angle = np.random.uniform(0, 2*np.pi)
                        # apply gate
                        qc.rx(angle, i)
                        # add to summary
                        circ_summary[i, j] += 1
                    elif gate == 'ry':
                        # choose random angle
                        angle = np.random.uniform(0, 2*np.pi)
                        # apply gate
                        qc.ry(angle, i)
                        # add to summary
                        circ_summary[i, j] += 2
                    elif gate == 'rz':
                        # choose random angle
                        angle = np.random.uniform(0, 2*np.pi)
                        # apply gate
                        qc.rz(angle, i)
                        # add to summary
                        circ_summary[i, j] += 3
                    elif gate == 'p':
                        # choose random angle
                        angle = np.random.uniform(0, 2*np.pi)
                        # apply gate
                        qc.p(angle, i)
                        # add to summary
                        circ_summary[i, j] += 4
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
                optimized_summary[control_qubit_index].append(int(f'5{target_qubit_index}'))
                optimized_summary[target_qubit_index].append(int(f'6{control_qubit_index}'))
            else:
                # the index in the GATE_SET is what we add to the summary
                # find index
                name_index = GATE_SET.index(gate_name)
                optimized_summary[qubit_index].append(name_index)
    
    
    # return the circuit as matrix and summary
    matrix = Operator(qc).data
    real_matrix = np.real(matrix)
    imag_matrix = np.imag(matrix)

    n = int(np.log(matrix.shape[0]*matrix.shape[1]) / np.log(2))
    ranks = [2,]*n     
   
    # use tucker decomposition to reduce the size of the matrix
    # convert the matrix to a tensor
    tensor_real = real_matrix.reshape(ranks)
    # Perform Tucker Decomposition
    
    print(f'Rank of decomposition: {ranks}')
    core_real, factors_real = tucker(tensor_real, rank=ranks)

    # convert the matrix to a tensor
    tensor_imag = imag_matrix.reshape(ranks)

    # Perform Tucker Decomposition
    core_imag, factors_imag = tucker(tensor_imag, rank=ranks)

    if check_fidelity: # compute the fidelity of the decomposition
        # reconstruct the matrix using multi mode product
        reconstr_real = tl.tenalg.multi_mode_dot(core_real, factors_real, modes=list(range(n)))
        reconstr_real = reconstr_real.reshape(matrix.shape[0], matrix.shape[1])

        reconstr_imag = tl.tenalg.multi_mode_dot(core_imag, factors_imag, modes=list(range(n)))
        reconstr_imag = reconstr_imag.reshape(matrix.shape[0], matrix.shape[1])

        # compute the fidelity
        fidelity_real = np.linalg.norm(np.array(reconstr_real) - real_matrix)
        fidelity_imag = np.linalg.norm(np.array(reconstr_imag) - imag_matrix)

        print(f'Fidelity of real part: {fidelity_real}')
        print(f'Fidelity of imaginary part: {fidelity_imag}')

    return core_real, factors_real, core_imag, factors_imag, optimized_summary

def build_dataset(total_num):
    '''Builds dataset of total_num random circuits with randomized N qubits in N2 qubit hilbert space and random depth.'''

    # even spacing of N up to N2=10
    # random depth
    pass

def test():
    # Example: Create a sample circuit
    qc = QuantumCircuit(5)
    qc.cx(2,3)
    qc.rx(0.5, 0)
    qc.ry(0.5, 0)
    qc.ry(0.6, 1)
    qc.rz(0.7, 2)
    qc.p(0.8, 1)
    qc.cx(0, 2)
    qc.cx(1, 3)
    # qc.measure_all()

    # Iterate through the circuit data
    for instruction, qargs, _ in qc.data:
        gate = instruction
        gate_name = gate.name

        # Debugging: Print all gates
        print(f"Gate: {gate_name}")

        # Check for RX, RY, RZ, P, or CX (CNOT) gates
        if gate_name in ['rx', 'ry', 'rz', 'p', 'cx']:
            # For CX (CNOT) gates, identify control and target qubits using find_bit
            if gate_name == 'cx':
                control_qubit_index = qc.find_bit(qargs[0]).index
                target_qubit_index = qc.find_bit(qargs[1]).index
                print(f" CNOT Gate - Control Qubit: {control_qubit_index}, Target Qubit: {target_qubit_index}")
            else:
                qubit_indices = [qc.find_bit(q).index for q in qargs]
                print(f" Qubits: {qubit_indices}")



if __name__ == '__main__':
    N = 10
    depth = 100
    # N2 = 10
    N2 = N
    core_real, factors_real, core_imag, factors_imag, summary = gen_random_circuit(N, depth, N2, check_fidelity=True)

    # print(len(factors_imag))

    # Store decomposed components
    # with h5py.File('tucker_decomposition_{N}_{depth}.h5', 'w') as f:
        # Storing the core tensor
        # f.create_dataset('core_real', data=core_real)
        # f.create_dataset('core_imag', data=core_imag)
        
        # Storing each factor matrix
        # for i, factor in enumerate(factors_real):
        #     f.create_dataset(f'factors_real_{i}', data=factor)

        # for i, factor in enumerate(factors_imag):
        #     f.create_dataset(f'factors_imag_{i}', data=factor)

    # np.savez_compressed('tucker_{N}_{depth}_{N2}.npz', core_real=core_real, factors_real=factors_real, core_imag=core_imag, factors_imag=factors_imag)
    # print(core_real)
    # print(core_real.shape)
    # np.savez_compressed(f'core_real_{N}_{N2}_{depth}.npy', core_real)
    # what if we save each matrix in the core as a separate file?
    # iterate through each matrix in the core; that is, through N2 different for loops
    
    # Function to iterate and save 2D slices
    def save_2d_slices(core_tensor, prefix="slice"):
        '''DOESN'T WORK'''
        # Create a directory to save the slices
        os.makedirs(f"tensor_slices", exist_ok=True)

        # Iterate through each mode of the tensor
        for mode in range(len(core_tensor.shape)):
            # Iterate through each slice in this mode
            for index in range(core_tensor.shape[mode]):
                # Extract the 2D slice
                slice_indices = [slice(None)] * len(core_tensor.shape)
                slice_indices[mode] = index
                matrix_slice = core_tensor[tuple(slice_indices)]

                # Save the slice
                filename = f"{prefix}_mode{mode}_index{index}.npy"
                filepath = os.path.join("tensor_slices", filename)
                np.save(filepath, matrix_slice)

 