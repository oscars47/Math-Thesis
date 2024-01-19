# file to generate matrices for sphynx to train on
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import matplotlib.pyplot as plt

def gen_random_circuit(N, depth, N2=None):
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
                gate = np.random.choice(['rx', 'ry', 'rz', 'p', 'cx'], p=p)
            else:
                gate = np.random.choice(['rx', 'ry', 'rz', 'p'], p=p_new)
            if gate in ['rx', 'ry', 'rz', 'p'] and circ_summary[i, j] == 0:
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

    # return the circuit as matrix and summary
    matrix = Operator(qc).data
    real_matrix = np.real(matrix)
    imag_matrix = np.imag(matrix)
    # combine into one tensor
    tensor = np.stack((real_matrix, imag_matrix), axis=-1)
    return tensor, circ_summary

if __name__ == '__main__':
    N = 10
    depth = 20
    # N2 = 10
    N2 = N
    matrix, summary = gen_random_circuit(N, depth, N2)
    np.save(f'tensor{N}_{depth}_{N2}.npy', matrix)
    np.save(f'summary_{N}_{depth}_{N2}.npy', summary)