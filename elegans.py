# file to implement the q-elegans model as a circuit which should be able to arbitrarily approximate any circuit

import numpy as np
from oscars_toolbox.trabbit import trabbit

# ------ define gates ------ #
# jones matrices for single qubit rotation; see Simon and Mukunda that show HQQ = SU(2)
def R(alpha): return np.matrix([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def H(theta): return np.matrix([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
def Q(alpha): return R(alpha) @ np.matrix(np.diag([np.exp(np.pi / 4 * 1j), np.exp(-np.pi / 4 * 1j)])) @ R(-alpha)
def Rz(theta): return np.matrix(np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]))
# pauli x, identity, and hadamard
X = np.matrix([[0, 1], [1, 0]])
I2 = np.eye(2)
Had = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
# function for CNOT
def CNOT(N, i, j, theta):
    '''Returns the CNOT gate on N qubits with control qubit i and target qubit j.

    Params:
        N: number of qubits
        i: control qubit
        j: target qubit
        theta1: angle of rotation for H gate
        theta2: angle of rotation for Q gate
    
    '''

    assert i < j, 'This construction requires i < j'

    # define projection operators: |0><0|, |1><1|
    P0 = np.array([[1, 0], [0, 0]])
    P1 = np.array([[0, 0], [0, 1]])

    # initialize gate
    gate = 1
    # loop over all qubits: |0>
    for k in range(N):
        if k == i:
            # control
            gate = np.kron(gate, P0)
        else:
            # add identity for non-control and non-target qubits
            gate = np.kron(gate, I2)

    # second part of the gate for when control qubit is |1>
    gate1 = 1
    for k in range(N):
        if k == i:
            gate1 = np.kron(gate1, P1)
        elif k == j: # target part
            # gate1 = np.kron(gate1, (H(theta1/2) @ Q(theta2/2) @ Q(theta3/2)) @ X @ (H(-theta1/2) @ Q(-theta2/2) @ Q(-theta3/2))) # if control is |0>, then target is identity
            gate1 = np.kron(gate1, Rz(theta/2) @ X @ Rz(-theta/2))
        else:
            gate1 = np.kron(gate1, I2)

    # combine the two parts
    gate = gate + gate1
    return gate

# ------ define circuit ------ #
def get_circuit(n):
    '''Returns a parametrized circuit that can approximate any unitary on n qubits.'''
    def circuit_func(params):
        # first apply Had to all qubits
        circuit = Had
        for _ in range(n-1):
            circuit = np.kron(circuit, Had)

        # apply CNOT gates on all pairs of qubits
        CNOT_params = params[:3*n**2].reshape((n, n))
        # loop through all qubits as control
        for i in range(n):
            # loop through all qubits as target
            for j in range(n):
                # skip if control and target are the same
                if i == j:
                    continue
                # apply CNOT, which in general is defined: CNOT = |0><0| x I + |1><1| x X. we extend it 
                if i < j:
                    circuit = circuit @ CNOT(n, i, j, CNOT_params[i, j])
                else:
                    circuit = circuit @ CNOT(n, j, i, CNOT_params[i, j])

        # apply HQQ gates on all qubits
        HQQ_params = params[3*n**2:].reshape((n, 3))
        # loop through all qubits
        for i in range(n):
            # apply HQQ
            circuit = np.kron(circuit, H(HQQ_params[i, 0]) @ Q(HQQ_params[i, 1]) @ Q(HQQ_params[i, 2]))

    return circuit_func

# ------ define cost function ------ #
def loss_func(params, target):
    '''Returns the loss function for a given target unitary.'''
    circuit_func = get_circuit(int(np.log2(len(target))))
    return np.linalg.norm(circuit_func(params) - target)

def random(): return np.random.uniform(0, 2*np.pi)

def find_params(target):
    '''Returns the parameters that minimize the loss function for a given target unitary.'''
    # initialize parameters
    n = int(np.log2(len(target)))
    bounds = [(0, 2*np.pi)] * (3*n**2 + 3*n)
    # minimize loss function
    loss_func_param = lambda params: loss_func(params, target)
    return trabbit(loss_func=loss_func_param, random_gen=random, bounds=bounds)

# ------ test ------ #
def random_circuit(n, d, I2_prob = 0.2, H_prob = 0.2, Q_prob = 0.2, HQQ_prob = 0.2, CNOT_prob = 0.2):
    '''Returns a random quantum cictui on n qubits for a depth d.'''
    # initialize random unitary
    unitary = Had
    for _ in range(n-1):
        unitary = np.kron(unitary, Had)

    # loop through all layers
    for k in range(d):
        # choose which gates to apply; if CNOT, chose random control and target qubits
        gates = np.random.choice(['I2', 'H', 'Q', 'HQQ', 'CNOT'], size=n, p=[I2_prob, H_prob, Q_prob, HQQ_prob, CNOT_prob])
        term = 1
        for gate in gates:
            if gate == 'I2':
                term = np.kron(term, I2)
            elif gate == 'H':
                term = np.kron(term, Had)
            elif gate == 'Q':
                term = np.kron(term, Q(random()))
            elif gate == 'HQQ':
                term = np.kron(term, H(random()) @ Q(random()) @ Q(random()))
            elif gate == 'CNOT':
                i = np.random.choice(n, size=1, replace=False)
                j = k
                if j > i:
                    i, j = j, i
                term = term @ CNOT(n, i, j, random(), random(), random())
        # apply term to unitary
        unitary = unitary @ term

    return unitary

if __name__ == '__main__':
    N = 2
    print(np.round(CNOT(N, 0, 1, 0, np.pi/2, np.pi/2), 3))




    
