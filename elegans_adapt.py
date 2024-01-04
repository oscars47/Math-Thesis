# based on elegans_new.py, implements a learning circuit with adaptive architecture
import numpy as np
from functools import partial
from tqdm import trange
from oscars_toolbox.trabbit import trabbit

# ---- define universal gate set ----- #
gate_map = {
    'Rx': lambda theta: np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]]),
    'Ry': lambda theta: np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]]),
    'Rz': lambda theta: np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]]),
    'P': lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]]),
    'CNOT': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0 ,0, 1, 0]]),
}
I2 = np.eye(2)
base_sequence = ['Rx', 'Ry', 'Rz', 'P']

# ---- prepare random circuit ----- #
def random_circuit(N, depth, Rx_prob = 1/5, Ry_prob = 1/5, Rz_prob = 1/5, P_prob = 1/5, CNOT_prob = 1/5):
    '''Returns a random circuit for a given number N of qubits and depth and probabilities for each gate being applied to a qubit.'''

    # initialize probabilities
    p = np.array([Rx_prob, Ry_prob, Rz_prob, P_prob, CNOT_prob])
    p /= np.sum(p)

    circ = np.eye(2**N)

    # generate genes
    # get a list of lists of [gate, param] where gate is str and param is float or None if gate == CNOT
    genes = []
    for i in range(N): # iterate over qubits
        genes_i = []
        for _ in range(depth): # iterate over depth
            if i < N - 1:
                # get a random gate
                gate = np.random.choice(list(gate_map.keys()), p=p)
            else:
                # exclude CNOT for the last qubit
                gate = np.random.choice([g for g in gate_map.keys() if g != 'CNOT'])
            if gate in ['Rx', 'Ry', 'Rz', 'P']:
                param = np.random.uniform(0, 2*np.pi)
                genes_i.append([gate, param])
            elif gate == 'CNOT' and i < N - 1:
                genes_i.append([gate, None])
            else:
                raise ValueError(f"Unsupported gate type: {gate}")
        genes.append(genes_i)
            
    # print gates applied
    for i, gates in enumerate(genes):
        print(f'qubit {i}: {gates}')

    circ = create_circuit(N, genes)

    return circ, genes

# ---- convert circuit genes to circuit ----- #
def create_circuit(N, genes):
    '''Converts list of list of [gate, param] where gate is str and param is float or None if gate == CNOT and calculates the resulting unitary'''

    # apply random gates to each qubit for the given depth
    qc = np.eye(2**N)
    for i, gates in enumerate(genes):
        for j, gate in enumerate(gates):
            if gate[0] in ['Rx', 'Ry', 'Rz', 'P']:  # Parameterized gates
                if i == 0:
                    term = gate_map[gate[0]](gate[1])
                    for _ in range(N-1):
                        term = np.kron(term, I2)
                else:
                    term = I2
                    for _ in range(i-1):
                        term = np.kron(I2, term)
                    term = np.kron(term, gate_map[gate[0]](gate[1]))
                    for _ in range(N-i-1):
                        term = np.kron(term, I2)
            elif gate[0] == 'CNOT':
                if i == 0:
                    term = gate_map[gate[0]]
                    for _ in range(N-1):
                        term = np.kron(term, I2)
                else:
                    term = I2
                    for _ in range(i-1):
                        term = np.kron(I2, term)
                    term = np.kron(term, gate_map[gate[0]])
                    for _ in range(N-i-2):
                        term = np.kron(term, I2)
            else:
                raise ValueError(f"Unsupported gate type: {gate[0]}")
        # apply the gate to the circuit
        qc = term @ qc
    return qc

# ---- learning circuit ----- #
def base_learner(N, params):
    '''Only uses Rx Ry Rz P block for all of the qubits'''
    assert len(params) == 4 * N, f'Need 4N params, got {len(params)}'

    # reshape params
    params = np.reshape(params, (N, 4))

    # initialize circuit
    # get a list of lists of [gate, param] where gate is str and param is float or None if gate == CNOT
    genes = [[[gate, params[i][j]] for j, gate in enumerate(base_sequence)] for i in range(N)]
    qc = create_circuit(N, genes)
    return qc

def random_angles(N, model=0):
    '''Returns a random circuit with 4N params.'''
    if model==0:
        return np.random.uniform(0, 2*np.pi, size=(4 * N))
    
# ----- learning ----- #
def loss(params, circ_func, target, N):
    '''Returns the loss between the circuit with the given params and the target matrix.'''
    return np.linalg.norm(circ_func(N, params) - target)

def find_params(target, model=0):
    '''Finds the params that minimize the loss between the circuit with the given params and the target matrix.

    Params:
        :target: the target matrix
        :model: whether to use the Rx Ry Rz P block (0) or just the Rx Ry Rz block (1)
    '''

    N = int(np.log2(target.shape[0]))

    if model==0:
        random_func = partial(random_angles, N=N, model=model)
        loss_func = partial(loss, circ_func = base_learner, target=target, N=N)

    # minimize the loss
    x_best, loss_best = trabbit(loss_func, random_func, alpha=0.8, temperature=.1, num=50, tol=1e-4)
    print(f'loss in find_params: {loss_best}')
    return x_best

# ------ sample circuits ----- #
def sample_circuits(N, depth, model=0):
    '''Returns a list of num random circuits with N qubits and depth depth.'''
    if model == 0:

        params = np.random.uniform(0, 2*np.pi, size=(N, depth))
        gates = np.random.choice(base_sequence, size=(N, depth))
        genes = []
        for i in range(N):
            genes_i = []
            for j in range(depth):
                # Ensure each element in genes_i is a pair [gate, parameter]
                genes_i.append([gates[i][j], params[i][j]])
            genes.append(genes_i)

        genes[0].append(['CNOT', None])

        params = np.random.uniform(0, 2*np.pi, size=(N, depth))
        gates = np.random.choice(base_sequence, size=(N, depth))
        for i in range(N):
            genes_i = []
            for j in range(depth):
                # Ensure each element in genes_i is a pair [gate, parameter]
                genes_i.append([gates[i][j], params[i][j]])
            genes.append(genes_i)

        print(genes)




    for j, gates in enumerate(genes):
        print(f'qubit {j}: {gates}')

    return create_circuit(N, genes)

if __name__ == '__main__':
    from elegans_new import print_matrix
    num_qubits = 3
    depth = 10

    target = sample_circuits(num_qubits, depth)
    print_matrix(target)

   
    # target, genes = random_circuit(num_qubits, depth)
    # params = find_params(target)

   

#    for j, gate in enumerate(gates):
#             if gate[0] in ['Rx', 'Ry', 'Rz', 'P']:  # Parameterized gates
#                 term = gate_map[gate[0]](gate[1])
#                 if i == 0:
#                     term = np.kron(term, np.eye(2**(N-1)))
#                 elif i == N-1:
#                     term = np.kron(np.eye(2**(N-1)), term)
#                 else:
#                     term = np.kron(np.eye(2**i), term)
#                     term = np.kron(term, np.eye(2**(N-i-1)))
#             elif gate[0] == 'CNOT':
#                 if i == 0:
#                     term = gate_map['CNOT']
#                     term = np.kron(term, np.eye(2**(N-2)))
#                 elif i == N-2:
#                     term = gate_map['CNOT']
#                     term = np.kron(np.eye(2**(N-2)), term)
#                 else:
#                     term = gate_map['CNOT']
#                     term = np.kron(np.eye(2**i), term)
#                     term = np.kron(term, np.eye(2**(N-i-2)))
#             else:
#                 raise ValueError(f"Unsupported gate type: {gate[0]}")