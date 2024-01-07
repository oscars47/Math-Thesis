# based on elegans_new.py, implements a learning circuit with adaptive architecture
import numpy as np
from functools import partial
from tqdm import trange
from oscars_toolbox.trabbit import trabbit

# ---- define universal gate set ----- #
gate_map = {
    'Rx': lambda theta: np.array([[np.cos(theta), -1j * np.sin(theta)], [-1j * np.sin(theta), np.cos(theta)]]),
    'Ry': lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
    'Rz': lambda theta: np.array([[np.exp(-1j * theta), 0], [0, np.exp(1j * theta)]]),
    'P': lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]]),
    'CNOT': lambda theta: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta), np.sin(theta)], [0 ,0, np.sin(theta), np.cos(theta)]]),
}
I2 = np.eye(2)
RP_sequence = ['Rx', 'Ry', 'Rz', 'P']

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
                genes_i.append([gate, np.pi/2])
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

    # remove any CNOTs if placed in the last qubit; if so, remove
    genes_last = []
    for i, gates in enumerate(genes[-1]): # check list for last qubit
        if gates[0] != 'CNOT':
            genes_last.append(gates)
    genes[-1] = genes_last

    qc = np.eye(2**N)
    for i, gates in enumerate(genes):
        # apply random gates to each qubit for the given depth
        for j, gate in enumerate(gates):
            if gate[0] in ['Rx', 'Ry', 'Rz', 'P']:  # Parameterized gates
                term = gate_map[gate[0]](gate[1])
                if i == 0:
                    term = np.kron(term, np.eye(2**(N-1)))
                elif i == N-1:
                    term = np.kron(np.eye(2**(N-1)), term)
                else:
                    term = np.kron(np.eye(2**i), term)
                    term = np.kron(term, np.eye(2**(N-i-1)))
            elif gate[0] == 'CNOT':
                term = gate_map[gate[0]](gate[1])
                if i == 0:
                    term = np.kron(term, np.eye(2**(N-2)))
                elif i == N-2:
                    term = np.kron(np.eye(2**(N-2)), term)
                else:
                    term = np.kron(np.eye(2**i), term)
                    term = np.kron(term, np.eye(2**(N-i-2)))
            else:
                raise ValueError(f"Unsupported gate type: {gate[0]}")
            # apply the gate to the circuit
            qc = term @ qc
    return qc

# ---- learning circuit ----- #
def learner(N, params, base_sequence=RP_sequence):
    '''Requires list of gates to be inserted for all of the qubits'''

    # reshape params
    params = np.reshape(params, (N, len(RP_sequence)))

    # initialize circuit
    # get a list of lists of [gate, param] where gate is str and param is float or None if gate == CNOT
    genes = [[[gate, params[i][j]] for j, gate in enumerate(base_sequence)] for i in range(N)]
    qc = create_circuit(N, genes)
    return qc

def random_angles(N, depth):
    '''Returns a random circuit with 4N params.'''
    return np.random.uniform(0, 2*np.pi, size=(depth*N))
    
# ----- learning ----- #
def loss(params, circ_func, target, N):
    '''Returns the loss between the circuit with the given params and the target matrix.'''
    return np.linalg.norm(circ_func(N, params) - target)

def find_params(target, model=0, threshold=1e-4):
    '''Finds the params that minimize the loss between the circuit with the given params and the target matrix.

    Params:
        :target: the target matrix
        :model: whether to use the Rx Ry Rz P block (0) or adapt using parametrized CNOT gates (1)
    '''
    def run(base_sequence=RP_sequence):
        '''Runs the learning algorithm with the given base sequence.'''
        cirq_func = partial(learner, base_sequence=base_sequence)
        random_func = partial(random_angles, N=N, depth=len(base_sequence))
        loss_func = partial(loss, circ_func = cirq_func, target=target, N=N)

        # minimize the loss
        x_best, loss_best = trabbit(loss_func, random_func, alpha=0.8, temperature=.1, num=1, tol=1e-4)
        return x_best, loss_best

    N = int(np.log2(target.shape[0]))

    if model==0:
        x_best, loss_best = run()
        print(f'loss in find_params: {loss_best}')
        return x_best, loss_best
        
    elif model == 1: # adaptive
        # initialize circuit
        base_sequence = ['Rx']
        x_best, loss_best = run(base_sequence=base_sequence) # first try only Rx Ry Rz P
        print(f'loss after RP run: {loss_best}')
        if loss_best > threshold: # if not sufficiently small, continue
             # need to add CNOTp layer
            base_sequence += ['CNOT']
            x_best, loss_best = run(base_sequence=base_sequence)
            print(f'loss after CNOT run: {loss_best}')
            c = 1 # add counter
            while loss_best > threshold and c < 10: # add more CNOTp layers
                base_sequence += RP_sequence
                x_best, loss_best = run(base_sequence=base_sequence)
                print(f'loss for {c} adding RP after CNOT run: {loss_best}')
                print(f'len of base_sequence: {len(base_sequence)}')
                if loss_best < threshold:
                    break

                base_sequence +=  ['CNOT']# add another CNOT layer
                x_best, loss_best = run(base_sequence=base_sequence)
                print(f'loss for {c} adding additional CNOT: {loss_best}')
                if loss_best < threshold:
                    break
                print(f'len of base_sequence: {len(base_sequence)}')
                c += 1
        return x_best, loss_best

# ------ sample circuits ----- #
def sample_circuits(N, depth, config=0):
    '''Returns a list of num random circuits with N qubits and depth depth.'''
    params = np.random.uniform(0, 2*np.pi, size=(N, depth))
    gates = np.random.choice(RP_sequence, size=(N, depth))
    genes = []
    for i in range(N):
        genes_i = []
        for j in range(depth):
            # Ensure each element in genes_i is a pair [gate, parameter]
            genes_i.append([gates[i][j], params[i][j]])
        genes.append(genes_i)
    if config==0 or config==1:
        genes[0].append(['CNOT', np.pi/2])

    params = np.random.uniform(0, 2*np.pi, size=(N, depth))
    gates = np.random.choice(RP_sequence, size=(N, depth))
    for i in range(N):
        genes_i = []
        for j in range(depth):
            # Ensure each element in genes_i is a pair [gate, parameter]
            genes_i.append([gates[i][j], params[i][j]])
        genes[i].extend(genes_i)

    if config == 1:
        genes[0].append(['CNOT', np.pi/2])

    for j, gates in enumerate(genes):
        print(f'qubit {j}: {gates}')

    return create_circuit(N, genes)

# ------ rigorous testing ------ #
def benchmark(N, depth, gen_func, reps=20):
    '''Returns the avg and sem of loss of the model over reps num of trials.'''
    loss_list = []
    for _ in trange(reps):
        # generate random target
        target = gen_func(N, depth)

        # find params
        params, loss_best = find_params(target)
        loss_list.append(loss_best)
    print(f'loss: {np.mean(loss_list)} ± {np.std(loss_list)/np.sqrt(reps)}')
    return np.mean(loss_list), np.std(loss_list)/np.sqrt(reps)

if __name__ == '__main__':
    # from elegans_new import print_matrix

    # initialize circuit params #
    num_qubits = 3
    depth = 5

    # define parametrized circuit functions # 
    # sample_cirq_func0 = partial(sample_circuits, config=0)
    # sample_cirq_func1 = partial(sample_circuits, config=1)

    # sample0 = sample_circuits(num_qubits, depth, config=0)
    # target = random_circuit(num_qubits, depth, CNOT_prob=0)[0]
    # find_params(target, model=0)
    
    # benchmark #
    # benchmark(num_qubits, depth, sample_cirq_func0, reps=20)
    # benchmark(num_qubits, depth, sample_cirq_func1, reps=20)

    # print(create_circuit(N=num_qubits, genes = [[['Rx', 0.1], ['Ry', 0.2], ['Rz', 0.3], ['P', 0.4], ['CNOT', np.pi/2]], [['Rx', 0.5], ['Ry', 0.6], ['Rz', 0.7], ['P', 0.8], ['CNOT', np.pi/2]], [['Rx', 0.9], ['Ry', 1.0], ['Rz', 1.1], ['P', 1.2]]]))
    # print(gate_map['CNOT'](np.pi/2))

    base_sequence = ['Rx', 'P']
    

    ex = learner()

    