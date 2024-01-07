# redo of elegans_adapt.py fixing the issues with genes
import numpy as np
from scipy.linalg import qr
import itertools, copy
from functools import partial
from tqdm import trange
from oscars_toolbox.trabbit import trabbit
from circuit import Circuit, gate_map

# ---- generate random circuit ----- #
def random_circuit(N, depth, Rx_prob = 1/5, Ry_prob = 1/5, Rz_prob = 1/5, P_prob = 1/5, CNOT_prob = 1/5, verbose=True):
    '''Returns a random circuit for a given number N of qubits and depth and probabilities for each gate being applied to a qubit.

    Params:
        N (int): number of qubits
        depth (int): depth of circuit
        Rx_prob (float): probability of Rx gate
        Ry_prob (float): probability of Ry gate
        Rz_prob (float): probability of Rz gate
        P_prob (float): probability of P gate
        CNOT_prob (float): probability of CNOT gate
        verbose (bool): whether to print the genes
    
    '''

    # initialize probabilities
    p = np.array([Rx_prob, Ry_prob, Rz_prob, P_prob, CNOT_prob])
    p /= np.sum(p)

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
    
    # create circuit object
    circ = Circuit(N=N, genes=genes)

    if verbose: print(circ.genes)

    return circ.create_circuit()

# ---- generate random unitary ----- #
def random_unitary(N):
    '''Returns a random unitary matrix of size 2^N x 2^N.'''
    random_complex_matrix = np.random.randn(2**N, 2**N) + 1j * np.random.randn(2**N, 2**N)
    
    # QR decomposition
    Q, R = qr(random_complex_matrix)
    
    # ensure unitarity by making the diagonal of R real and positive
    Q = np.dot(Q, np.diag(np.diagonal(R) / np.abs(np.diagonal(R))))
    
    return Q

def check_unitary(U):
    '''Checks if U is unitary by returning norm of difference between identity and UU^dagger.'''
    return np.linalg.norm(np.eye(U.shape[0]) - U @ U.conj().T)

# ----- learning ----- #
def loss(params, circ_func, target):
    '''Returns the loss between the circuit with the given params and the target matrix.'''
    return np.linalg.norm(circ_func(params) - target)

def random_angles(num_params):
    '''Returns params for the circuit used in optimization'''
    return np.random.uniform(0, 2*np.pi, size=(num_params))

def find_params(target, tol=1e-4):
    '''Finds the params that minimize the loss between the circuit with the given params and the target matrix.

    Params:
        :target: the target matrix
        :tol: the tolerance for the loss. if loss < tol, then stop
    '''

    N = int(np.log2(target.shape[0]))

    ## helper params for RP ##
    RP_GATES = ['Rx', 'Ry', 'Rz', 'P']
    RP_GATES_ALL = [['Rx', 'Ry', 'Rz', 'P'] for _ in range(N)]
    SINGLE_GATES = []
    for i in range(1, len(RP_GATES)+1):
        SINGLE_GATES.extend(list(itertools.combinations(RP_GATES, i)))

    # make sure SINGLE_GATES properly formatted
    for i, sequence in enumerate(SINGLE_GATES):
        if len(sequence) == 1:
            SINGLE_GATES[i] = [sequence[0]]
        else:
            SINGLE_GATES[i] = [gate for gate in sequence]

    ## helper for CNOT ##
    pairs = list(range(N-1)) # all qubits except for last 1
    # get all possible combinations of pairs
    pairs_combinations = []
    for i in range(1, len(pairs)+1):
        pairs_combinations.extend(list(itertools.combinations(pairs, i)))

    def run(circ, gates_test):
        '''Runs the circuit with the given gates and finds optimal params. returns the params and loss.

        Params:
            :circ: the circuit object
            :gates_test: the gates to test
            :params_test: the params to test

        Returns:
            :x_best: the optimal params
            :loss_best: the loss of the circuit
        '''
        # total num of fitting params is sum of len of each gate list
        num_params = sum([len(gates_test[i]) for i in range(N)])
        # call test on circ
        learner = partial(circ.try_genes, new_gates=gates_test)
        loss_func = partial(loss, circ_func = learner, target=target)
        random_func = partial(random_angles, num_params)

        # minimize the loss
        x_best, loss_best = trabbit(loss_func, random_func, alpha=1, temperature = 0, num=1, tol=tol, verbose=True)

        return x_best, loss_best
    
    def prune_RP(circ):
        '''Adds a Rx Ry Rz P block to each qubit and then determines the simplest configuration for each individual gate'''
        # add Rx Ry Rz P block to each qubit
        x_best_initial, loss_best_initial = run(circ, RP_GATES_ALL)
        print(f'initial RP loss: {loss_best_initial}')

        # initialize params
        best_gates = [[] for _ in range(N)]

        for i in range(N): # for every qubit
            # initialize with what we have so far
            min_loss = loss_best_initial
            best_gate_seq = RP_GATES 

            loss_ls = []

            for gate_seq in SINGLE_GATES: # for every possible sequence of RP gates
                gates_test = copy.deepcopy(RP_GATES_ALL)
                gates_test[i] = gate_seq # replace the gates for the qubit
                _, loss_best = run(circ, gates_test)

                loss_ls.append(loss_best)

            # find the best gate sequence
            min_loss = min(loss_ls)
            min_index = loss_ls.index(min_loss)
            best_gate_seq = SINGLE_GATES[min_index]
               
            # update the best gates
            best_gates[i] = best_gate_seq

        # solve for the best params
        x_best_final, loss_best_final = run(circ, best_gates)
        print(f'best RP loss: {loss_best_final}')

        
        # update the circuit
        circ.update_genes(best_gates, x_best_final)

        # check loss
        loss_final = loss(None, circ.create_circuit, target)
        print(f'final RP loss: {loss_final}')
        return circ, loss_final
    
    def add_CNOT(circ):
        '''Try adding a CNOT layer to each possible collection of pairs of qubits'''

        # initialize params
        current_loss = loss(None, circ.create_circuit, target)
        # current_loss = run(circ, [[] for _ in range(N)])[1]
        print(f'initial CNOT loss: {current_loss}')
        best_loss = current_loss
        best_gates = [[] for _ in range(N)]
        best_params = []

        # iterate over all possible combinations of pairs
        for pairs in pairs_combinations:
            test_gates = [[] for _ in range(N)]
            for pair in pairs:
                test_gates[pair] = ['CNOT']
            print(test_gates)

            # test the loss
            params, loss_val = run(circ, test_gates)

            # circ_func = partial(circ.try_genes, new_gates=test_gates)
            # loss_val = loss(None, circ_func, target)
            print(f'CNOT loss: {loss_val} with {pairs}')
            print(circ.genes)
            # update the best gates
            if loss_val < best_loss:
                best_loss = loss_val
                best_gates = test_gates
                best_params = params
            
        # update the circuit
        circ.update_genes(best_gates, best_params)
        print(f'best gates: {best_gates}')
        # check loss
        loss_final = loss(None, circ.create_circuit, target)
        print(f'final CNOT loss: {loss_final}')
        return circ, best_loss
    
    # create circuit object
    circ = Circuit(N=N)


    # circ, loss_final = add_CNOT(circ)
    
    # add RP block
    circ, loss_final = prune_RP(circ)
    if loss_final >= tol:
        # add CNOT layer
        circ, loss_final = add_CNOT(circ)
        c = 0
        while loss_final >= tol and c < 10:
            print('-------')
            print(f'c = {c}')
            print('-------')
            circ, loss_final = prune_RP(circ)
            if loss_final < tol:
                break
            circ, loss_final = add_CNOT(circ)
            if loss_final < tol:
                break
            c += 1
    print('Genes', circ.genes)
    return circ.genes, loss_final


# ------ rigorous testing ------ #
def benchmark(N, depth, gen_func, reps=20):
    '''Returns the avg and sem of loss of the model over reps num of trials.'''
    loss_list = []
    for _ in trange(reps):
        # generate random target
        target = gen_func(N, depth)

        # find params
        _, loss_best = find_params(target)
        loss_list.append(loss_best)
    print(f'loss: {np.mean(loss_list)} ± {np.std(loss_list)/np.sqrt(reps)}')
    return np.mean(loss_list), np.std(loss_list)/np.sqrt(reps)
        
if __name__ == '__main__':
    num_qubits = 3
    depth = 5
    target = random_circuit(num_qubits, depth)
    # target = Circuit(N=num_qubits, genes = [[['Rx', np.pi], ['P', np.pi/3]], [['Rx', np.pi], ['Rz', np.pi/6]], [['Rx', np.pi]]]).create_circuit()    
    find_params(target)
    # genes = [[['Rx', 0.1], ['Ry', 0.2], ['Rz', 0.3], ['P', 0.4], ['CNOT', np.pi/2]], [['Rx', 0.5], ['Ry', 0.6], ['Rz', 0.7], ['P', 0.8], ['CNOT', np.pi/2]], [['Rx', 0.9], ['Ry', 1.0], ['Rz', 1.1], ['P', 1.2]]]
    
    
    
    # circ = Circuit(N=num_qubits)

    # gates = [['Rx', 'Ry', 'Rz', 'P'], ['Rx', 'Ry', 'Rz', 'P'], ['Rx', 'Ry', 'Rz', 'P']]
    # params = [[np.pi, np.pi/2, np.pi/3, np.pi/4], [np.pi, np.pi/2, np.pi/3, np.pi/4], [np.pi, np.pi/2, np.pi/3, np.pi/4]]

    # # convert params to single list
    # params = [param for sublist in params for param in sublist]

    # print(circ.try_genes(new_gates = gates, new_params=params))
    # circ.update_genes(gates, params)
    # print(circ.create_circuit())

