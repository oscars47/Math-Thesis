# based on elegans_new.py, implements a learning circuit with adaptive architecture
import numpy as np
from functools import partial
import itertools, copy
from tqdm import trange
from oscars_toolbox.trabbit import trabbit

# ---- define universal gate set ----- #
# "[S]ingle qubit and CNOT gates together can be used to implement an arbitrary two-level operation on the state space of n-qubibits" --  (Nielsen and Chuang, 2000, p.191) #
gate_map = {
    'Rx': lambda theta: np.array([[np.cos(theta), -1j * np.sin(theta)], [-1j * np.sin(theta), np.cos(theta)]]),
    'Ry': lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
    'Rz': lambda theta: np.array([[np.exp(-1j * theta), 0], [0, np.exp(1j * theta)]]),
    'P': lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]]),
    'CNOT': lambda theta: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta), np.sin(theta)], [0 ,0, np.sin(theta), np.cos(theta)]]),
}
I2 = np.eye(2)
RP_SEQUENCE = ['Rx', 'Ry', 'Rz', 'P']
RP_GENES = [[gate, None] for gate in RP_SEQUENCE] # list of [gate, param] where param is None if gate is to be optimized
# print('RP', RP_genes)
# print('------')
# print(list(itertools.combinations(RP_genes, 3)))
# print('------')                  
SINGLE_GENES = []
for i in range(1, len(RP_GENES)+1):
    SINGLE_GENES.extend(list(itertools.combinations(RP_GENES, i)))

# make sure SINGLE_GENES properly formatted
for i, sequence in enumerate(SINGLE_GENES):
    if len(sequence) == 1:
        SINGLE_GENES[i] = [sequence[0]]
    else:
        SINGLE_GENES[i] = [gate for gate in sequence]
print(SINGLE_GENES)

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
                print(genes)
                raise ValueError(f"Unsupported gate type: {gate[0]}")
            # apply the gate to the circuit
            qc = term @ qc
    return qc

# ---- learning circuit ----- #
def learner(params_new, N, genes, base_sequence = None):
    '''Requires genes, list of lists for each qubit of [gate, param] where gate is str and param is float. if param is None, then it is treated as a param to be optimized. otherwise, it is a set gene.

    Params:
        :N: number of qubits
        :genes: list of lists for each qubit of [gate, param] where gate is str and param is float. if param is None, then it is treated as a param to be optimized. otherwise, it is a set gene.
        :params_new: new parameters to be optimized
        :base_sequence: if not None, then use this sequence for all qubits. otherwise, use the genes as is
    
    
    '''

    # DEPRECATED: if base_sequence is None, then use the genes as is
    #  genes = []
    #     for i, sequence in enumerate(base_sequence):
    #         params_i = params[i:i+len(sequence)]
    #         genes_i = [[gate, params_i[j]] for j, gate in enumerate(sequence)]
    #         genes.append(genes_i

    if base_sequence is not None: # if have universal sequence to be applied to all qubits
        # reshape params
        params = np.reshape(params, (N, len(base_sequence)))

        # initialize circuit
        # get a list of lists of [gate, param] where gate is str and param is float or None if gate == CNOT
        genes = [[[gate, params[i][j]] for j, gate in enumerate(base_sequence)] for i in range(N)]
    else: # if have separate sequence for each qubit
        assert len(genes)==N, f'Need a sequence for each qubit, got {genes}'
        genes_c = genes.copy()
        # process each separate sequence
        for i, sequence in enumerate(genes_c):
            for j, gate in enumerate(sequence):
                if len(gate)==1:
                    gate = gate[0]
                if len(genes_c[i][j]) == 1:
                    genes_c[i][j] = [gate, None]
                if gate[1] is None: # insert param to be optimized
                    genes_c[i][j][1] = params_new[i+j]
    qc = create_circuit(N, genes_c)
    return qc

def random_angles(num_params):
    '''Returns params for the circuit used in optimization'''
    return np.random.uniform(0, 2*np.pi, size=(num_params))

def count_nones(genes):
    '''Counts the number of None params in the genes'''
    count = 0
    for qubit in genes:
        print('qubit', qubit)
        for gate in qubit:
            try:
                print(gate)
                if gate[1] is None:
                    count += 1
            except TypeError:
                print('-----')
                print(genes)
                print('--')
                print(gate)
    return count

def insert_genes(genes, params):
    '''Replaces the None params in the genes with the params'''
    for i, qubit in enumerate(genes):
        for j, gate in enumerate(qubit):
            try:
                if gate[1] is None:
                    genes[i][j][1] = params[i][j]
            except TypeError: 
                print('-----')
                print(genes)

    return genes

def insert_gene_seq(genes, qubit, gene_seq, params):
    '''Inserts the gene_seq into the genes for the given qubit'''
    for i, gene in enumerate(gene_seq):
        if gene[1] is None:
            genes[qubit][i][1] = params[qubit][i]
    return genes

def insert_CNOTs_at_pairs(genes_pair, pair_ls, params):
    for k, pair in enumerate(pair_ls): # add CNOTs to each pair
        genes_pair[pair].append(['CNOT', params[k]])
    return genes_pair
    
# ----- learning ----- #
def loss(params, circ_func, target):
    '''Returns the loss between the circuit with the given params and the target matrix.'''
    return np.linalg.norm(circ_func(params) - target)

def find_genes(target, model=0, threshold=1e-4):
    '''Finds the params that minimize the loss between the circuit with the given params and the target matrix.

    Params:
        :target: the target matrix
        :model: whether to use the Rx Ry Rz P block (0) or adapt using parametrized CNOT gates (1)
    '''
    N = int(np.log2(target.shape[0]))

    ## helper for CNOT ##
    pairs = list(range(N-1)) # all qubits except for last 1
    # get all possible combinations of pairs
    pairs_combinations = []
    for i in range(1, len(pairs)+1):
        pairs_combinations.extend(list(itertools.combinations(pairs, i)))
    #####################
    def run(genes):
        '''Runs the learning algorithm with the given genes. Allows for keeping previous parameters in genes'''
        # initialize circuit
        cirq_func = partial(learner, N=N, genes=genes)
        random_func = partial(random_angles, num_params = count_nones(genes))
        loss_func = partial(loss, circ_func = cirq_func, target=target)
        

        ## somehow single_genes is getting modified in the following line

        # minimize the loss
        x_best, loss_best = trabbit(loss_func, random_func, alpha=0.8, temperature=.1, num=1, tol=1e-4)

        # insert the new params into the genes
        genes = insert_genes(genes, x_best)
        return genes, loss_best

    def prune_RP(genes = [], initialize=True):
        # first find the best using RP. then for each qubit, try to replace with one of the 15 choices from single_sequences
        RP_genes = copy.deepcopy(RP_GENES)
        single_genes = copy.deepcopy(SINGLE_GENES)

        if initialize:
            # initialize genes by adding RP_sequence to each qubit
            if len(genes) > 0:
                for i in range(N):
                    genes[i].extend(RP_genes)
            else:
                genes = [RP_genes for _ in range(N)]

        genes, loss_best = run(genes=genes)
        print(f'loss after RP run: {loss_best}')

        # choose the sequence for each qubit from the base_sequences list separately
        for i in range(N):
            # find the best sequence
            min_loss = 99
            best_gene_seq = None
            best_params = None

            for gene_seq in single_genes: # get the gene sequence from the list of all possible sequences
                # single_genes = copy.deepcopy(SINGLE_GENES)
                print(gene_seq, single_genes)
                genes_i = copy.deepcopy(genes)
                genes_i[i] = copy.deepcopy(gene_seq)
                x_best_i, loss_best_i = run(genes=genes_i)
                if loss_best_i < min_loss:
                    min_loss = loss_best_i
                    best_gene_seq = genes_i[i]
                    best_params = x_best_i
            print('updating')
            # update genes
            genes[i] = insert_gene_seq(genes, i, best_gene_seq, best_params)
        
        # check loss with these new genes
        loss_best = loss(genes, create_circuit, target, N)
        # report best new loss
        print('best loss after pruning:', loss_best)
        return genes, loss_best

    def add_CNOT(genes):
        '''Try adding a CNOT layer to each possible collection of pairs of qubits'''
        # try adding CNOT according to pairs_combinations
         # find the best sequence
        min_loss = 99
        best_pair_ls = None
        best_params = None

        for pair_ls in pairs_combinations: # get the gene sequence from the list of all possible sequences
            genes_pair = genes.copy()
            genes_pair = insert_CNOTs_at_pairs(genes_pair, pair_ls, [None]*len(pair_ls))
            
            x_best_i, loss_best_i = run(genes=genes_pair)
            if loss_best_i < min_loss:
                min_loss = loss_best_i
                best_pair_ls = pair_ls
                best_params = x_best_i

        # update genes
        genes = insert_CNOTs_at_pairs(genes, best_pair_ls, best_params)

        # run again
        print(f'loss after CNOTs: {min_loss}')
        # insert the new params into the genes
        genes = insert_genes(genes, x_best)
        return genes, min_loss

    if model==0:
        x_best, loss_best = run(base_sequence=['Rx', 'Ry', 'Rz', 'P'])
        print(f'loss in find_params: {loss_best}')
        return x_best, loss_best
    
    elif model==1: # adaptive, using new genes framework
        # initialize genes
        genes, loss_best = prune_RP()
        print(f'loss after pruning: {loss_best}')
        if loss_best > threshold:
            # add a layer of CNOTs
            genes, loss_best = add_CNOT(genes)
            c = 0
            while loss_best > threshold and c < 10:
                print('-------')
                print(f'c = {c}')
                print('-------')
                # add new RP pruned sequence
                genes, loss_best = prune_RP(genes)
                if loss_best < threshold:
                    break
                # add another layer of CNOTs
                genes, loss_best = add_CNOT(genes)
                c += 1
            
        return genes, loss_best

        
    elif model == -1: # adaptive, DEPRECATED
        # initialize circuit
        x_best, loss_best, base_sequence, base_sequence = prune_RP() # first try only Rx Ry Rz P
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

    elif model==0.5: # testing
        # consider all possible combinations of one, two, etc of Rx Ry Rz and P to use for each qubit to define the learner
        # each qubit sequence is chosen independently
        single_sequences = []
        for i in range(1, len(RP_sequence)+1):
            single_sequences.extend(list(itertools.combinations(RP_sequence, i)))
        
        # first find the best using RP. then for each qubit, try to replace with one of the 15 choices from single_sequences
        base_sequence = [RP_sequence for _ in range(N)]
        x_best, loss_best = run(base_sequence=base_sequence)
        print(f'loss after RP run: {loss_best}')

        # choose the sequence for each qubit from the base_sequences list separately
        for i in range(N):
            # find the best sequence
            min_loss = 99
            best_single_sequence = None

            for sequence in single_sequences:
                base_sequence_i = base_sequence.copy()
                base_sequence_i[i] = sequence
                x_best_i, loss_best_i = run(base_sequence=base_sequence_i)
                if loss_best_i < min_loss:
                    min_loss = loss_best_i
                    best_single_sequence = sequence
            # update base_sequence
            base_sequence[i] = best_single_sequence

        # run again
        x_best, loss_best = run(base_sequence=base_sequence)
        print(f'loss after pruning: {loss_best}')
        print('sequence:', base_sequence)

        # x_best, loss_best = run(base_sequence=[['Rx'], ['Rx'], ['Rx']])
        # print(f'loss in find_params: {loss_best}')
        # return x_best, loss_best
# ------ sample circuits ----- #
def sample_circuits(N, depth, config=0):
    '''Returns a list of num random circuits with N qubits and depth depth.'''
    params = np.random.uniform(0, 2*np.pi, size=(N, depth))
    gates = np.random.choice(RP_SEQUENCE, size=(N, depth))
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
    gates = np.random.choice(RP_SEQUENCE, size=(N, depth))
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
    from elegans_new import print_matrix

    # initialize circuit params #
    num_qubits = 3
    depth = 5

    # define parametrized circuit functions # 
    # sample_cirq_func0 = partial(sample_circuits, config=0)
    # sample_cirq_func1 = partial(sample_circuits, config=1)

    # sample0 = sample_circuits(num_qubits, depth, config=0)
    target = random_circuit(num_qubits, depth)[0]
    find_genes(target, model=1)
    
    # benchmark #
    # benchmark(num_qubits, depth, sample_cirq_func0, reps=20)
    # benchmark(num_qubits, depth, sample_cirq_func1, reps=20)

    