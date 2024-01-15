# redo of elegans_adapt.py fixing the issues with genes
import os
import numpy as np
from scipy.linalg import qr
import itertools, copy
from functools import partial
from tqdm import trange
from time import time
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

def find_params(target, tol=1e-4, model=2.1, depth=10, debug = True):
    '''Finds the params that minimize the loss between the circuit with the given params and the target matrix.

    Params:
        :target: the target matrix
        :tol: the tolerance for the loss. if loss < tol, then stop
        :model: which model to use. 0 is the full model that includes CNOT, 1 only uses RP, 2 uses RP and CNOT but doesn't prune RP and never formally updates params so they have to be relearned, 2.1 does model 2 but then at the end prunes all the RP sections
        :depth: the max depth of the circuit
        :debug: whether to include print statements in model 2.
    '''

    N = int(np.log2(target.shape[0]))
    I2 = np.eye(2)

    ## helper params for RP ##
    RP_GATES = ['Rx', 'Ry', 'Rz', 'P']
    RP_GATES_ALL = [RP_GATES for _ in range(N)]
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

    def run(circ, gates_test, target=target, num_qubits=N):
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
        num_params = sum([len(gates_test[i]) for i in range(num_qubits)])
        # call test on circ
        learner = partial(circ.try_genes, new_gates=gates_test)
        loss_func = partial(loss, circ_func = learner, target=target)
        random_func = partial(random_angles, num_params)

        # minimize the loss
        x_best, loss_best = trabbit(loss_func, random_func, alpha=1, temperature = 0, num=5, tol=tol, verbose=True)

        return x_best, loss_best
    
    def prune_RP(circ):
        '''Adds a Rx Ry Rz P block to each qubit and then determines the simplest configuration for each individual gate'''
        # get initial loss
        loss_no_param = loss(None, circ.create_circuit, target)
        print(f'initial loss no params, {loss_no_param}')
        
        # add Rx Ry Rz P block to each qubit
        x_best_initial, loss_best_initial = run(circ, RP_GATES_ALL)
        print(f'initial RP loss: {loss_best_initial}')

        if loss_no_param < loss_best_initial:
            return circ, loss_no_param

        # initialize params
        best_gates = [[] for _ in range(N)]

        for i in range(N): # for every qubit
            # initialize with what we have so far

            for gate_seq in SINGLE_GATES: # for every possible sequence of RP gates
                # need to include everything up to qubit i in the best_gates
                best_gates_copy = copy.deepcopy(best_gates)
                RP_COPY = copy.deepcopy(RP_GATES_ALL)
                # splice together up to i from best_gates and then RP_COPY
                
                gates_test = best_gates_copy[:i]
                gates_test+= RP_COPY[i:]

                gates_test[i] = gate_seq # replace the gates for the qubit
                print(f'qubit {i}')
                print(f'gate seq, {gate_seq}')
                print(gates_test)
                _, loss_best = run(circ, gates_test)

                if loss_best < tol or loss_best <= loss_best_initial: # if have satisfactory results, save this as min
                    best_gates[i] = gate_seq
                    break
   
            if len(best_gates[i])==0: # if haven't found satisfactory
                best_gates[i] = copy.deepcopy( RP_GATES_ALL)[i]
        
        # solve for the best params
        x_best_final, loss_best_final = run(circ, best_gates)
        print(f'best RP loss: {loss_best_final}')
        print(f'initial loss: {loss_best_initial}')

        if loss_best_final < loss_best_initial:
            # update the circuit
            circ.update_genes(best_gates, x_best_final)

        else:
            circ.update_genes(RP_GATES_ALL, x_best_initial)

        # check loss
        loss_final = loss(None, circ.create_circuit, target)
        print(f'final RP loss: {loss_final}')


        return circ, loss_final
    
    def add_CNOT(circ, update=True, current_gates=None):
        '''Try adding a CNOT layer to each possible collection of pairs of qubits.

        Params:
            :circ: the circuit object
            :update: whether to update the circuit object
            :current_gates: the current gates to test -- only used if update=False    
        '''
        # initialize params
        current_loss = loss(None, circ.create_circuit, target)
        # current_loss = run(circ, [[] for _ in range(N)])[1]
        print(f'initial CNOT loss: {current_loss}')
        best_loss = current_loss
        best_gates = [[] for _ in range(N)]
        best_params = []

        # iterate over all possible combinations of pairs
        for pairs in pairs_combinations:
            if current_gates is not None: print(f'current gates {current_gates}')
            if update:
                test_gates = [[] for _ in range(N)]

            elif not(update) and current_gates is not None:
                test_gates = [copy.deepcopy(gate) for gate in current_gates]
    
            for pair in pairs:
                print(f'pair {pair}')
                if update:
                    test_gates[pair] = ['CNOT']
                elif not(update) and current_gates is not None:
                    print(test_gates[pair])
                    test_gates[pair].append('CNOT')

            # print(f'test gates: {test_gates}')

            # test the loss
            params, loss_val = run(circ, test_gates)

            # update the best gates
            if np.isclose(loss_val, best_loss, 1e-5) or loss_val < best_loss:
                best_loss = loss_val
                best_gates = test_gates
                best_params = params

            # if min loss is good enough, exit early
            if best_loss < tol:
                print(f'test gates exiting: {test_gates}')
                break
        if update:
            if best_loss < current_loss:
                # update the circuit
                circ.update_genes(best_gates, best_params)
                # check loss
                loss_final = loss(None, circ.create_circuit, target)
            else: # don't update in this case
                loss_final = current_loss
            print(f'final CNOT loss: {loss_final}')
            return circ, best_loss
        else:
            return best_gates, best_params, best_loss

    def add_RP(circ, current_gates):
        '''Appends RP block to each qubit'''
        gates_test = [copy.deepcopy(gate) for gate in current_gates]
        for i in range(N):
            if len(gates_test[i])==0 or gates_test[i][-1] == 'CNOT':
                gates_test[i] += copy.deepcopy(RP_GATES_ALL[i])
        x_best, loss_best = run(circ, gates_test)
        return x_best, loss_best, gates_test
    
    def do_model2():
        '''implements model 2. 

        Returns: 
        :circ: the circ object of the learned circuit
        
        '''
        x_RP, loss_RP, gates_test = add_RP(circ, [[] for _ in range(N)])
        print(f'initial loss: {loss_RP}')

        # gates_test = [copy.deepcopy(gate) for gate in RP_GATES_ALL]

        if loss_RP < tol:
            circ.update_genes(gates_test, x_RP)
            return circ, loss_RP
        
        # add CNOT layer
        gates_test, best_params, best_loss = add_CNOT(circ, update=False, current_gates=gates_test)
        print(f'best CNOT loss: {best_loss}')
        print(f'best CNOT gates: {gates_test}')

        if best_loss < tol:
            circ.update_genes(gates_test, best_params)
            loss_final = loss(None, circ.create_circuit, target)
            return circ, loss_final
        
        
        c = 0
        while best_loss >= tol and c < depth:
            print('-------')
            print(f'c = {c}')
            print('test gates', gates_test)
            print('-------')
            best_params, best_loss, gates_test = add_RP(circ, current_gates=gates_test)
            print(f'best RP loss: {best_loss}')

            if best_loss < tol:
                circ.update_genes(gates_test, best_params)
                loss_final = loss(None, circ.create_circuit, target)
                break

            gates_test, best_params, best_loss = add_CNOT(circ, update=False, current_gates=gates_test)
            print(f'best CNOT loss: {best_loss}')

            if best_loss < tol:
                circ.update_genes(gates_test, best_params)
                loss_final = loss(None, circ.create_circuit, target)
                break
            c += 1
        return circ, loss_final
    
    # create circuit object
    circ = Circuit(N=N)

    if model == 0:

        # add RP block to each qubit
        circ, loss_RP = prune_RP(circ)
        print(f'initial loss: {loss_RP}')

        if loss_RP < tol:
            return circ.genes, loss_RP
        
        # add CNOT layer
        circ, loss_CNOT = add_CNOT(circ)
        print(f'best CNOT loss: {loss_CNOT}')

        if loss_CNOT < tol:
            return circ.genes, best_loss
        
        print('current genes', circ.genes)


        # loss_final = loss_final_RP
        if loss_CNOT >= tol:
            # add CNOT layer
            circ, loss_final = prune_RP(circ)
            c = 0
            while loss_final >= tol and c < depth:
                print('-------')
                print(f'c = {c}')
                print('-------')
                circ, loss_final = add_CNOT(circ)
                if loss_final < tol:
                    break
                circ, loss_final = prune_RP(circ)
                if loss_final < tol:
                    break
                c += 1

    elif model == 1:
        circ, loss_final = prune_RP(circ)

    elif model == 2 or model == 2.1: # use RP and CNOT but don't prune RP and never formally update params so they have to be relearned 
        circ, loss_final = do_model2() 
        print(f'after model 2, {circ.genes}')

    if model == 2.1:
        '''Implement final check for sequences of RP'''
        print('implementing final check for RPs')
        gates_test, best_params = circ.gates, circ.params
        new_gates = [[] for _ in range(N)]
        new_params = []
        k = 0 # need to keep track of global gate position to get the right param
        print(gates_test)
        for j, qubit_gates in enumerate(gates_test):
            print(f'at {j}th qubit: {qubit_gates}')
            i = 0
            while i < len(qubit_gates):
                print(f'analyzing gate {i}, {qubit_gates[i]}')
                if (len(qubit_gates) - i >= 3) and (qubit_gates[i] == 'Rx' and qubit_gates[i+1] == 'Ry' and qubit_gates[i+2] == 'Rz' and qubit_gates[i+3] == 'P'):
                    print('got RP sequence')
                    # get the desired RP circ as new target
                    targ_RP = Circuit(N = 1)
                    targ_RP.update_genes(new_gates=[['Rx', 'Ry', 'Rz', 'P']], new_params=best_params[k+i:k+i+4])
                    targ_RP_mat = targ_RP.create_circuit()
                    
                    if not(np.all(np.isclose(targ_RP_mat, I2, tol))): # only add gate if not close to identity
                        # learning logic
                        for gate_seq in SINGLE_GATES: # for every possible sequence of RP gates
                            print(f'gate seq: {gate_seq}')
                            
                            # need to include everything up to qubit i in the best_gates
                            x_RP_seq, loss_RP_seq = run(targ_RP, [copy.deepcopy(gate_seq)], target=targ_RP_mat, num_qubits = 1)

                            if loss_RP_seq < tol:
                                new_gates[j]+=copy.deepcopy(gate_seq)
                                new_params.append(x_RP_seq)
                                print(f'successfully learned RP section with loss {loss_RP_seq}')
                                print(new_gates)
                                break
                    i+=4
                else:
                    print(f'not RP: {qubit_gates[i]}')
                    # add what we have here currently
                    new_gates[j]+=[copy.deepcopy(qubit_gates[i])]
                    new_params.append(best_params[k+i])
                    i+=1
                
        k += len(qubit_gates)
    # recalculate loss
    circ = Circuit(N=N)
    # update the genes
    # print(sum([len(qubit_gates) for qubit_gates in new_gates]), len(new_params))
    print(new_gates)
    circ.update_genes(new_gates=new_gates, new_params=new_params)
    # get loss relative to original main target
    loss_final = loss(None, circ.create_circuit, target)

    print('Genes', circ.genes)
    return circ.genes, loss_final

def sample_circuit(choice):
    '''returns sample circuits for testing purposes'''
    if choice==0: # sample CNOT task
        genes = [[['CNOT', np.pi/2]], [], []]    
    elif choice==1:
        genes = [[['CNOT', np.pi/2]], [['Rx', np.pi/4]], [['P', np.pi/6]]]
    elif choice==2:
        genes = [[['CNOT', np.pi/2]], [['Rx', np.pi/4], ['CNOT', np.pi/3]], [['P', np.pi/6]]]
    elif choice==3:
        genes = [[['Rz', np.pi/7],['CNOT', np.pi/2]], [['Rx', np.pi/4]], [['P', np.pi/6]]]
    elif choice==4:
        genes = [[['Rz', np.pi/7],['CNOT', np.pi/2]], [['Rx', np.pi/4], ['CNOT', np.pi/3]], [['P', np.pi/6]]]
    elif choice==5:
        genes = [[['Rz', np.pi/7],['CNOT', np.pi/2]], [['Rx', np.pi/4]], [['P', np.pi/6]], [['Ry', np.pi/3]]]
    elif choice==6:
        genes = [[['Rz', np.pi/7],['CNOT', np.pi/2]], [['Rx', np.pi/4], ['CNOT', np.pi]], [['P', np.pi/6]], [['Ry', np.pi/3]]]
    elif choice==7:
        genes = [[['Rz', np.pi/7],['CNOT', np.pi/2]], [['CNOT', np.pi/4], ['Rx', np.pi/4]], [['P', np.pi/6]], [['Ry', np.pi/3]]]
    elif choice==8:
        genes = [[['CNOT', np.pi/2], ['Rz', np.pi/7]], [['CNOT', np.pi/4], ['Rx', np.pi/4]], [['P', np.pi/6]], [['Ry', np.pi/3]]]
    elif choice==9:
        genes = [[['CNOT', np.pi/2], ['Rz', np.pi/7]], [['Rx', np.pi/4], ['CNOT', np.pi/4]], [['P', np.pi/6]], [['Ry', np.pi/3]]]
    
    print(f'sample test genes: {genes}')
    
    circ = Circuit(N=len(genes), genes=genes)
    return circ.create_circuit()

def sample_gates(choice):
    '''Tests decomposition for several key gates in quantum computation'''
    if choice == 'H':
        return np.array([[1, 1], [1, -1]]) * 1 / np.sqrt(2)
    elif choice == 'X':
        return np.array([[0, 1], [1, 0]])
    elif choice == 'Y':
        return np.array([[0, -1j], [1j, 0]])
    elif choice == 'Z':
        return np.array([[1, 0], [0, 1]])
    elif choice == 'CZ':
        pass
    elif choice == 'SWAP':
        pass

    elif choice == 'CCNOT':
        pass
    
    elif choice == 'CSWAP':
        pass

    elif choice == 'CCCNOT':
        pass

# ------ rigorous testing ------ #
def benchmark(N, depth, gen_func, reps=20, log_text=False):
    '''Returns the avg and sem of loss of the model over reps num of trials.'''
    t0 = time()
    loss_list = []
    for _ in trange(reps):
        # generate random target
        target = gen_func(N, depth)

        # find params
        _, loss_best = find_params(target)
        loss_list.append(loss_best)
    print(f'loss: {np.mean(loss_list)} ± {np.std(loss_list)/np.sqrt(reps)}')
    tf = time()
    dt = tf - t0
    mean = np.mean(loss_list)
    sem = np.std(loss_list)/np.sqrt(reps)
    if log_text:
        # confirm directory exists
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        with open(f'logs/log_{t0}_{N}_{depth}.txt', 'a') as f:
            f.write(f'{N}, {depth}, {mean}, {sem}, {dt}\n')
    else:
        return mean, sem, dt
        
if __name__ == '__main__':
    num_qubits = 3
    depth = 5
    target = random_circuit(num_qubits, depth)
    # target = sample_circuit(9)
    # print(np.round(target, 5))
    find_params(target, model=2.1, depth=depth)



    ## solved all sample circuits. now need to be able to simplify the circuits. just simplify the RP part itself -- like instead of optimizing against the entire circuit, optimize against the RP part and then add that.

