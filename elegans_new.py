# implements new circuit to be the general circuit

import numpy as np
import matplotlib.pyplot as plt
from oscars_toolbox.trabbit import trabbit
from scipy.optimize import minimize, linear_sum_assignment
from functools import partial
from datetime import datetime

# -------- universal gates -------- #
I2 = np.eye(2)
def Rx(theta): return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
def Ry(theta): return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
def Rz(theta): return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
def P(phi): return np.array([[1, 0], [0, np.exp(1j * phi)]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0 ,0, 1, 0]])
def CNOTp(alpha): return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.sin(alpha), np.cos(alpha)], [0, 0, np.cos(alpha), np.sin(alpha)]]) # parametrized CNOT

# -------- circuit -------- #
def circuit(N, params):
    '''Implements approximation circuit for a given number of params.

    Circuit: 
        - Rx Ry Rz P initially (4N params)
        - CNOTps between every pair of qubits plus RP tensor RP on either side (N+16N = 17N params)
        - Rx Ry Rz P at the end (4N params)
    
    uses 18N total params.
    '''

    assert len(params) == 25 * N, f'Need 25N params, got {len(params)}'

    # divide up params
    init_params = params[:4 * N]
    CNOT_params = params[4 * N: 5 * N]
    CNOT_extra_b1_params = params[5*N: 9*N]
    CNOT_extra_b2_params = params[9*N: 13*N]
    CNOT_extra_a1_params = params[13*N: 17*N]
    CNOT_extra_a2_params = params[17*N: 21*N]
    final_params = params[-4*N:]

    # reshape params
    init_params = np.reshape(init_params, (N, 4))
    CNOT_extra_b1_params = np.reshape(CNOT_extra_b1_params, (N, 4))
    CNOT_extra_b2_params = np.reshape(CNOT_extra_b2_params, (N, 4))
    CNOT_extra_a1_params = np.reshape(CNOT_extra_a1_params, (N, 4))
    CNOT_extra_a2_params = np.reshape(CNOT_extra_a2_params, (N, 4))
    final_params = np.reshape(final_params, (N, 4))

    # initialize circuit
    for i in range(N):
        if i == 0:
            circ = Rx(init_params[i][0]) @ Ry(init_params[i][1]) @ Rz(init_params[i][2]) @ P(init_params[i][3])
        else:
            circ = np.kron(circ, Rx(init_params[i][0]) @ Ry(init_params[i][1]) @ Rz(init_params[i][2]) @ P(init_params[i][3]))

    # CNOTs
    for i in range(N-1):
        # apply CNOT between ith and (i+1)th qubits. prepare term to multiply the circuit by which is I for all qubits except the ith and (i+1)th qubits
        RP_b1 = Rx(CNOT_extra_b1_params[i][0]) @ Ry(CNOT_extra_b1_params[i][1]) @ Rz(CNOT_extra_b1_params[i][2]) @ P(CNOT_extra_b1_params[i][3])
        RP_b2 = Rx(CNOT_extra_b2_params[i][0]) @ Ry(CNOT_extra_b2_params[i][1]) @ Rz(CNOT_extra_b2_params[i][2]) @ P(CNOT_extra_b2_params[i][3])
        RP_a1 = Rx(CNOT_extra_a1_params[i][0]) @ Ry(CNOT_extra_a1_params[i][1]) @ Rz(CNOT_extra_a1_params[i][2]) @ P(CNOT_extra_a1_params[i][3])
        RP_a2 = Rx(CNOT_extra_a2_params[i][0]) @ Ry(CNOT_extra_a2_params[i][1]) @ Rz(CNOT_extra_a2_params[i][2]) @ P(CNOT_extra_a2_params[i][3])

        RP_init = np.kron(RP_b1, RP_b2)
        RP_final = np.kron(RP_a1, RP_a2)
        if i == 0:
            term = RP_final @ CNOTp(CNOT_params[i])@ RP_init
            # fill in the rest of the term with identity matrices
            for _ in range(N-2):
                term = np.kron(term, I2)
        else:
            # fill in the first i-1 qubits with identity matrices
            term = I2
            for _ in range(i-1):
                term = np.kron(term, I2)
            # add CNOT
            term = np.kron(term, RP_final @ CNOTp(CNOT_params[i])@ RP_init)
            # fill in the rest of the term with identity matrices
            for _ in range(N-2-i):
                term = np.kron(term, I2)
        # multiply the circuit by the term
        circ = term @ circ

    # final gates
    for i in range(N):
        if i == 0:
            term = Rx(final_params[i][0]) @ Ry(final_params[i][1]) @ Rz(final_params[i][2]) @ P(final_params[i][3])
        else:
            term = np.kron(term, Rx(final_params[i][0]) @ Ry(final_params[i][1]) @ Rz(final_params[i][2]) @ P(final_params[i][3]))

    # multiply the circuit by the term
    circ = term @ circ

    return circ

def circuit_mini(N, params):
    '''Only uses Rx Ry Rz block for all of the qubits'''
    assert len(params) == 3 * N, f'Need 3N params, got {len(params)}'

    # reshape params
    params = np.reshape(params, (N, 3))

    # initialize circuit
    for i in range(N):
        if i == 0:
            circ = Rx(params[i][0]) @ Ry(params[i][1]) @ Rz(params[i][2])
        else:
            circ = np.kron(circ, Rx(params[i][0]) @ Ry(params[i][1]) @ Rz(params[i][2]))
    
    return circ

def circuit_miniP(N, params):
    '''Only uses Rx Ry Rz P block for all of the qubits'''
    assert len(params) == 4 * N, f'Need 4N params, got {len(params)}'

    # reshape params
    params = np.reshape(params, (N, 4))

    # initialize circuit
    for i in range(N):
        if i == 0:
            circ = Rx(params[i][0]) @ Ry(params[i][1]) @ Rz(params[i][2]) @ P(params[i][3])
        else:
            circ = np.kron(circ, Rx(params[i][0]) @ Ry(params[i][1]) @ Rz(params[i][2]) @ P(params[i][3]))

    return circ

def circuit_CNOT(N, params):
    '''Only uses CNOT block for all of the qubits'''
    assert len(params) == 17*N, f'Need 8N params, got {len(params)}'

    CNOT_params = params[:N]
    CNOT_extra_b1_params = params[N: 5*N]
    CNOT_extra_b2_params = params[5*N: 9*N]
    CNOT_extra_a1_params = params[9*N: 13*N]
    CNOT_extra_a2_params = params[13*N:]

    # reshape params
    CNOT_extra_b1_params = np.reshape(CNOT_extra_b1_params, (N, 4))
    CNOT_extra_b2_params = np.reshape(CNOT_extra_b2_params, (N, 4))
    CNOT_extra_a1_params = np.reshape(CNOT_extra_a1_params, (N, 4))
    CNOT_extra_a2_params = np.reshape(CNOT_extra_a2_params, (N, 4))

    circ = np.eye(2**N)

    for i in range(N-1):
        # apply CNOT between ith and (i+1)th qubits. prepare term to multiply the circuit by which is I for all qubits except the ith and (i+1)th qubits
        RP_b1 = Rx(CNOT_extra_b1_params[i][0]) @ Ry(CNOT_extra_b1_params[i][1]) @ Rz(CNOT_extra_b1_params[i][2]) @ P(CNOT_extra_b1_params[i][3])
        RP_b2 = Rx(CNOT_extra_b2_params[i][0]) @ Ry(CNOT_extra_b2_params[i][1]) @ Rz(CNOT_extra_b2_params[i][2]) @ P(CNOT_extra_b2_params[i][3])
        RP_a1 = Rx(CNOT_extra_a1_params[i][0]) @ Ry(CNOT_extra_a1_params[i][1]) @ Rz(CNOT_extra_a1_params[i][2]) @ P(CNOT_extra_a1_params[i][3])
        RP_a2 = Rx(CNOT_extra_a2_params[i][0]) @ Ry(CNOT_extra_a2_params[i][1]) @ Rz(CNOT_extra_a2_params[i][2]) @ P(CNOT_extra_a2_params[i][3])

        RP_init = np.kron(RP_b1, RP_b2)
        RP_final = np.kron(RP_a1, RP_a2)
        if i == 0:
            term = RP_final @ CNOTp(CNOT_params[i])@ RP_init
            # fill in the rest of the term with identity matrices
            for _ in range(N-2):
                term = np.kron(term, I2)
        else:
            # fill in the first i-1 qubits with identity matrices
            term = I2
            for _ in range(i-1):
                term = np.kron(term, I2)
            # add CNOT
            term = np.kron(term, RP_final @ CNOTp(CNOT_params[i])@ RP_init)
            # fill in the rest of the term with identity matrices
            for _ in range(N-2-i):
                term = np.kron(term, I2)
        # multiply the circuit by the term
        circ = term @ circ
    return circ

# -------- randomization -------- #
def random(N):
    '''Returns a random circuit with 25N params.'''
    return np.random.uniform(0, 2*np.pi, size=(25 * N))

def random_mini(N):
    '''Returns a random circuit with 3N params.'''
    return np.random.uniform(0, 2*np.pi, size=(3 * N))

def random_miniP(N):
    '''Returns a random circuit with 4N params.'''
    return np.random.uniform(0, 2*np.pi, size=(4 * N))

def random_CNOT(N):
    '''Returns a random circuit with 16N params.'''
    return np.random.uniform(0, 2*np.pi, size=(17 * N))

def random_circuit(N, depth, Rx_prob = 1/5, Ry_prob = 1/5, Rz_prob = 1/5, P_prob = 1/5, CNOT_prob = 1/5):
    '''Returns a random circuit for a given number N of qubits and depth and probabilities for each gate being applied to a qubit.'''

    # initialize probabilities
    p = np.array([Rx_prob, Ry_prob, Rz_prob, P_prob, CNOT_prob])
    p /= np.sum(p)

    circ = np.eye(2**N)

    # save a list of lists so we can see what gates were applied to each qubit
    gates_applied = [[] for _ in range(N)]

    # apply random gates to each qubit for the given depth
    for _ in range(depth):
        # apply random gates to each qubit
        gates = np.random.choice(['Rx', 'Ry', 'Rz', 'P', 'CNOT'], size=N, p=p)
        for j, gate in enumerate(gates):
            param = np.random.uniform(0, 2*np.pi)
            if gate == 'Rx':
                # print('Rx')
                if j == 0:
                    term = Rx(param)
                    # put identity matrices after the Rx
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
                else:
                    # put identity matrices before the Rx
                    term = I2
                    for _ in range(j-1):
                        term = np.kron(term, I2)
                    term = np.kron(term, Rx(param))
                    # put identity matrices after the Rx
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
            elif gate == 'Ry':
                # print('Ry')
                if j == 0:
                    term = Ry(param)
                    # put identity matrices after the Ry
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
                else:
                    # put identity matrices before the Ry
                    term = I2
                    for _ in range(j-1):
                        term = np.kron(term, I2)
                    term = np.kron(term, Ry(param))
                    # put identity matrices after the Ry
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
            elif gate == 'Rz':
                # print('Rz')
                if j == 0:
                    term = Rz(param)
                    # put identity matrices after the Rz
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
                else:
                    # put identity matrices before the Rz
                    term = I2
                    for _ in range(j-1):
                        term = np.kron(term, I2)
                    term = np.kron(term, Rz(param))
                    # put identity matrices after the Rz
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
            elif gate == 'P':
                # print('P')
                if j == 0:
                    term = P(param)
                    # put identity matrices after the P
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
                else:
                    # put identity matrices before the P
                    term = I2
                    for _ in range(j-1):
                        term = np.kron(term, I2)
                    term = np.kron(term, P(param))
                    # put identity matrices after the P
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
            elif gate == 'CNOT':
                param = None
                if j == 0:
                    term = CNOT
                    # fill in the rest of the term with identity matrices
                    for _ in range(N-2):
                        term = np.kron(term, I2)
                elif j == N-1:
                    # move it to N-2
                    term = I2
                    for _ in range(N-3):
                        term = np.kron(term, I2)
                    term = np.kron(term, CNOT)
                else:
                    # fill in the first j-1 qubits with identity matrices
                    term = I2
                    for _ in range(j-1):
                        term = np.kron(term, I2)
                    # add CNOT
                    term = np.kron(term, CNOT)
                    # fill in the rest of the term with identity matrices
                    for _ in range(N-2-j):
                        term = np.kron(term, I2)
            # add to the circuit
            gates_applied[j].append([gate, param])
            # multiply the circuit by the term
            circ = term @ circ 

    # print gates applied
    for i, gates in enumerate(gates_applied):
        print(f'qubit {i}: {gates}')

    return circ

# -------- learning -------- #
def loss(params, circ_func, target, N):
    '''Returns the loss between the circuit with the given params and the target matrix.'''
    return np.linalg.norm(circ_func(N, params) - target)   

def find_params(target, model=0, do_tr=True):
    '''Finds the params that minimize the loss between the circuit with the given params and the target matrix.

    Params:
        :target: the target matrix
        :model: whether to use the full circuit (0) or just the Rx Ry Rz block (1) or Rx Ry Rz P block (2)
        :do_tr: whether to use trabbit or not
    '''

    N = int(np.log2(target.shape[0]))

    if model==0:
        random_func = partial(random, N)
        loss_func = partial(loss, circ_func = circuit, target=target, N=N)

    elif model == 1:
        random_func = partial(random_mini, N)
        loss_func = partial(loss, circ_func = circuit_mini, target=target, N=N)

    elif model == 2:
        random_func = partial(random_miniP, N)
        loss_func = partial(loss, circ_func = circuit_miniP, target=target, N=N)
    elif model == 3:
        random_func = partial(random_CNOT, N)
        loss_func = partial(loss, circ_func = circuit_CNOT, target=target, N=N)

    if do_tr:
        # minimize the loss
        x_best, loss_best = trabbit(loss_func, random_func, temperature=0.1, alpha=0.8, num=50, tol=1e-4)
        print(f'loss in find_params: {loss_best}')
        print(f'params: {x_best}')

        # results = minimize(loss_func, x0=random_func(), method='Nelder-Mead', options={'maxiter': 1000})
        # loss_m = results.fun

        return x_best, loss_best
    else:
        # minimize the loss
        results = minimize(loss_func, x0=random_func(), method='Nelder-Mead', options={'maxiter': 1000})

        x_best = results.x
        loss_m = results.fun

        print(f'loss in find_params: {loss_m}')

        return x_best, loss_m
    
# -------- testing -------- #
def match_eigenvalues(eigenvals1, eigenvals2):
    # create distance matrix
    distance_matrix = np.abs(eigenvals1[:, np.newaxis] - eigenvals2)
    
    # solve assignment problem using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # return matched eigenvalues and the total distance
    matched_eigenvals1 = eigenvals1[row_ind]
    matched_eigenvals2 = eigenvals2[col_ind]
    total_distance = np.linalg.norm(matched_eigenvals1 - matched_eigenvals2)
    return matched_eigenvals1, matched_eigenvals2, total_distance

def benchmark(num_qubit, depth, model, reps, Rx_prob = 1/5, Ry_prob = 1/5, Rz_prob = 1/5, P_prob = 1/5, CNOT_prob = 1/5):
    '''Returns the avg and sem of loss of the model over reps num of trials.'''
    loss_list = []
    for _ in trange(reps):
        # generate random target
        target = random_circuit(num_qubit, depth, Rx_prob=Rx_prob, Ry_prob = Ry_prob, Rz_prob=Rz_prob, P_prob=P_prob, CNOT_prob=CNOT_prob)
        # find params
        params, loss_best = find_params(target, model=model)
        # calculate loss
        loss = loss_best
        # add to list
        loss_list.append(loss)

    # calculate avg and sem
    avg = np.mean(loss_list)
    sem = np.std(loss_list) / np.sqrt(reps)
    print(f'For {reps} random trials, loss is {avg} +- {sem}')
        
# -------- visualize -------- #
def print_matrix(matrix_ls, title=None, savefig=False, display=False):
    '''Prints a matrix in a nice way.'''
    if len(matrix_ls) == 1:
        # print out the matrix
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    elif len(matrix_ls) == 2:
        # print out the two matrices side by sie
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    else:
        raise ValueError(f'Can only print 1 or 2 matrices, got {len(matrix_ls)}')
    
    for i, matrix in enumerate(matrix_ls):
        # get magnnitude and phase
        mag = np.abs(matrix)
        phase = np.angle(matrix)
        # where magnitude is 0, phase is 0
        phase[mag == 0] = 0

        # plot with colorbar
        im0 = ax[0][i].imshow(mag)
        im1 = ax[1][i].imshow(phase)

        fig.colorbar(im0, ax=ax[0][i])
        fig.colorbar(im1, ax=ax[1][i])

        # set title
        if len(matrix_ls) == 2:
            if i == 0:
                ax[0][i].set_title('Target')
            elif i == 1:
                ax[0][i].set_title('Approx')

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    date = str(datetime.now()).split('.')

    if savefig and title is not None:
        plt.savefig(f'elegans_new_figs/{title}_{date}.pdf')
    elif savefig:
        plt.savefig(f'elegans_new_figs/{date}.pdf')

    if display:
        plt.show()

if __name__ == '__main__':
    from tqdm import trange
    N = 6 # PROBLEM WITH N = 2
    depth = 10
    model = 2 # use full circuit or just Rx Ry Rz block

    benchmark(N, depth, model, 20)