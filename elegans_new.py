# implements new circuit to be the general circuit

import numpy as np
import matplotlib.pyplot as plt
from oscars_toolbox.trabbit import trabbit
from functools import partial
from datetime import datetime

# -------- universal gates -------- #
I2 = np.eye(2)
def Rx(theta): return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
def Ry(theta): return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
def Rz(theta): return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
def P(phi): return np.array([[1, 0], [0, np.exp(1j * phi)]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0 ,0, 1, 0]])

# -------- circuit -------- #
def circuit(N, params):
    '''Implements approximation circuit for a given number of params.

    Circuit: 
        - Rx Ry Rz P initially (4N params)
        - CNOTs between every pair of qubits (no params)
        - Rx Ry Rz P at the end (4N params)
    
    uses 8N total params.
    '''

    assert len(params) == 8 * N, f'Need 8N params, got {len(params)}'

    # divide up params
    init_params = params[:4 * N]
    final_params = params[4 * N:]

    # reshape params
    init_params = np.reshape(init_params, (N, 4))
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
        if i == 0:
            term = CNOT
            # fill in the rest of the term with identity matrices
            for _ in range(N-2):
                term = np.kron(term, I2)
        else:
            # fill in the first i-1 qubits with identity matrices
            term = I2
            for j in range(i-1):
                term = np.kron(term, I2)
            # add CNOT
            term = np.kron(term, CNOT)
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

# -------- randomization -------- #
def random(N):
    '''Returns a random circuit with 8N params.'''
    return np.random.uniform(-np.pi, np.pi, size=(8 * N))

def test_circuit(N):
    '''Returns a random circuit with 8N params.'''
    return circuit(N, random(N))

def random_circuit(N, depth, I2_prob = 1/6, Rx_prob = 1/6, Ry_prob = 1/6, Rz_prob = 1/6, P_prob = 1/6, CNOT_prob = 1/6):
    '''Returns a random circuit for a given number N of qubits and depth and probabilities for each gate being applied to a qubit.'''

    # initialize probabilities
    p = np.array([I2_prob, Rx_prob, Ry_prob, Rz_prob, P_prob, CNOT_prob])
    p /= np.sum(p)

    circ = np.eye(2**N)

    # apply random gates to each qubit for the given depth
    for _ in range(depth):
        # apply random gates to each qubit
        gates = np.random.choice(['I2', 'Rx', 'Ry', 'Rz', 'P', 'CNOT'], size=N, p=p)
        for j, gate in enumerate(gates):
            if gate == 'I2':
                term = np.eye(2**N)
            elif gate == 'Rx':
                if j == 0:
                    term = Rx(np.random.uniform(-np.pi, np.pi))
                    # put identity matrices after the Rx
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
                else:
                    # put identity matrices before the Rx
                    term = I2
                    for _ in range(j-1):
                        term = np.kron(term, I2)
                    term = np.kron(term, Rx(np.random.uniform(-np.pi, np.pi)))
                    # put identity matrices after the Rx
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
            elif gate == 'Ry':
                if j == 0:
                    term = Ry(np.random.uniform(-np.pi, np.pi))
                    # put identity matrices after the Ry
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
                else:
                    # put identity matrices before the Ry
                    term = I2
                    for _ in range(j-1):
                        term = np.kron(term, I2)
                    term = np.kron(term, Ry(np.random.uniform(-np.pi, np.pi)))
                    # put identity matrices after the Ry
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
            elif gate == 'Rz':
                if j == 0:
                    term = Rz(np.random.uniform(-np.pi, np.pi))
                    # put identity matrices after the Rz
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
                else:
                    # put identity matrices before the Rz
                    term = I2
                    for _ in range(j-1):
                        term = np.kron(term, I2)
                    term = np.kron(term, Rz(np.random.uniform(-np.pi, np.pi)))
                    # put identity matrices after the Rz
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
            elif gate == 'P':
                if j == 0:
                    term = P(np.random.uniform(-np.pi, np.pi))
                    # put identity matrices after the P
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
                else:
                    # put identity matrices before the P
                    term = I2
                    for _ in range(j-1):
                        term = np.kron(term, I2)
                    term = np.kron(term, P(np.random.uniform(-np.pi, np.pi)))
                    # put identity matrices after the P
                    for _ in range(N-j-1):
                        term = np.kron(term, I2)
            elif gate == 'CNOT':
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
            # multiply the circuit by the term
            circ = term @ circ 

    return circ

# -------- learning -------- #
def loss(params, target, N):
    '''Returns the loss between the circuit with the given params and the target matrix.'''
    return np.linalg.norm(circuit(N, params) - target)   

def find_params(target):
    '''Finds the params that minimize the loss between the circuit with the given params and the target matrix.'''

    N = int(np.log2(target.shape[0]))

    random_func = partial(random, N)
    loss_func = partial(loss, target=target, N=N)
    
    # minimize the loss
    x_best, loss_best = trabbit(loss_func, random_func, temperature=0.01, alpha=1)
    print(f'loss: {loss_best}')
    print(f'params: {x_best}')

    return x_best, loss_best

# -------- visualize -------- #
def print_matrix(matrix_ls, title=None, savefig=False):
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

    plt.show()

if __name__ == '__main__':
    N = 3
    depth = 2
    target = random_circuit(N, depth)

    params, loss = find_params(target)
    print_matrix([target, circuit(N, params)], title=f'N={N}, depth={depth}, loss={loss}', savefig=True)
    

