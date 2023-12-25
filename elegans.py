# file to implement the q-elegans model as a circuit which should be able to arbitrarily approximate any circuit
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from functools import partial
from oscars_toolbox.trabbit import trabbit

# ------ define gates ------ #
# jones matrices for single qubit rotation; see Simon and Mukunda that show HQQ = SU(2)
def R(alpha): return np.matrix([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def Rp(alpha): return np.matrix([[np.sin(alpha), np.cos(alpha)], [np.cos(alpha), np.sin(alpha)]])
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

    Note: for validation purposes, np.kron(np.kron(P0, I2), I2) + np.kron(np.kron(P1, I2), X) is solution for CNOT(3, 0, 2, 0)
    
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
            gate1 = np.kron(gate1, Rp(theta))
        else:
            gate1 = np.kron(gate1, I2)

    # combine the two parts
    gate = gate + gate1
    return gate

# ------ define circuit ------ #
def get_circuit(n, params, config=0):
    '''Returns a parametrized circuit that can approximate any unitary on n qubits.

    Params:
        n: number of qubits
        params: parameters for the circuit
        config: configuration of the circuit. 0 is only HQQ, 1 is CNOT + HQQ, 2 is HQQ + CNOT + HQQ
    '''

    def apply_HQQ(HQQ_params, circuit):
        # loop through all qubits
        for i in range(n):
            # apply HQQ at ith qubit, I2 everywhere else
            # build up that term and then multiply to the circuit
            term = 1
            for j in range(n):
                if j == i:
                    term = np.kron(term, H(HQQ_params[i, 0]) @ Q(HQQ_params[i, 1]) @ Q(HQQ_params[i, 2]))
                else:
                    term = np.kron(term, I2)
            circuit = circuit @ term

        return circuit

    def apply_CNOT(CNOT_params, circuit):
        # loop through all qubits as control
        for i in range(n):
            # loop through all qubits as target
            for j in range(n):
                # apply CNOT, which in general is defined: CNOT = |0><0| x I + |1><1| x X. we extend it 
                if i < j:
                    circuit = circuit @ CNOT(n, i, j, CNOT_params[i, j])
                elif i > j:
                    circuit = circuit @ CNOT(n, j, i, CNOT_params[i, j])

        return circuit

    assert config in [0, 1, 2], f'config must be 0, 1, or 2. you have = {config}'

    # first apply Had to all qubits
    circuit = Had
    for _ in range(n-1):
        circuit = np.kron(circuit, Had)

    if config == 0:
        # apply HQQ gates on all qubits
        HQQ_params = params.reshape((n, 3))
        circuit = apply_HQQ(HQQ_params, circuit)
   
    elif config == 1:
        CNOT_params = params[:n**2].reshape((n, n))      
        HQQ_params = params[n**2:].reshape((n, 3))
        circuit = apply_CNOT(CNOT_params, circuit)
        circuit = apply_HQQ(HQQ_params, circuit)

    elif config == 2:
        HQQ_params = params[:3*n].reshape((n, 3))
        CNOT_params = params[3*n:3*n+n**2].reshape((n, n))
        HQQ_params2 = params[3*n+n**2:].reshape((n, 3))
        circuit = apply_HQQ(HQQ_params, circuit)
        circuit = apply_CNOT(CNOT_params, circuit)
        circuit = apply_HQQ(HQQ_params2, circuit)

    return circuit

# ------ define cost function ------ #
def loss_func(params, target, config):
    '''Returns the loss function for a given target unitary.'''
    n = int(np.log2(target.shape[0]))
    circuit = get_circuit(n, params, config)
    return np.linalg.norm(circuit - target)
    # print(np.sqrt(np.abs(np.trace(circuit @ target.conj().T))**2))
    # return 1-np.sqrt(np.abs(np.trace(circuit @ target.conj().T))**2)

def random_func(size=1): 
    rand = np.random.uniform(0, 2*np.pi, size=size)
    if size == 1:
        return rand[0]
    else:
        return rand

def find_params(target, config=0):
    '''Returns the parameters that minimize the loss function for a given target unitary.'''
    # initialize parameters
    n = int(np.log2(target.shape[0]))
    # minimize loss function
    loss_func_param = lambda params: loss_func(params, target, config)
    if config == 0:
        random_gen = partial(random_func, size=3*n)
        bounds = [(0, 2*np.pi)] * 3*n
    elif config == 1:
        random_gen = partial(random_func, size=n**2+3*n)
        bounds = [(0, 2*np.pi)] * (n**2 + 3*n)
    elif config == 2:
        random_gen = partial(random_func, size=3*n+n**2+3*n)
        bounds = [(0, 2*np.pi)] * (3*n + n**2 + 3*n)
    
    return trabbit(loss_func=loss_func_param, random_gen=random_gen, bounds=bounds, alpha=1, temperature=.01, verbose=True)

# ------ test ------ #
def random_circuit(n, d, I2_prob = 0.2, H_prob = 0.2, Q_prob = 0.2, HQQ_prob = 0.2, CNOT_prob = 0.2):
    '''Returns a random quantum cictui on n qubits for a depth d.'''
    # initialize random unitary
    unitary = Had
    for _ in range(n-1):
        unitary = np.kron(unitary, Had)

    p = np.array([I2_prob, H_prob, Q_prob, HQQ_prob, CNOT_prob])
    p /= np.sum(p)

    # loop through all layers
    for _ in range(d):
        # choose which gates to apply; if CNOT, chose random control and target qubits
        gates = np.random.choice(['I2', 'H', 'Q', 'HQQ', 'CNOT'], size=n, p=p)
        term = np.array([1])
        # initialize list of CNOT gates to apply at the end
        CNOT_gates = []
        for j, gate in enumerate(gates):
            if gate == 'I2':
                term = np.kron(term, I2)
            elif gate == 'H':
                term = np.kron(term, H(random_func()))
            elif gate == 'Q':
                term = np.kron(term, Q(random_func()))
            elif gate == 'HQQ':
                term = np.kron(term, H(random_func()) @ Q(random_func()) @ Q(random_func()))
            elif gate == 'CNOT':
                i = np.random.choice(n, size=1, replace=False)
                if j < i:
                    i_temp = i
                    i = j
                    j = i_temp
                elif j == i:
                    # choose another qubit
                    while j == i:
                        j = np.random.choice(n, size=1, replace=False)
                    if j < i:
                        i_temp = i
                        i = j
                        j = i_temp
                CNOT_gates.append(CNOT(n, i, j, random_func()))
        # figure out how many I2 to add to make dimensions match
        if len(CNOT_gates) > 0:
            for _ in range(len(CNOT_gates)):  
                term = np.kron(term, I2)
            for CNOT_gate in CNOT_gates:
                term = term @ CNOT_gate
        # apply term to unitary
        unitary = unitary @ term

    return unitary

# ----- measure circuits ------ #
def get_entropy(density_matrix):
    '''Returns the entropy of a given density matrix.'''
    # get eigenvalues
    eigvals = np.linalg.eigvals(density_matrix)
    # get probabilities
    probs = np.abs(eigvals)**2
    # return entropy. fully entangled circuit 
    return -np.sum(probs * np.log(probs))

# ------ plot matrix ------ #
def print_matrix(matrix, title=None, savefig=False):
    '''Prints a matrix in a nice way.'''
    # get magnnitude and phase
    mag = np.abs(matrix)
    phase = np.angle(matrix)
    # where magnitude is 0, phase is 0
    phase[mag == 0] = 0

    # plot with colorbar
    fig, ax = plt.subplots(2, 1, figsize=(5, 8))
    im0 = ax[0].imshow(mag)
    im1 = ax[1].imshow(phase)

    ax[0].set_title('Magnitude')
    ax[1].set_title('Phase')

    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    date = str(datetime.now()).split('.')
    if savefig and title is not None:
        plt.savefig(f'elegans_figs/{title}_{date}.pdf')
    elif savefig:
        plt.savefig(f'elegans_figs/{date}.pdf')

    plt.show()

def compare_matrices(A, B, title=None, savefig=False):
    '''Prints two matrices side by side.'''
    # get magnnitude and phase
    mag_A = np.abs(A)
    phase_A = np.angle(A)
    # where magnitude is 0, phase is 0
    phase_A[mag_A == 0] = 0

    mag_B = np.abs(B)
    phase_B = np.angle(B)
    # where magnitude is 0, phase is 0
    phase_B[mag_B == 0] = 0

    # plot with colorbar
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    im0 = ax[0, 0].imshow(mag_A)
    im1 = ax[1, 0].imshow(phase_A)
    im2 = ax[0, 1].imshow(mag_B)
    im3 = ax[1, 1].imshow(phase_B)

    ax[0, 0].set_title('Magnitude')
    ax[1, 0].set_title('Phase')
    ax[0, 1].set_title('Magnitude')
    ax[1, 1].set_title('Phase')
    
    fig.colorbar(im0, ax=ax[0, 0])
    fig.colorbar(im1, ax=ax[1, 0])
    fig.colorbar(im2, ax=ax[0, 1])
    fig.colorbar(im3, ax=ax[1, 1])

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    date = str(datetime.now()).split('.')
    if savefig and title is not None:
        plt.savefig(f'elegans_figs/{title}_{date}.pdf')
    elif savefig:
        plt.savefig(f'elegans_figs/{date}.pdf')
    
if __name__ == '__main__':
    N = 5
    d = 5
    circuit = random_circuit(N, d, CNOT_prob=0)

    x_best, loss_best = find_params(circuit, config=2)
    print(x_best)
    print(loss_best)

    approx = get_circuit(N, x_best)

    compare_matrices(circuit, approx, title=f'N={N}, d={d}, loss={loss_best}', savefig=True)

    





    
