# rewriting elegans_new.py using cirq for easier visualization
import numpy as np
import cirq
from oscars_toolbox.trabbit import trabbit
from functools import partial
from tqdm import trange


gate_map = {
    'RX': lambda p: cirq.rx(p),
    'RY': lambda p: cirq.ry(p),
    'RZ': lambda p: cirq.rz(p),
    'P': lambda p: cirq.PhasedXPowGate(phase_exponent=p),
    'CNOT': cirq.CNOT,
}

# ---- generate random circuit (borrowed from ga.py) ----- #
def get_random_circuit(num_qubits, depth, Rx_prob = 1/5, Ry_prob = 1/5, Rz_prob = 1/5, P_prob = 1/5, CNOT_prob = 1/5):
    '''Return a random circuit with num_qubits qubits and depth depth.'''
    # get probability list
    prob_list = np.array([Rx_prob, Ry_prob, Rz_prob, P_prob, CNOT_prob])
    # normalize
    prob_list /= np.sum(prob_list)

    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]  # Create a line of qubits
    qc = cirq.Circuit()
    for _ in range(depth):
        for i in range(num_qubits):
            if i != num_qubits - 1:  # If not the last qubit
                gate = np.random.choice(list(gate_map.keys()), p=prob_list)
            else:  # Exclude 'CNOT' for the last qubit
                gate = np.random.choice([g for g in gate_map.keys() if g != 'CNOT'])
    
            if gate in ['RX', 'RY', 'RZ', 'P']:  # Parameterized gates
                param = np.random.uniform(0, 2*np.pi)
                qc.append(gate_map[gate](param).on(qubits[i]))
            elif gate == 'CNOT' and i < num_qubits - 1:  # 'CNOT' gate, ensuring not on the last qubit
                qc.append(gate_map[gate].on(qubits[i], qubits[i+1]))
            else:  # Non-parameterized gates
                raise ValueError(f"Unsupported gate type: {gate}")
    print(qc)
    return qc

# ---- learning circuit ----- #
def random_miniP(N):
    '''Returns a random circuit with 4N params.'''
    return np.random.uniform(-np.pi, np.pi, size=(4 * N))

def circuit_miniP(N, params):
    '''Only uses Rx Ry Rz P block for all of the qubits'''
    assert len(params) == 4 * N, f'Need 4N params, got {len(params)}'

    # reshape params
    params = np.reshape(params, (N, 4))

    qubits = [cirq.LineQubit(i) for i in range(N)]  # create a line of qubits
    qc = cirq.Circuit() # initialize circuit
    for i in range(N):
        qc.append(gate_map['RX'](params[i][0]).on(qubits[i]))
        qc.append(gate_map['RY'](params[i][1]).on(qubits[i]))
        qc.append(gate_map['RZ'](params[i][2]).on(qubits[i]))
        qc.append(gate_map['P'](params[i][3]).on(qubits[i]))

    return qc

# ------ learning ----- #
def loss(params, circ_func, target, N):
    '''Returns the loss between the circuit with the given params and the target matrix.'''
    return np.linalg.norm(circ_func(N, params).unitary() - target)   

def find_params(target, model=0):
    '''Finds the params that minimize the loss between the circuit with the given params and the target matrix.

    Params:
        :target: the target matrix
        :model: whether to use the Rx Ry Rz P block (0) or just the Rx Ry Rz block (1)
    '''

    N = int(np.log2(target.shape[0]))

    if model==0:
        random_func = partial(random_miniP, N)
        loss_func = partial(loss, circ_func = circuit_miniP, target=target, N=N)

    # minimize the loss
    x_best, loss_best = trabbit(loss_func, random_func, temperature=0.1, alpha=0.8, num=50, tol=1e-4)
    print(f'loss in find_params: {loss_best}')
    print(f'params: {x_best}')

    return x_best, loss_best

def benchmark(num_qubits, depth, model, reps, Rx_prob = 1/5, Ry_prob = 1/5, Rz_prob = 1/5, P_prob = 1/5, CNOT_prob = 1/5):
    '''Returns the avg and sem of loss of the model over reps num of trials.'''
    loss_list = []
    for _ in trange(reps):
        # generate random target
        target = get_random_circuit(num_qubits, depth, Rx_prob=Rx_prob, Ry_prob = Ry_prob, Rz_prob=Rz_prob, P_prob=P_prob, CNOT_prob=CNOT_prob).unitary()

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

if __name__ == '__main__':
    benchmark(6, 10, 0, 20, CNOT_prob=0)

    
