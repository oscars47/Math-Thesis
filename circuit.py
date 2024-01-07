import numpy as np
import copy

# ---- define universal gate set ----- #
# "[S]ingle qubit and CNOT gates together can be used to implement an arbitrary two-level operation on the state space of n-qubibits" --  (Nielsen and Chuang, 2000, p.191) #
gate_map = {
    'Rx': lambda theta: np.array([[np.cos(theta), -1j * np.sin(theta)], [-1j * np.sin(theta), np.cos(theta)]]),
    'Ry': lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
    'Rz': lambda theta: np.array([[np.exp(-1j * theta), 0], [0, np.exp(1j * theta)]]),
    'P': lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]]),
    'CNOT': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0 ,0, 1, 0]]),
}

## ---- manage circuit through gene encoding ----- ##
class Circuit():
    '''Genes come in 2 types: set and new. Set have learned params, whereas new are random params. Each gene contains a gate and param.
    
    Methods:
    create_circuit: takes in list of list of [gate, param] where gate is str and param is float or None if gate == CNOT and calculates the resulting unitary
    try_genes: used for testing out new parameters without actually changing the genes of the circuit. 
    update_genes: commits the new gates and params to memory.
    
    '''
    def __init__(self, N, genes=None):
        self.N = N
        self.genes = genes
        if genes is None:
            self.genes = [[] for _ in range(N)] # initialize empty genes
    
    def create_circuit(self, test_genes=None):
        '''Converts list of list of [gate, param] where gate is str and param is float or None if gate == CNOT and calculates the resulting unitary'''

        if test_genes is None:
            test_genes = self.genes
            assert test_genes[0] != [], 'Need genes to create circuit'

        test_genes = copy.deepcopy(test_genes)

        N = self.N

        qc = np.eye(2**N)
        for i, gates in enumerate(test_genes):
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
                    term = gate_map[gate[0]]
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

    def try_genes(self, new_params, new_gates):
        '''Takes in new list of lists of gates per qubit and list of params per qubit and returns the resulting unitary'''

        if new_gates is None or new_params is None:
            return self.create_circuit()

        assert len(new_gates) == self.N, f'Need {self.N} qubit lists, got {len(new_gates)}'     

        # print('new gates:', new_gates) 
        # print('new params:', new_params)

        N = self.N

        # get list of lengths
        lens = [len(gates) for gates in new_gates]

        old_genes = copy.deepcopy(self.genes)
        # add new genes to the list
        for i in range(N): # for each qubit list
            for j in range(len(new_gates[i])): # for each gate in the qubit list
                if new_gates[i][j] != 'CNOT':
                    params_index = sum(lens[:i]) + j

                    self.genes[i] += [[new_gates[i][j], new_params[params_index]]]
                elif new_gates[i][j] == 'CNOT':
                    old_genes[i] += [[new_gates[i][j], None]]
                # otherwise, do nothing

        # remove any CNOTs if placed in the last qubit; if so, remove
        genes_last = []
        for i, gates in enumerate(old_genes[-1]): # check list for last qubit
            if gates[0] != 'CNOT':
                genes_last.append(gates)
        old_genes[-1] = genes_last

        # create circuit
        return self.create_circuit(old_genes)
    
    def update_genes(self, new_gates, new_params):
        '''Takes in new list of lists of gates per qubit and list of params per qubit and updates the genes'''
        
        assert len(new_gates) == self.N, f'Need {self.N} qubit lists, got {len(new_gates)}'
        
        N = self.N

        # get list of lengths
        lens = [len(gates) for gates in new_gates]

        # add new genes to the list
        for i in range(N):
            for j in range(len(new_gates[i])):
                if new_gates[i][j] != 'CNOT':
                    # get the index of the params, which is the sum of the lengths of the previous genes
                    params_index = sum(lens[:i]) + j

                    self.genes[i] += [[new_gates[i][j], new_params[params_index]]]
                elif new_gates[i][j] == 'CNOT':
                    self.genes[i] += [[new_gates[i][j], None]]
        
        # remove any CNOTs if placed in the last qubit; if so, remove
        genes_last = []
        for i, gates in enumerate(self.genes[-1]):
            if gates[0] != 'CNOT':
                genes_last.append(gates)
        self.genes[-1] = genes_last

if __name__ == '__main__':
    num_qubits = 3
    genes = [[['Rx', 0.1], ['Ry', 0.2], ['Rz', 0.3], ['P', 0.4], ['CNOT', None]], [['Rx', 0.5], ['Ry', 0.6], ['Rz', 0.7], ['P', 0.8], ['CNOT', None]], [['Rx', 0.9], ['Ry', 1.0], ['Rz', 1.1], ['P', 1.2]]]

    circ = Circuit(N=num_qubits, genes=genes)
    print(circ.create_circuit())
