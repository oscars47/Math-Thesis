import numpy as np
import cirq
from scipy.optimize import linear_sum_assignment
from tqdm import trange

# Global variables
global TARG_CIRC  # Target circuit to be approximated
global TARGET_UNITARY  # Target state to be approximated
global NUM_QUBITS  # Number of qubits of target state
global TOTAL_GATES  # Total number of gates in the target circuit

gate_map = {
    'RX': lambda p: cirq.rx(p),
    'RY': lambda p: cirq.ry(p),
    'RZ': lambda p: cirq.rz(p),
    'P': lambda p: cirq.PhasedXPowGate(phase_exponent=p),
    'CX': cirq.CNOT,
}

# ------ helper functions ------ #
def get_random_circuit(num_qubits, depth):
    '''Return a random circuit with num_qubits qubits and depth depth.'''
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]  # Create a line of qubits
    qc = cirq.Circuit()
    for _ in range(depth):
        for i in range(num_qubits):
            if i != num_qubits - 1:  # If not the last qubit
                gate = np.random.choice(list(gate_map.keys()))
            else:  # Exclude 'CX' for the last qubit
                gate = np.random.choice([g for g in gate_map.keys() if g != 'CX'])
    
            if gate in ['RX', 'RY', 'RZ', 'P']:  # Parameterized gates
                param = np.random.uniform(0, 2*np.pi)
                qc.append(gate_map[gate](param).on(qubits[i]))
            elif gate == 'CX' and i < num_qubits - 1:  # 'CX' gate, ensuring not on the last qubit
                qc.append(gate_map[gate].on(qubits[i], qubits[i+1]))
            else:  # Non-parameterized gates
                raise ValueError(f"Unsupported gate type: {gate}")
    return qc

# ------ genetic algorithm ------ #
class Individual:
    '''Class for an individual in the population.'''
    def __init__(self, genes):
        ''':genes: list of lists of gate (str), parameter (float), where index in main list is qubit index.'''
        self.genes = genes
        # print('genes', self.genes, '\n')
        self.num_gates = np.sum([len(gate_list) for gate_list in genes])
        # depth is len of longest list in genes
        self.depth = max([len(gate_list) for gate_list in genes])
        self.fitness = self.get_fitness()

    # Function to create a gate instance based on a string and parameter
    def create_gate(gate_str, param=None):
        if gate_str in gate_map:
            if param is not None:
                return gate_map[gate_str](param)
            else:
                return gate_map[gate_str]
        else:
            raise ValueError(f"Unsupported gate type: {gate_str}")

    # express the genes as a circuit
    def get_circ(self):
        '''Return the matrix representation of the circuit.'''
        qubits = [cirq.LineQubit(i) for i in range(NUM_QUBITS)]  # Create a line of qubits
        qc = cirq.Circuit()
        for j, qubit_ls in enumerate(self.genes): # consider each qubit
            for gate_param in qubit_ls:
                # apply a moment to each qubit
                gate = gate_param[0]
                try:
                    if gate in ['RX', 'RY', 'RZ', 'P']:
                        param = gate_param[1]
                except:
                    print(qubit_ls)
                    print(gate_param)
                    raise ValueError(f"Unsupported gate type hi: {gate}")

                if (j < self.num_gates - 1) and (gate != 'CX'): # if the last gate is a CX, add a random gate to the last qubit
                    # gate = np.random.choice([g for g in gate_map.keys() if g != 'CX'])
                    # update genes by removing the last gate and adding a new random gate
                    
                    try:
                        if gate in ['RX', 'RY', 'RZ', 'P']:  # Parameterized gates
                            qc.append(gate_map[gate](param).on(qubits[j]))
                        elif gate == 'CX' and j < num_qubits - 1:  # 'CX' gate, ensuring not on the last qubit
                            qc.append(gate_map[gate].on(qubits[j], qubits[j+1]))
                        else:  # Non-parameterized gates
                            raise ValueError(f"Unsupported gate type: {gate}")
                    except:
                        raise ValueError(f"Unsupported gate type: {gate_param}")
                else:
                    # append the identity (P(0)) to the last qubit
                    qc.append(gate_map['P'](0).on(qubits[j]))
        return qc

    # evaluate the fitness of the individual using the circuit
    def get_fitness_ip(self):
        '''Calculates the inner product of the circuit with the target circuit.'''
        qc = self.get_circ()
        # calculate the fidelity of the circuit with the target circuit
        qc_unitary = cirq.unitary(qc)
        # Calculate the fidelity
        fidelity = np.abs(np.trace(np.dot(TARGET_UNITARY.conj().T, qc_unitary)))**2 / (len(TARGET_UNITARY) * len(qc_unitary))
        return fidelity
    
    def get_fitness(self):
        '''Computes total fitness of the individual, which includes a penalization for the number of gates.'''
        # return self.get_fitness_ip() - self.num_gates * 1/TOTAL_GATES # penalize for number of gates; may need to multiply by some factor
        return self.get_fitness_ip()

        
    # mutate the individual's genes: change a gate or parameter
    def mutate(self, mut_rate):
        for i, qubit_genes in enumerate(self.genes):
            for j in range(len(qubit_genes)):
                if np.random.uniform(0, 1) < mut_rate:
                    # choose new gate 
                    gate = np.random.choice(list(gate_map.keys()))
                    # choose new parameter
                    if gate in ['RX', 'RY', 'RZ', 'P']:
                        param = np.random.uniform(0, 2*np.pi)
                    else:
                        param = None
                    self.genes[i][j] = [gate, param]
        return self
    
# crossover with another individual
def crossover(ind1, ind2):
    '''Takes as as input two Individuals and returns new Individual.'''
    # choose a random point to split the genes
    split_point = np.random.randint(0, len(ind1.genes))
    # create new individual with genes from ind1 up to split point
    new_genes = ind1.genes[:split_point] + ind2.genes[split_point:]
    return Individual(new_genes)

# initialize a population of individuals
def initialize_population(pop_size):
    '''Initialize a population of circuits with random parameters.'''
    population = []
    print('Initializing population...')
    for _ in trange(pop_size): # for each individual
        # create a random circuit
        genes = []
        for _ in range(NUM_QUBITS): # for each qubit, create a list of gates of variable length
            gate_genes = []
            # choose a random number of gates
            num_gates = np.random.randint(1, TOTAL_GATES)
            for _ in range(num_gates): # for each gate
                # choose a random gate
                gate = np.random.choice(list(gate_map.keys()))
                if gate in ['RX', 'RY', 'RZ', 'P']:
                    # choose a random parameter
                    param = np.random.uniform(0, 2*np.pi)
                    gate_genes.append([gate, param])
                else:
                    gate_genes.append([gate, None])
            genes.append(gate_genes)
        population.append(Individual(genes))
    print('Created population of {} individuals.'.format(pop_size))
    return population

# select individuals based on their fitness
def select(population, fitnesses, num_parents):
    '''Select individuals based on their fitness. Lower fitness is better.'''
    # sort the population by fitness
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
    # return the best individuals
    return sorted_population[:num_parents]

# run the genetic algorithm
def run_ga(target, num_qubits, pop_size = 10, epochs = 50, mut_rate = 0.1):
    '''Run the genetic algorithm according to population size for epochs and individual mutation rate per epoch. Input target is the target is the desired unitary.'''
    global TARGET_UNITARY
    global NUM_QUBITS
    TARGET_UNITARY = target
    NUM_QUBITS = num_qubits

    print('Running genetic algorithm with {} qubits, {} depth, {} population size, {} epochs, and {} mutation rate.'.format(num_qubits, depth, pop_size, epochs, mut_rate))
    
    # initialize population
    population = initialize_population(pop_size)

    # evolution loop
    for generation in trange(epochs):
        # evaluate fitness
        print('Evaluating fitness...')
        fitnesses = [ind.get_fitness() for ind in population]

        # selection
        print('Selecting parents...')
        parents = select(population, fitnesses, pop_size // 2)

        # crossover -- balances exploration (searching new areas of the solution space) and exploitation (refining current promising areas); avoids local minima
        print('Crossover...')
        offspring = [crossover(parents[i], parents[-i-1]) for i in range(len(parents)//2)]

        # mutation
        print('Mutation...')
        population = parents + [ind.mutate(mut_rate) for ind in offspring]

        # print best fitness in generation
        print(f"Epoch {generation}: Best Fitness = {max(fitnesses)}")

    # best solution
    best_individual = population[np.argmin(fitnesses)]
    print("Best Individual:", best_individual.get_circ())
    # actual inner product
    print("Actual inner product:", best_individual.get_fitness_ip())
    print('Number of gates in approximation:', best_individual.num_gates)
    print('Number of gates in target:', TOTAL_GATES)
    print('Actual circuit:', TARG_CIRC)
    return best_individual
        
if __name__ == "__main__":
    # Parameters
    num_qubits = 6
    depth = 10
    pop_size = 1000
    epochs = 500
    mut_rate = 0.1

    TARG_CIRC = get_random_circuit(num_qubits, depth)
    TARG_UNITARY = cirq.unitary(TARG_CIRC)
    TOTAL_GATES = depth * num_qubits

    # run the genetic algorithm
    best_individual = run_ga(TARG_UNITARY, num_qubits, pop_size, epochs, mut_rate)
