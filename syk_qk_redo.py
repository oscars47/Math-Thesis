# file to redo the syk protocol using the jafferis et al 2022 paper

import numpy as np
from qiskit import transpile, QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator, DensityMatrix, partial_trace, entropy
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import PauliEvolutionGate, EfficientSU2
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA, NumPyMinimumEigensolver
from qiskit.circuit import Parameter
from qiskit.opflow import I, X, Y, Z, PauliSumOp, PauliOp
import matplotlib.pyplot as plt
import os, time, json
from math import factorial
from tqdm import trange



def time_evolve(H, tf=1):
    '''Time evolves the Hamiltonian H from t0 to tf in steps.'''
    print('Time evolving...')
    # get the circuit 
    # first convert paulisumop to pauli evolution gate
    H_evo = PauliEvolutionGate(H, time=tf)
    H_circ = QuantumCircuit(H.num_qubits)
    H_circ.append(H_evo, range(H.num_qubits))

    # time = Parameter('t')
    # # Define the evolution
    # trotter_evolution = PauliTrotterEvolution(trotter_mode='suzuki', reps=2)
    # evolved_op = trotter_evolution.convert(H.exp_i(), evo_time=time)

    # # Convert to Quantum Circuit
    # H_circ = evolved_op.to_circuit()
    # H_circ.draw('mpl')
    # plt.show()

    # log the total number of gates in the qc
    num_gates_initial_dict = H_circ.count_ops()
    num_gates_initial = sum(num_gates_initial_dict.values())
    print('Number of gates in initial circuit:', num_gates_initial)

    # run transpiler
    H_opt_circ = transpile(H_circ, optimization_level=1, basis_gates=['cx', 'u3'])
    print('Transpiled...')

    # log the number of gates in the optimized qc
    num_gates_opt_dict = H_opt_circ.count_ops()
    num_gates_opt = sum(num_gates_opt_dict.values())
    print('Number of gates in optimized circuit:', num_gates_opt)

    return H_opt_circ # just return the optimized circuit

## ------ wormhole protocol functions ------- ##
def majorana_to_qubit_op(j, n_qubits, left=True):
    '''Performs Jordan Wigner transformation from Majorana to qubit operators, as in Jafferis et al 2022.

    psi_L = 1/sqrt(2) Z^k X
    psi_R = 1/sqrt(2) Z^k Y

    Params:
        l (int): index of Majorana operator
        num_qubits (int): number of qubits
        left (bool): whether the Majorana operator is on the left or right side 
    
    '''
    l = (j+1) // 2

    # start with identity operator for all qubits
    operator_chain = [I*1/np.sqrt(2)] * n_qubits
    # add Z k times
    for i in range(l):
        operator_chain[i] = Z
    if left:
        # add X
        operator_chain[-1] = X
    else:
        # add Y
        operator_chain[-1] = Y

    # now combine
    # convert the list of single-qubit operators to a PauliOp
    full_operator = operator_chain[0]
    for op in operator_chain[1:]:
        full_operator = full_operator ^ op

    return full_operator

def get_SYK(n_majorana, J=2, left=True):
    '''Returns the SYK Hamiltonian as a PauliSumOp object in Qiskit.

    Params:
        N (int): number of Majorana fermions
        J (float): effective variance
        save (bool): whether to save the matrix representation of the circuit
        save_name (str): name of the file to save as
    
    '''

    print(f'Generating SYK Hamiltonian for N_m = {n_majorana}...')

    # Parameters
    n_qubits = n_majorana // 2

    var = factorial(3) * J**2 / (n_majorana**3)
    J = np.random.normal(0, np.sqrt(var), (n_majorana, n_majorana, n_majorana, n_majorana))

    # Ensure Jijkl is antisymmetric
    for i in range(n_majorana):
        for j in range(n_majorana):
            for k in range(n_majorana):
                for l in range(n_majorana):
                    if i >= j or j >= k or k >= l:
                        J[i, j, k, l] = 0                     

    # initialize Hamiltonian as all 0s
    H = PauliSumOp.from_list([("I" * n_qubits, 0.0)]) 

    for i in range(n_majorana):
        for j in range(n_majorana):
            for k in range(n_majorana):
                for l in range(n_majorana):
                    if J[i, j, k, l] != 0:
                        term = majorana_to_qubit_op(i, n_qubits, left=left) @ majorana_to_qubit_op(j, n_qubits,left=left) @ majorana_to_qubit_op(k, n_qubits, left=left) @ majorana_to_qubit_op(l,  n_qubits, left=left)
                        H += J[i, j, k, l] * term

    return H

def get_H_LR(N_m, J=2):
    '''prepare syk hamiltonians for L and R systems'''

    H_L = get_SYK(N_m, J= J, left=True)
    H_R = get_SYK(N_m, J =J, left=False)

    identity_N = I ^ (N_m//2)

    # Initialize the Hamiltonian for the TFD state
    H_LR = PauliSumOp.from_list([("I" * N_m, 0.0)])

    # Embed H_L in the left N qubits and H_R in the right N qubits
    for term_L in H_L:
        # Tensor product of term_L with identity on the right
        H_LR += term_L.tensor(identity_N)

    for term_R in H_R:
        # Tensor product of identity on the left with term_R
        H_LR += identity_N.tensor(term_R)

    return H_LR, H_L, H_R

def get_V(n_majorana):
    '''Returns the V operator for the SYK model as a PauliSumOp object in Qiskit.'''
    print(f'Generating V operator for N = {n_majorana}...')
    V = PauliSumOp.from_list([("I" * n_majorana, 0.0)])
    # go through all majorana operators
    for i in range(n_majorana):
        term_L = majorana_to_qubit_op(i, n_majorana//2, left=True)
        term_R = majorana_to_qubit_op(i, n_majorana//2, left=False)
        # term_L has a bunch of Is at the end to match n_majorana
        # term_R has a bunch of Is at the beginning to match n_majorana
        # so we need to add the correct number of Is to the other term
        term_L = term_L ^ (I ^ (n_majorana//2))
        term_R = (I ^ (n_majorana//2 )) ^ term_R
        
        term  = term_L @ term_R  
        V += term

    # weight by 1/4*N
    V = 1/(4*n_majorana) * V
    return V

def run_VQE(H_LR, V, beta=4, ans=0, display_circs=False, benchmark=False):
    '''Run VQE on the Hamiltonian H using my circuit ansatz

    Params:
        H_LR (PauliSumOp): Hamiltonian to run VQE on
        beta (float): inverse temperature
        ans (int): which ansatz to use:
            0: EfficientSU2
            1: U(3) on all + cnot chain
            2: U(3) on all + cnot ladder + cnot chain
        display_circs (bool): whether to display the circuits
        benchmark (bool): whether to compare the min eigenvalue to the exact value with NumPyMinimumEigensolver
    
    '''

    N = H_LR.num_qubits//2
    H_TFD = H_LR
    H_TFD += 1j*beta*V

    if ans == 0:
        ansatz = EfficientSU2(2*N)

    elif ans == 1 or ans == 2:
        ansatz = QuantumCircuit(2*N)
        # define here. each qubit i gets a U(3) gate and then CNOT to next qubit i+1 and to the corresponding one at i + N/2
        # Define parameters for the U3 gates
        # Creating a list of three parameters for each qubit
        theta = [Parameter(f'θ_{i}') for i in range(4*N)]
        phi = [Parameter(f'φ_{i}') for i in range(4*N)]
        lambda_ = [Parameter(f'λ_{i}') for i in range(4*N)]

        # Apply parameterized U(3) gate on each qubit
        for i in range(2*N):
            ansatz.u3(theta[i], phi[i], lambda_[i], i)

        if ans == 1:    
            # Apply CNOT to connect each qubit i to i+1 and also to i + N/2 (for even N)
            for i in range(2*N-1):
                ansatz.cx(i, i+1)

        elif ans == 2:
            # Apply CNOT to connect each qubit i to i+1 and also to i + N/2 (for even N)
            for i in range(N):
                # Linear entanglement to next qubit
                if (i+1) % N != 0:
                    ansatz.cx(i, (i+1) % N)  # Wrap around using modulo N
                
                # Circular entanglement to qubit i + N/2, ensuring it wraps within the index
                if (i + N) < 2*N:
                    ansatz.cx(i, i + N)
        # Apply parameterized U(3) gate on each qubit
        for i in range(2*N, 4*N):
            ansatz.u3(theta[i], phi[i], lambda_[i], i)

    # Display the circuit
    if display_circs:
        ansatz.draw('mpl')
        plt.savefig('results_new/vqe_circuit.pdf')

    # now perform VQE
    # Set up the optimizer
    optimizer = SPSA(maxiter=100)

    # Set up the backend and quantum instance
    seed = 47
    algorithm_globals.random_seed = seed
    backend = Aer.get_backend('aer_simulator_statevector')
    quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

    # Run VQE with your custom ansatz
    vqe = VQE(ansatz, optimizer, quantum_instance=quantum_instance)
    result = vqe.compute_minimum_eigenvalue(H_TFD)
    optimal_parameters = result.optimal_parameters

    # Assuming `ansatz` is your parameterized quantum circuit used in VQE
    optimal_circuit = ansatz.assign_parameters(optimal_parameters)

    # Output the result
    if display_circs:
        optimal_circuit.draw('mpl')
        plt.title(f'Optimal VQE Circuit, E = {result.eigenvalue.real}')
        timestamp = int(time.time())
        plt.savefig(f'results_new/optimal_circuit_{timestamp}.pdf')

    if not benchmark:
        print(f"Minimum eigenvalue: {result.eigenvalue.real}")
        return optimal_circuit
    else:
        min_eig = result.eigenvalue.real
        # now get the exact value
        # Initialize the solver
        solver = NumPyMinimumEigensolver()

        # Find the ground state
        result = solver.compute_minimum_eigenvalue(operator=H_TFD)

        # Extract the ground state energy
        ground_state_energy = result.eigenvalue.real

        # compare
        return np.abs(ground_state_energy - min_eig)

## ---- main ---- ##
def protocol_round(H_R, tfd, expV, tev_nt0, tev_pt0, t,steps=100, save_circs=False):
    '''Implements a single round of the the wormhole protocol for the SYK model with n_majorana fermions at time t, interaction param mu, and inverse temperature beta.'''
    ## STEP 1: generate TFD and apply negative time evolution on L
    # make tfd
    total_circuit = QuantumCircuit(2*H_R.num_qubits + 2)

    ## STEP 2: swap in register Q of bell pair into tfd
    total_circuit.h(0)
    total_circuit.cx(0, 1)

    # swap in the bell pair
    total_circuit.swap(1, 2)

     # set the tfd within the larger full circuit with registers P, Q before it and T at the end
    for gate in tfd.data:
        qubits = [q.index + 2 for q in gate[1]]
        total_circuit.append(gate[0], qubits) # start at 2 to account for the extra registers

    # apply backwards time evolution to L part of tfd
    for gate in tev_nt0.data: 
        qubits = [q.index + 2 for q in gate[1]]
        total_circuit.append(gate[0], qubits)

     ## STEP 3: apply forward time evolution to L part of tfd
    for gate in tev_pt0.data:
        qubits = [q.index +2 for q in gate[1]]
        total_circuit.append(gate[0], qubits)

    ## STEP 4: apply expV to all tfd
    for gate in expV.data:
        qubits = [q.index + 2 for q in gate[1]]
        total_circuit.append(gate[0], qubits)

    ## STEP 5: apply forward time evolution by t1 to R part of tfd
    tev_pt1 = time_evolve(H_R, tf=t, steps=steps) # quantum circuit
    for gate in tev_pt1.data:
        qubits = [q.index + H_R.num_qubits +2 for q in gate[1]]
        total_circuit.append(gate[0], qubits)

    ## STEP 6: SWAP out qubit (skip, since we'll just measure on the last qubit)

    # optimize
    tfd_final = transpile(total_circuit, optimization_level=1, basis_gates=['cx', 'u3'])
    if save_circs:
        tfd_final.draw('mpl').savefig(f'results_new/tfd_final_{t}.pdf')
    print('number of gates:', tfd_final.count_ops())

    ## STEP 8: compute the mutual info
    # I = S(R) + S(T) - S(TR)

    # get the density matrix
    # Run the circuit on a statevector simulator backend
    backend = Aer.get_backend('statevector_simulator')
    job = execute(tfd_final, backend)
    result = job.result()

    # Get the statevector
    statevector = result.get_statevector(tfd_final)

    # Form the density matrix from the statevector
    density_matrix = DensityMatrix(statevector)

    # get the reduced density matrices
    rho_P = partial_trace(state = density_matrix, qargs = range(1, total_circuit.num_qubits))
    rho_T = partial_trace(state = density_matrix, qargs = range(total_circuit.num_qubits-1))
    rho_PT = partial_trace(state = density_matrix, qargs = range(1, total_circuit.num_qubits-1))

    # compute the mutual info
    I = entropy(rho_P) + entropy(rho_T) - entropy(rho_PT)
    print(f'Mutual info {I} at time {t}')

    return I

def full_protocol(N_m, tf = 10, ans = 0, t_steps = 10, t0= 2.8):
    '''Runs the full wormhole protocol from t = 0 to t = tf in t_steps for the SYK model with N_m fermions. ans specfiies the ansatz to use for VQE. t0 is the time for the initial negative/positive time evolution.'''

    # --- prepare the Hamiltonians --- #
    H_LR, H_L, H_R = get_H_LR(N_m)
   

    # --- prepare the V operator --- #
    V = get_V(N_m)
    expV = time_evolve(V, tf=1, steps=100)

    # --- run VQE to get the TFD state for given choice of ansatz --- #
    TFD = run_VQE(H_LR,  V, ans = ans, display_circs=False)

    # --- negative and positive time ev  --- #
    tev_nt0 = time_evolve(H_L, tf=-t0)
    tev_pt0 = time_evolve(H_L, tf=t0)

    # --- run the protocol --- #
    mutual_infos = []
    for t in np.linspace(0, tf, t_steps):
        mutual_infos.append(protocol_round(H_R, TFD, expV, tev_nt0, tev_pt0, t))

    # save the mutual infos
    if not os.path.exists('results_new'):
        os.makedirs('results_new')
    mutual_infos = np.array(mutual_infos)
    timestamp = int(time.time())
    np.save(f'results_new/mutual_infos_{N_m}_{ans}_{timestamp}.npy', mutual_infos)

    # make the plot
    plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(0, tf, t_steps), mutual_infos)
    plt.xlabel('Time')
    plt.ylabel('Mutual Info')
    plt.title(f'Mutual Info for N_m = {N_m}, ans = {ans}')
    plt.savefig(f'results_new/mutual_info_{N_m}_{ans}_{timestamp}.pdf')
    
    return mutual_infos

def benchmark_vqe(num_iter):
    '''for each ansatz 0 - 2, calculate difference between learned and exact min eigenvalue for num_iter iterations'''
    if not os.path.exists('results_new'):
        os.makedirs('results_new')

    # create dictionary for results, where each key is the ansatz and the value is a list of differences
    results = {i: [] for i in range(3)}

    for i in trange(num_iter):
        # get the hamiltonians
        H_LR, H_L, H_R = get_H_LR(N_m)
        V = get_V(N_m)
        for ans in range(3):
            diff = run_VQE(H_LR, V, ans=ans, benchmark=True)
            results[ans].append(diff)
            
            
    # save the results
    timestamp = int(time.time())
    # Save the dictionary to a file
    with open(f'results_new/benchmark_{timestamp}_{num_iter}.json', 'w') as f:
        json.dump(results, f, indent=4)

    # find the avg and std dev for each ansatz if num_iter > 1
    if num_iter > 1:
        results_avg = {i: [] for i in range(3)}
        for ans in range(3):
            avg = np.mean(results[ans])
            std = np.std(results[ans])
            results_avg[ans] = [avg, std]

        # Save the dictionary to a file
        with open(f'results_new/benchmark_avg_{timestamp}_{num_iter}.json', 'w') as f:
            json.dump(results_avg, f, indent=4)

if __name__ == '__main__':
    N_m=10

    
