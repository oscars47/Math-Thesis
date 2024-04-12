# file to redo the syk protocol using the jafferis et al 2022 paper

import numpy as np
from qiskit import transpile, QuantumCircuit, Aer, execute
from qiskit_ibm_provider import IBMProvider
from qiskit.quantum_info import Operator, DensityMatrix, partial_trace, entropy
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import PauliEvolutionGate, EfficientSU2
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA, L_BFGS_B
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.circuit import Parameter
from qiskit.opflow import I, X, Y, Z, PauliSumOp, PauliOp
from qiskit.quantum_info import partial_trace, DensityMatrix, entropy
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography
from qiskit_ibm_runtime import Options
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import os, time, json
from math import factorial
from tqdm import trange
from oscars_toolbox.trabbit import trabbit
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from math import comb
from scipy.optimize import approx_fprime

def time_evolve(H, tf=1):
    '''Time evolves the Hamiltonian H from t0 to tf in steps.'''
    # print('Time evolving...')
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
    # print('Number of gates in initial circuit:', num_gates_initial)

    # run transpiler
    H_opt_circ = transpile(H_circ, optimization_level=1, basis_gates=['cx', 'u3'])
    # print('Transpiled...')

    # log the number of gates in the optimized qc
    num_gates_opt_dict = H_opt_circ.count_ops()
    num_gates_opt = sum(num_gates_opt_dict.values())
    # print('Number of gates in optimized circuit:', num_gates_opt)

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

def get_SYK_from_params(n_majorana, params,left=True):
    '''allows defining parametrized SYK Hamiltonian as a PauliSumOp object in Qiskit.

    '''

    # reshape into 4D tensor
    params = params.reshape(n_majorana, n_majorana, n_majorana, n_majorana)

    n_qubits = n_majorana // 2

    # initialize Hamiltonian as all 0s
    H = PauliSumOp.from_list([("I" * n_qubits, 0.0)]) 


    for i in range(n_majorana):
        for j in range(n_majorana):
            for k in range(n_majorana):
                for l in range(n_majorana):
                    if params[i, j, k, l] != 0:
                        term = majorana_to_qubit_op(i, n_qubits, left=left) @ majorana_to_qubit_op(j, n_qubits,left=left) @ majorana_to_qubit_op(k, n_qubits, left=left) @ majorana_to_qubit_op(l,  n_qubits, left=left)
                        H += params[i, j, k, l] * term

    return H

def get_random_SYK_params(n_majorana, J=2, binned=False, num_bins=10):
    '''returns random SYK params for a given number of Majorana fermions, in form of n_majorana x n_majorana x n_majorana x n_majorana tensor

    Params:
        n_majorana (int): number of Majorana fermions
        J (float): effective variance
        binned (bool): whether to bin the Jijkl values
    
    '''

    var = factorial(3) * J**2 / (n_majorana**3)
    Jp = np.random.normal(0, np.sqrt(var), (n_majorana, n_majorana, n_majorana, n_majorana))

    bins = np.linspace(-np.sqrt(var), np.sqrt(var), num_bins)

    # Ensure Jijkl is antisymmetric
    for i in range(n_majorana):
        for j in range(n_majorana):
            for k in range(n_majorana):
                for l in range(n_majorana):
                    if i >= j or j >= k or k >= l:
                        Jp[i, j, k, l] = 0 
                    if binned:
                        index = np.digitize(Jp[i, j, k, l], bins)-1
                        Jp[i, j, k, l] = bins[index]

    # plot histogram of Jijkl values
    # Jp_flat = Jp.flatten()
    # plt.hist(Jp_flat[np.abs(Jp_flat)>0], bins=100)
    # plt.xlabel('Jijkl')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Jijkl values')
    # plt.show()

    return Jp

def get_SYK(n_majorana, J=2, left=True):
    '''Returns the SYK Hamiltonian as a PauliSumOp object in Qiskit.

    Params:
        N (int): number of Majorana fermions
        J (float): effective variance
        save (bool): whether to save the matrix representation of the circuit
        save_name (str): name of the file to save as
    
    '''
    # get coeffficients
    Jp = get_random_SYK_params(n_majorana, J=J)

    return get_SYK_from_params(n_majorana, Jp, left=left)

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
    # print(f'Generating V operator for N = {n_majorana}...')
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
            2: U(3) on all + cnot ladder + cnot chain (only on L)
            3: U(3) on all + every qubit connected to the next
            4: U(3) on all + every qubit connected to the next + every qubit connected to the corresponding one at (i + N) % 2N
        display_circs (bool): whether to display the circuits
        benchmark (bool): whether to compare the min eigenvalue to the exact value with NumPyMinimumEigensolver
    
    '''

    N = H_LR.num_qubits//2
    H_TFD = H_LR
    H_TFD += 1j*beta*V

    if ans == 0:
        ansatz = EfficientSU2(2*N, reps=1)
        # print out the circuit in the basis of u3 and cx gates
        # ansatz_t = transpile(ansatz, basis_gates=['cx', 'u3'])
        # if display_circs:
        # ansatz_t.draw('mpl')
        # plt.savefig('results_new/vqe_circuit_0.pdf')

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
            ansatz.u(theta[i], phi[i], lambda_[i], i)

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
        for i in range(2*N):
            ansatz.u(theta[i+2*N], phi[i+2*N], lambda_[i+2*N], i)
    elif ans == 3:
        ansatz = QuantumCircuit(2*N)
        # define here. each qubit i gets a U(3) gate and then CNOT to next qubit i+1 and to the corresponding one at i + N/2
        # Define parameters for the U3 gates
        # Creating a list of three parameters for each qubit
        theta = [Parameter(f'θ_{i}') for i in range(6*N)]
        phi = [Parameter(f'φ_{i}') for i in range(6*N)]
        lambda_ = [Parameter(f'λ_{i}') for i in range(6*N)]

        # Apply parameterized U(3) gate on each qubit
        for i in range(2*N):
            ansatz.u(theta[i], phi[i], lambda_[i], i)

        for i in range(N):
            # Linear entanglement to next qubit
            if (i+1) % N != 0:
                ansatz.cx(i, (i+1) % N)  # Wrap around using modulo N
            
            # Circular entanglement to qubit i + N/2, ensuring it wraps within the index
            if (i + N) < 2*N:
                ansatz.cx(i, i + N)

        # Apply parameterized U(3) gate on each qubit
        for i in range(2*N):
            ansatz.u(theta[i+2*N], phi[i+2*N], lambda_[i+2*N], i)

        for i in range(N):
            # Linear entanglement to next qubit
            if (i+1) % N != 0:
                ansatz.cx(i, (i+1) % N)  # Wrap around using modulo N
            
            # Circular entanglement to qubit i + N/2, ensuring it wraps within the index
            if (i + N) < 2*N:
                ansatz.cx(i, i + N)
                
        # Apply parameterized U(3) gate on each qubit
        for i in range(2*N):
            ansatz.u(theta[i+4*N], phi[i+4*N], lambda_[i+4*N], i)

    elif ans == 4:
        theta = [Parameter(f'θ_{i}') for i in range(4*N)]
        phi = [Parameter(f'φ_{i}') for i in range(4*N)]
        lambda_ = [Parameter(f'λ_{i}') for i in range(4*N)]

        ansatz = QuantumCircuit(2*N)
        # Apply parameterized U(3) gate on each qubit
        for i in range(2*N):
            ansatz.u(theta[i], phi[i], lambda_[i], i)

        # apply CNOTs
        for i in range(2*N):
            ansatz.cx(i, (i+1) % (2*N))
            ansatz.cx(i, (i+N) % (2*N))

        # Apply parameterized U(3) gate on each qubit
        for i in range(2*N):
            ansatz.u(theta[i+2*N], phi[i+2*N], lambda_[i+2*N], i)

    # Display the circuit
    if display_circs:
        ansatz.draw('mpl')
        plt.savefig(f'results_new/vqe_circuit_{ans}.pdf')

    # now perform VQE
    # Set up the optimizer
    # optimizer = SPSA(maxiter=1000)
    optimizer = L_BFGS_B(max_evals_grouped=100)

    # Set up the backend and quantum instance
    seed = 47
    algorithm_globals.random_seed = seed
    backend = Aer.get_backend('aer_simulator_statevector')
    quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

    # Run VQE with custom ansatz
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
        
        print(f"Minimum eigenvalue: {result.eigenvalue.real}")
    if not benchmark:
        return optimal_circuit
    else:
        min_eig = result.eigenvalue.real
        print(f"Minimum eigenvalue: {min_eig}")
        print('Running exact solve...')
        # now get the exact value
        # Initialize the solver
        solver = NumPyMinimumEigensolver()

        # Find the ground state
        result = solver.compute_minimum_eigenvalue(operator=H_TFD)

        # Extract the ground state energy
        ground_state_energy = result.eigenvalue.real
        print(f"Ground state energy: {ground_state_energy}")

        # compare as abs fractional error
        return np.abs(ground_state_energy - min_eig) / np.abs(ground_state_energy)

def compute_mi(circuit, display_circs=None, save_param=None):
    ''' computes mutual info between first and last qubit'''
    # I = S(R) + S(T) - S(TR)
    n_qubits = circuit.num_qubits

    # first transpile
    tfd_final = transpile(circuit, optimization_level=1, basis_gates=['cx', 'u3'])
    if display_circs:
        print('Number of gates:', tfd_final.count_ops())
        if save_param is not None:
            tfd_final.draw('mpl').savefig(f'results_new/total_circuit_{save_param}.pdf')
        else:
            tfd_final.draw('mpl').savefig('results_new/total_circuit.pdf')
    # get the density matrix
    # Run the circuit on a statevector simulator backend
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()

    # Get the statevector
    statevector = result.get_statevector(circuit)

    # Form the density matrix from the statevector
    density_matrix = DensityMatrix(statevector)

    # get the reduced density matrices
    rho_P = partial_trace(state = density_matrix, qargs = range(1, n_qubits))
    rho_T = partial_trace(state = density_matrix, qargs = range(n_qubits-1))
    rho_PT = partial_trace(state = density_matrix, qargs = range(1, n_qubits-1))

    # compute the mutual info
    return entropy(rho_P) + entropy(rho_T) - entropy(rho_PT)

def compute_mi_actual(circuit, backend, shots=10000):
    ''' Computes mutual info between first and last qubit using Qiskit Experiments for state tomography.'''

    # Setup state tomography on the first and last qubits
    tomo_experiment = StateTomography(circuit, measurement_indices=[0, circuit.num_qubits - 1])

    # Run the state tomography experiment
    experiment_data = tomo_experiment.run(backend, shots=shots).block_for_results()

    # Access analysis results directly
    analysis_results = experiment_data.analysis_results()

    # Process analysis results as needed
    tomo_result = analysis_results[0].value

    # Compute the mutual information
    # Get the reduced density matrices
    rho_P = partial_trace(tomo_result, [0])
    rho_T = partial_trace(tomo_result, [1])
    rho_PT = tomo_result  # The joint state of the first and last qubits

    # Compute the mutual info
    mutual_info = entropy(rho_P) + entropy(rho_T) - entropy(rho_PT)
    return mutual_info

## ---- main ---- ##
def protocol_round(H_R, tfd, expV, tev_nt0, tev_pt0, t, display_circs=False, save_param=None):
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
    tev_pt1 = time_evolve(H_R, tf=t) 
    for gate in tev_pt1.data:
        qubits = [q.index + H_R.num_qubits +2 for q in gate[1]]
        total_circuit.append(gate[0], qubits)

    ## STEP 6: SWAP out qubit (skip, since we'll just measure on the last qubit)

    I = compute_mi(total_circuit, display_circs=display_circs, save_param=save_param)
    if display_circs:
        print('Mutual info:', I)
    # print(f'Mutual info at t = {t}: {I}')
    return I

def full_protocol(N_m, tf = 10, ans = 0, t_steps = 10, t0= 2.8, mu=-12, subdir=None, display_circs=False, plot_result=False):
    '''Runs the full wormhole protocol from t = 0 to t = tf in t_steps for the SYK model with N_m fermions. ans specfiies the ansatz to use for VQE. t0 is the time for the initial negative/positive time evolution.
    
    Params:
        N_m (int): number of Majorana fermions
        tf (float): final time
        ans (int): which ansatz to use for VQE
        t_steps (int): number of steps to divide the time interval into
        t0 (float): time for the initial negative/positive time evolution
        mu (float): interaction parameter
        subdir (str): name of the subdirectory (of results_new) to save the results in 
        display_circs (bool): whether to display the circuits
        plot_result (bool): whether to plot the mutual info at the end
    
    '''

    # --- prepare the Hamiltonians --- #
    H_LR, H_L, H_R = get_H_LR(N_m)
   
    # --- prepare the V operator --- #
    V = get_V(N_m)
    # multiply by mu
    V = mu * V
    expV = time_evolve(V, tf=1)

    # --- run VQE to get the TFD state for given choice of ansatz --- #
    TFD = run_VQE(H_LR,  V, ans = ans, display_circs=display_circs, benchmark=False)

    # --- negative and positive time ev  --- #
    tev_nt0 = time_evolve(H_L, tf=-t0)
    tev_pt0 = time_evolve(H_L, tf=t0)

    if display_circs:
        tev_nt0.draw('mpl')
        plt.savefig('results_new/tev_nt0.pdf')
        tev_pt0.draw('mpl')
        plt.savefig('results_new/tev_pt0.pdf')

    # --- run the protocol --- #
    mutual_infos = []
    for t in np.linspace(0, tf, t_steps):
        mutual_infos.append(protocol_round(H_R, TFD, expV, tev_nt0, tev_pt0, t, display_circs=display_circs))

    # save the mutual infos
    if subdir is  None:
        if not os.path.exists('results_new'):
            os.makedirs('results_new')
        save_dir = 'results_new'
    else:
        if not os.path.exists(f'results_new/{subdir}'):
            os.makedirs(f'results_new/{subdir}')
        save_dir = f'results_new/{subdir}'

    mutual_infos = np.array(mutual_infos)
    timestamp = int(time.time())
    np.save(os.path.join(save_dir, f'mutual_infos_{N_m}_{ans}_{mu}_{timestamp}.npy'), mutual_infos)

    # make the plot
    if plot_result:
        plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(0, tf, t_steps), mutual_infos)
        plt.xlabel('Time')
        plt.ylabel('Mutual Information')
        plt.title(f'Mutual Information for $N_m = {N_m}$, ans = {ans}, $\mu = {mu}$')
        plt.savefig(os.path.join(save_dir, f'mutual_info_{N_m}_{ans}_{mu}_{timestamp}.pdf'))
        
    return mutual_infos

def repeat_full_protocol(N_m, num_reps=5, ans=0, tf=10, t_steps=10, mu=-12, t0=2.8, subdir=None, display_circs=False):
    '''Runs the full wormhole protocol from t = 0 to t = tf in t_steps for the SYK model with N_m fermions. ans specfiies the ansatz to use for VQE. t0 is the time for the initial negative/positive time evolution.
    
    Params:
        N_m (int): number of Majorana fermions
        tf (float): final time
        ans (int): which ansatz to use for VQE
        t_steps (int): number of steps to divide the time interval into
        t0 (float): time for the initial negative/positive time evolution
        mu (float): interaction parameter
        subdir (str): name of the subdirectory (of results_new) to save the results in 
        display_circs (bool): whether to display the circuits
    
    '''

    # --- prepare the Hamiltonians --- #
    H_LR, H_L, H_R = get_H_LR(N_m)

    for _ in trange(num_reps):
   
        # --- prepare the V operator --- #
        V = get_V(N_m)
        # multiply by mu
        V = mu * V
        expV = time_evolve(V, tf=1)

        # --- run VQE to get the TFD state for given choice of ansatz --- #
        TFD = run_VQE(H_LR,  V, ans = ans, display_circs=display_circs, benchmark=False)

        # --- negative and positive time ev  --- #
        tev_nt0 = time_evolve(H_L, tf=-t0)
        tev_pt0 = time_evolve(H_L, tf=t0)

        if display_circs:
            tev_nt0.draw('mpl')
            plt.savefig('results_new/tev_nt0.pdf')
            tev_pt0.draw('mpl')
            plt.savefig('results_new/tev_pt0.pdf')

        # --- run the protocol --- #
        mutual_infos = []
        for t in np.linspace(0, tf, t_steps):
            mutual_infos.append(protocol_round(H_R, TFD, expV, tev_nt0, tev_pt0, t, display_circs=display_circs))

        # save the mutual infos
        if subdir is  None:
            if not os.path.exists('results_new'):
                os.makedirs('results_new')
            save_dir = 'results_new'
        else:
            if not os.path.exists(f'results_new/{subdir}'):
                os.makedirs(f'results_new/{subdir}')
            save_dir = f'results_new/{subdir}'

        mutual_infos = np.array(mutual_infos)
        timestamp = int(time.time())
        np.save(os.path.join(save_dir, f'mutual_infos_{N_m}_{ans}_{mu}_{timestamp}.npy'), mutual_infos)

        # make the plot
        plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(0, tf, t_steps), mutual_infos)
        plt.xlabel('Time')
        plt.ylabel('Mutual Information')
        plt.title(f'Mutual Information for $N_m = {N_m}$, ans = {ans}, $\mu = {mu}$')
        plt.savefig(os.path.join(save_dir, f'mutual_info_{N_m}_{ans}__{mu}_{timestamp}.pdf'))

## ---- benchmarking functions ---- ##
def run_vqe_for_ansatz(ans, H_LR, V, display_circs=False):
        '''helper function for parallelization'''
        diff = run_VQE(H_LR, V, ans=ans, benchmark=True, display_circs=display_circs)  # Assuming run_VQE is defined elsewhere
        return ans, diff

def benchmark_vqe(N_m, num_iter, max_ans=4, display_circs=False):
    '''for each ansatz 0 - 2, calculate difference between learned and exact min eigenvalue for num_iter iterations'''
    if not os.path.exists('results_new'):
        os.makedirs('results_new')

    # create dictionary for results, where each key is the ansatz and the value is a list of differences
    results = {i: [] for i in range(max_ans+1)}

    for i in trange(num_iter):
        # get the hamiltonians
        H_LR, H_L, H_R = get_H_LR(N_m)
        V = get_V(N_m)
        # print('Running VQE...')
        
        local_results = []
        # Execute the tasks in parallel
        with ProcessPoolExecutor() as executor:
            # Submit all VQE runs to the executor
            futures = [executor.submit(run_vqe_for_ansatz, ans, H_LR, V, display_circs) for ans in range(max_ans+1)]
            
            # Process the results as they complete
            for future in as_completed(futures):
                ans, diff = future.result()
                local_results.append((ans, diff))
                print(f'Ansatz {ans}: {diff}')

        # assign the results to the global dictionary
        for ans, diff in local_results:
            results[ans].append(diff)
            
    # save the results
    timestamp = int(time.time())
    # Save the dictionary to a file
    with open(f'results_new/benchmark_{timestamp}_{num_iter}.json', 'w') as f:
        json.dump(results, f, indent=4)

    # find the avg and std dev for each ansatz if num_iter > 1
    if num_iter > 1:
        results_avg = {i: [] for i in range(3)}
        for ans in range(4):
            avg = np.mean(results[ans])
            std = np.std(results[ans])
            results_avg[ans] = [avg, std]

        # Save the dictionary to a file
        with open(f'results_new/benchmark_avg_{timestamp}_{num_iter}.json', 'w') as f:
            json.dump(results_avg, f, indent=4)

def run_full_protocol(params):
    '''objective function for parallelization'''
    N_m, ans, t_steps, mu, subdir = params
    full_protocol(N_m, ans=ans, t_steps=t_steps, mu=mu, subdir=subdir)

def benchmark_mi(N_m, num_reps=5):
    '''computes the mutual info for each ansatz for num_reps repetitions'''
    # make one overall subdir based on time
    subdir = str(time.time())
    
    # flatten loops into series of tasks
    tasks = []
    for mu in [-12, -6, 0, 6, 12]:
        for ans in range(4):
            for _ in range(num_reps):
                tasks.append((N_m, ans, 10, mu, subdir))

    # execute the tasks in parallel
    with ProcessPoolExecutor() as executor:
        # submit all tasks to the executor
        futures = [executor.submit(run_full_protocol, task) for task in tasks]

        for future in as_completed(futures): # optionally handle returns here
            try:
                future.result()  # If the function returns something, you can capture it here
            except Exception as e:
                print(f"Task generated an exception: {e}")

## ---- reconstructing the mutual info with preset circuit ---- ##
def get_ansatz(config, simulate=True, resilience_level=1):
    '''generates the parametrized ansatz for the given number of qubits. Default is U3 gates on each qubit and then entangling all qubits in a chain.'''
    ansatz = QuantumCircuit(config)
    # Define parameters for the U3 gates
    theta = [Parameter(f'θ_{i}') for i in range(config-1)]
    phi = [Parameter(f'φ_{i}') for i in range(config-1)]
    lambda_ = [Parameter(f'λ_{i}') for i in range(config-1)]

    # put hadamard on first qubit
    ansatz.h(0)

    # Apply parameterized U(3) gate on each qubit
    for i in range(config-1):
            ansatz.u(theta[i], phi[i], lambda_[i], i+1)

    # Apply CNOT to connect each qubit i to i+1 
    for i in range(config-1):
        ansatz.cx(i, i+1)

    if simulate:
        ansatz = transpile(ansatz, optimization_level=1, basis_gates=['cx', 'u3'], resilience_level=resilience_level)
    else: # ECR, ID, RZ, SX, X
        ansatz = transpile(ansatz, optimization_level=1, basis_gates=['ecr', 'id', 'rz', 'sx', 'x'],resilience_level=resilience_level)

    return ansatz

def learn_point(mi_target, config):
    '''learns an ansatz specified by config to reproduce a given mutual information point

    config is actually the number of qubits, default architecture is the same: apply U3 gates to each qubit and then entangle all qubits in a chain, measure MI between initial and final qubit 

    '''
    # get the ansatz
    ansatz = get_ansatz(config)

    # define objective function
    def objective_function(ansatz, params):
        '''objective function for the optimizer'''
        # apply the parameters to the ansatz
        ansatz = ansatz.assign_parameters(params)
        # calculate the mutual info
        mi_learned = compute_mi(ansatz)
        # return the abs difference
        return np.abs(mi_target - mi_learned)
    
    # function for random initialization
    def init_params():
        '''initializes the parameters randomly'''
        return np.random.uniform(0, 2*np.pi, 3*(config-1))
    
    loss_func = lambda x: objective_function(ansatz, x)
    
    # run the optimizer
    x_best, loss_best = trabbit(loss_func, init_params, alpha=0.7, temperature=0.01)
    print(f'Best loss: {loss_best}')
    return x_best, loss_best

def reconstruct_total(mi_path='mi_data/mi_2.csv', config=3):
    '''reconstructs the mutual info for a given set of mutual info points using the specified ansatz'''
    
    mi_ls = np.loadtxt(mi_path, delimiter=',')
    # get the second column
    mi_ls = mi_ls[:, 1]
    
    angles_ls = []
    loss_ls = []
    for mi in mi_ls:
        angles, loss = learn_point(mi, config)
        angles_ls.append(angles)
        loss_ls.append(loss)
    # save the results
    timestamp = int(time.time())
    np.save(f'results_new/angles_{config}_{timestamp}.npy', angles_ls)
    np.save(f'results_new/loss_{config}_{timestamp}.npy', loss_ls)
    # print out avg loss
    print(f'Avg loss: {np.mean(loss_ls)}')
    print(f'SEM dev loss: {np.std(loss_ls) / np.sqrt(len(loss_ls) - 1)}')
    
def run_reconstruction(angles_path='results_new/angles_3_1708293092.npy', t_path = 'mi_data/mi_2.csv', num_times = 5, config=3, simulate=True, resilience_level=1, shots=10000):
    '''runs the reconstructed circuit with the learned parameters, angles, and either simulates or runs on hardware'''

    if not simulate:
        shots=2000

    # load the angles
    angles = np.load(angles_path)
    I_ls = []
    I_sem_ls = []
    if simulate:
        # backend = Aer.get_backend('statevector_simulator')
        provider = IBMProvider()
        backend = provider.get_backend('ibmq_qasm_simulator')
    else:
        provider = IBMProvider()
        backend = provider.get_backend('ibm_kyoto')

    # def execute_task(angle, config, simulate, backend):
    #     '''objective function for the optimizer'''
    #     ansatz = get_ansatz(config, simulate=simulate)
    #     ansatz = ansatz.assign_parameters(angle)
    #     return compute_mi_actual(ansatz, backend, shots=shots)

    # if simulate:
    for angle in angles:
        I_angles = []
        for i in range(num_times):
            # get the ansatz
            ansatz = get_ansatz(config, simulate=False, resilience_level=resilience_level)
            print(f'angle: {angle}, time: {i}')
            print('Number of gates:', ansatz.count_ops())
            print('Number of qubits:', ansatz.num_qubits)
            # apply the parameters to the ansatz
            ansatz = ansatz.assign_parameters(angle)
            # simulate or run on hardware
            I = compute_mi_actual(ansatz, backend, shots=shots)
            I_angles.append(I)

        I_ls.append(np.mean(I_angles))
        I_sem_ls.append(np.std(I_angles) / np.sqrt(num_times - 1))

    # else:
    #    # Function to manage the concurrent execution of tasks, limiting to 3 at a time
    #     def manage_tasks():
    #         results = []
    #         futures = []

    #         with ThreadPoolExecutor(max_workers=3) as executor:
    #             # Submit the first batch of tasks
    #             for angle in angles:
    #                 for _ in range(num_times):
    #                     if len(futures) < 3:
    #                         future = executor.submit(execute_task, angle, config, simulate, backend)
    #                         futures.append((future, angle))
    #                     else:
    #                         break

    #             while futures:
    #                 # Wait for the first future to complete
    #                 done, _ = as_completed(futures, timeout=None, return_when='FIRST_COMPLETED').__next__()

    #                 # Process completed future
    #                 for future, angle in futures:
    #                     if future == done:
    #                         try:
    #                             result = future.result()
    #                             results.append((angle, result))
    #                             futures.remove((future, angle))  # Remove this task from the list
    #                             break
    #                         except Exception as e:
    #                             print(f"Task for angle {angle} resulted in an error: {e}")

    #                 # Submit a new task if there are angles left to process
    #                 for angle in angles:
    #                     if len(futures) < 3:
    #                         new_future = executor.submit(execute_task, angle, config, simulate, backend)
    #                         futures.append((new_future, angle))
    #                         break

    #         return results

    #     results = manage_tasks()

    #     # Process and save the results
    #     # Assuming you want to compute mean and SEM for each angle
    #     angles_processed = set(angle for angle, _ in results)
    #     I_ls = []
    #     I_sem_ls = []

    #     for angle in angles_processed:
    #         angle_results = [result for angle_res, result in results if angle_res == angle]
    #         mean_I = np.mean(angle_results)
    #         sem_I = np.std(angle_results, ddof=1) / np.sqrt(len(angle_results))
    #         I_ls.append(mean_I)
    #         I_sem_ls.append(sem_I)

    # save the results
    timestamp = int(time.time())
    np.save(f'results_new/I_{config}_{timestamp}_{simulate}.npy', I_ls)
    np.save(f'results_new/I_sem_{config}_{timestamp}_{simulate}.npy', I_sem_ls)

    # plot
    # compare to the actual mutual info
    t_I_ls = np.loadtxt(t_path, delimiter=',')
    t_ls = t_I_ls[:, 0]
    I_actual_ls = t_I_ls[:, 1]
    plt.figure(figsize=(10, 10))
    plt.errorbar(t_ls, I_ls, yerr=I_sem_ls, fmt='o', color='red', label='Reconstructed')
    plt.scatter(t_ls, I_actual_ls, label='Actual', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.title(f'Mutual Information for Reconstructed Circuit, config = {config}')
    plt.savefig(f'results_new/I_{config}_{timestamp}_{simulate}.pdf')
    plt.show()

def plot_angles(angles_path='results_new/angles_3_1708293092.npy'):
    '''plots the angles for the reconstructed circuit'''
    angles = np.load(angles_path)
    print(angles)
    plt.figure(figsize=(5, 5))
    # plot each column of angles as a line
    # get colorwheel by number of columns
    colors = plt.cm.viridis(np.linspace(0, 1, angles.shape[1]))
    for i in range(angles.shape[1]):
        plt.plot(angles[:, i], label=f'Angle {i+1}', color=colors[i], marker='o', alpha=0.7)
    plt.xlabel('Instance')
    plt.ylabel('Angle')
    # move the legend outside the plot
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title('Angles for Reconstructed Circuit')
    plt.savefig('results_new/angles.pdf')
    plt.show()

## ---- alternate strategy for simplifying Hamiltonians ---- ##
def loss_h(params, H_targ_mat, left, lambda_fix=None):
    '''loss function for the left Hamiltonian with regularization term lambda_.

    Params:
        params (np.array): parameters for the Hamiltonian
        H_targ_mat (np.array): target Hamiltonian
        left (bool): whether to simplify the left or right Hamiltonian
        lambda_fix (float): if not None, will fix the lambda_ value to this value, so leave as None if you want to optimize for lambda_ as well
    
    '''
    if lambda_fix is None:
        lambda_ = params[-1]
        params = params[:-1]
    else:
        lambda_ = lambda_fix
    # get the Hamiltonian
    H_pred = get_SYK_from_params(N_m, left=left, params=params).to_matrix()
    # return the norm of diff
    return np.abs(np.linalg.norm(H_targ_mat - H_pred) + lambda_ * np.linalg.norm(params))

def random_h_coeff(N_m, lambda_fix=None, binned=False): 
    coeff = get_random_SYK_params(N_m, binned=binned)
    coeff = coeff.flatten()
    # randomly set all but 10% of the coefficients to 0
    # mask = np.random.choice([0, 1], size=coeff.shape, p=[0.9, 0.1])
    # coeff = coeff * mask
    if lambda_fix is None: # if no lambda provided, will assume we want to optimize for that as well
        lambda_ = np.random.uniform(0, 1)
        return np.append(coeff, lambda_)
    else:
        print(f'num non-zero terms: {np.count_nonzero(coeff)}')
        return coeff
    
def metropolis_step(current_params, current_loss, loss_function, temperature):
    '''implement single step of Metropolis-Hastings algorithm for MCMC optimization of the loss function.'''
    perturbation = np.random.normal(0, 1, current_params.shape)
    new_params = current_params + temperature * perturbation
    
    # compute loss for new parameters
    new_loss = loss_function(new_params)
    
    # compute acceptance probability
    if new_loss < current_loss:
        accept = True
    else:
        delta_loss = current_loss - new_loss
        accept_probability = np.exp(delta_loss / temperature)
        accept = np.random.rand() < accept_probability
    
    # accept or reject the new parameters
    if accept:
        return new_params, new_loss
    else:
        return current_params, current_loss

def mcmc_optimize(loss_function, initial_params, n_iterations, initial_temperature, annealing_rate=0.99):
    '''Optimize the loss function using the Metropolis-Hastings algorithm.'''
    current_params = initial_params
    current_loss = loss_function(current_params)
    temperature = initial_temperature
    
    for i in range(n_iterations):
        current_params, current_loss = metropolis_step(current_params, current_loss, loss_function, temperature)
        # simulated annealing; automatically cool
        temperature *= annealing_rate
        
        # if i % 100 == 0:  # Print progress every 100 iterations
        print(f'Iteration {i}, Loss: {current_loss}, Temperature: {temperature}')
    
    return current_params, current_loss

def simplify_H(N_m, left=True, method=True, lambda_fix=None, num_rand=100000, gd_rand_try=1000, gd_N=1000, gd_lr=0.0001, gd_tol=1e-5, temp_init = 0.01, parallelize=True, binned=False):
    '''simplifies the Hamiltonian by removing terms that are not relevant for the mutual information calculation
    
    Params:
        N_m (int): number of Majorana fermions
        left (bool): whether to simplify the left or right Hamiltonian
        method (int): whether to use random search alg (0), gradient descent (1) or MCMC (2)
        lambda_fix (float): if not None, will fix the lambda_ value to this value, so leave as None if you want to optimize for lambda_ as well
        num_rand (int): number of random initializations to try
        gd_rand_try (int): number of random initializations to try for GD
        gd_N (int): number of iterations for GD
        gd_lr (float): learning rate for GD
        gd_tol (float): tolerance for GD
        parallelize (bool): whether to parallelize the random initializations
        binned (bool): whether to use binned or unbinned coefficients

    '''
    # get H_L and H_R
    H = get_SYK(N_m, left=left)

    print('got Hamiltonians')

    # find the simplest representation that most closely approximates the original Hamiltonian #
    # perform ridge regression to find the simplest representation
    # initialize the parameters

    # partialize the loss function
    loss_h_ = partial(loss_h, H_targ_mat=H.to_matrix(), left=True, lambda_fix=lambda_fix)

    random_h_coeff_N_m = partial(random_h_coeff, N_m, lambda_fix=lambda_fix, binned=binned)

    # t0 = time.time()
    # x0 = random_h_coeff_N_m()
    # print(f'Initial loss: {loss_h_(x0)}')
    # print(f'Elapsed time: {time.time() - t0}')
    
    # run the optimizer
    # x_best, loss_best = trabbit(loss_h_, random_h_coeff_N_m, alpha=0.0001, temperature=0.01, verbose=True)
    # run custom GD
    # first pick best random initialization
    if method==1:
        loss_best = np.inf
        for i in range(gd_rand_try):
            x0 = random_h_coeff_N_m()
            loss = loss_h_(x0)
            if loss < loss_best:
                loss_best = loss
                x_best = x0
        print(f'Best rand loss for H_{left}: {loss_best}')
        for i in range(gd_N):
            grad = approx_fprime(x_best, loss_h_, 1e-6)
            x_best -= gd_lr * grad
            loss_best = loss_h_(x_best)
            print(f'Loss at iteration {i}: {loss_best}')
            if loss_best < gd_tol:
                break    
    elif method == 0:
        loss_best = np.inf
        if not parallelize:
            for i in trange(num_rand):
                # every multiple of 1000, print out the loss
                if i % 1000 == 0:
                    print(f'Loss at iteration {i}: {loss_best}')
                x0 = random_h_coeff_N_m()
                loss = loss_h_(x0)
                if loss < loss_best:
                    loss_best = loss
                    x_best = x0

        # run in parallel
        else:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(loss_h_, random_h_coeff_N_m) for _ in trange(num_rand)]
                
                # Process the results as they complete
                for future in as_completed(futures):
                    loss = future.result()
                    if loss < loss_best:
                        loss_best = loss
                        x_best = x0

    elif method == 2:
        x_best, loss_best = mcmc_optimize(loss_h_, random_h_coeff_N_m(), gd_N, temp_init)
    else:
        raise ValueError('Invalid method')

    print(f'Best loss for H_{left}: {loss_best}. Number of non-zero terms: {np.count_nonzero(x_best[:-1])}')
    if lambda_fix is None:
        print(f'Best lambda: {x_best[-1]}')
    else:
        print(f'Fixed lambda: {lambda_fix}')

    # save the results
    timestamp = int(time.time())
    print(f'timestamp: {timestamp}')
    np.save(f'results_new/params_{left}_{timestamp}_{lambda_fix}_{method}.npy', x_best)
    np.save(f'results_new/loss_{left}_{timestamp}_{lambda_fix}_{method}.npy', loss_best)
    np.save(f'results_new/H_{left}_{timestamp}_{lambda_fix}_{method}.npy', H.to_matrix())

    return x_best, loss_best

def test_simplify_H(H_actual_path, params_reconstr_path):
    '''tests the simplified Hamiltonian by comparing the mutual information for the reconstructed circuit with the actual circuit'''
    # load the actual Hamiltonian
    H_actual = np.load(os.path.join('results_new', H_actual_path))
    # load the reconstructed parameters
    params_reconstr_wridge = np.load(os.path.join('results_new', params_reconstr_path))
    params_reconstr = params_reconstr_wridge[:-1]
    # get the reconstructed Hamiltonian
    H_reconstr = get_SYK_from_params(N_m, left=True, params=params_reconstr)
    H_reconstr = H_reconstr.to_matrix()
    # get the mutual information for the actual and reconstructed Hamiltonians
    dist = np.linalg.norm(H_actual - H_reconstr)
    print(f'Distance between actual and reconstructed Hamiltonians: {dist}')
    print(f'Loss for reconstructed Hamiltonian: {loss_h(params_reconstr_wridge, H_actual, left=True)}')
    return dist

## ---- test their learned hamiltonian ---- ##
def test_learned_hamiltonian(N_m=10, ans=0, mu=-12, subdir=None, display_circs=False, plot_result=True, t0=2.8, tf=10, t_steps=10):
    n_qubits = N_m // 2
    H_L = PauliSumOp.from_list([("I" * n_qubits, 0.0)]) 
    H_R = PauliSumOp.from_list([("I" * n_qubits, 0.0)])

    params = [-.36, .19, -0.71, .22, .49]
    qubits = [[0, 1, 3, 4], [0, 2, 3, 6], [0, 2, 4, 5], [1, 2, 3, 5], [1, 2, 4, 6]]

    for i, q in enumerate(qubits):
        H_L += params[i] * (majorana_to_qubit_op(q[0], n_qubits, left=True) @ majorana_to_qubit_op(q[1], n_qubits, left=True) @ majorana_to_qubit_op(q[2], n_qubits, left=True) @ majorana_to_qubit_op(q[3], n_qubits, left=True))

        H_R += params[i] * (majorana_to_qubit_op(q[0], n_qubits, left=False) @ majorana_to_qubit_op(q[1], n_qubits, left=False) @ majorana_to_qubit_op(q[2], n_qubits, left=False) @ majorana_to_qubit_op(q[3], n_qubits, left=False))

    # Initialize the Hamiltonian for the TFD state
    H_LR = PauliSumOp.from_list([("I" * N_m, 0.0)])

    identity_N = I ^ (N_m//2)

    # Embed H_L in the left N qubits and H_R in the right N qubits
    for term_L in H_L:
        # Tensor product of term_L with identity on the right
        H_LR += term_L.tensor(identity_N)

    for term_R in H_R:
        # Tensor product of identity on the left with term_R
        H_LR += identity_N.tensor(term_R)

    # --- prepare the V operator --- #
    V = get_V(N_m)
    # multiply by mu
    V = mu * V
    expV = time_evolve(V, tf=1)

    # --- run VQE to get the TFD state for given choice of ansatz --- #
    TFD = run_VQE(H_LR,  V, ans = ans, display_circs=display_circs, benchmark=False)

    # --- negative and positive time ev  --- #
    tev_nt0 = time_evolve(H_L, tf=-t0)
    tev_pt0 = time_evolve(H_L, tf=t0)

    if display_circs:
        tev_nt0.draw('mpl')
        plt.savefig('results_new/tev_nt0.pdf')
        tev_pt0.draw('mpl')
        plt.savefig('results_new/tev_pt0.pdf')

    # --- run the protocol --- #
    mutual_infos = []
    for t in np.linspace(0, tf, t_steps):
        mutual_infos.append(protocol_round(H_R, TFD, expV, tev_nt0, tev_pt0, t, display_circs=display_circs))

    # save the mutual infos
    if subdir is  None:
        if not os.path.exists('results_new'):
            os.makedirs('results_new')
        save_dir = 'results_new'
    else:
        if not os.path.exists(f'results_new/{subdir}'):
            os.makedirs(f'results_new/{subdir}')
        save_dir = f'results_new/{subdir}'

    mutual_infos = np.array(mutual_infos)
    timestamp = int(time.time())
    np.save(os.path.join(save_dir, f'mutual_infos_{N_m}_{ans}_{mu}_{timestamp}.npy'), mutual_infos)

    # make the plot
    if plot_result:
        plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(0, tf, t_steps), mutual_infos)
        plt.xlabel('Time')
        plt.ylabel('Mutual Information')
        plt.title(f'Mutual Information for $N_m = {N_m}$, ans = {ans}, $\mu = {mu}$')
        plt.savefig(os.path.join(save_dir, f'mutual_info_{N_m}_{ans}_{mu}_{timestamp}.pdf'))
        
    return mutual_infos

## ---- making plots for thesis ---- ##
def plot_reconstruct(sim_path1=['results_new/I_3_1708557041_True.npy', 'results_new/I_sem_3_1708557041_True.npy'], sim_path2 = ['results_new/I_3_1708411432.npy', 'results_new/I_sem_3_1708411432.npy'], data_path=['results_new/I_3_1708513045_False.npy', 'results_new/I_sem_3_1708513045_False.npy'], comparison_path='mi_data/mi_2.csv'):

    # load the data
    I1 = np.load(sim_path1[0])
    I_sem1 = np.load(sim_path1[1])
    I2 = np.load(sim_path2[0])
    I_sem2 = np.load(sim_path2[1])
    I3 = np.load(data_path[0])
    I_sem3 = np.load(data_path[1])
    t_I_ls = np.loadtxt(comparison_path, delimiter=',')
    t_ls = t_I_ls[:, 0]
    I_actual_ls = t_I_ls[:, 1]

    # plot
    plt.figure(figsize=(10, 10))
    plt.scatter(t_ls, I_actual_ls, label='Actual', color='red')
    plt.plot(t_ls, I_actual_ls, color='red')
    plt.errorbar(t_ls, I1, yerr=I_sem1, fmt='o', color='orange', label='Simulator, 2000 shots')
    plt.plot(t_ls, I1, color='orange')
    plt.errorbar(t_ls, I2, yerr=I_sem2, fmt='o', color='gold', label='Simulator, 10000 shots')
    plt.plot(t_ls, I2, color='gold')
    plt.errorbar(t_ls, I3, yerr=I_sem3, fmt='o', color='blue', label='Kyoto Processor, 2000 shots')
    plt.plot(t_ls, I3, color='blue')
    plt.xlabel('Time')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.title('Mutual Information for Reconstructed Circuit')
    plt.savefig('results_new/I_reconstruct_total.pdf')
    plt.show()

def plot_MI_benchmark(subdir='1708211517.5868132', standardize=False):

    subdir0 = subdir
    subdir = os.path.join('results_new',subdir)

    # get all the .npy
    files = os.listdir(subdir)
    I_files = [f for f in files if '.npy' in f]

    # split file names by '_': [-2] gives the mu value, [-3] gives the ansatz
    I_dict = {}
    ans_unique = []
    mu_unique = []
    for f in I_files:
        I = np.load(os.path.join(subdir, f))
        mu = f.split('_')[-2]
        ans = f.split('_')[-3]
        ans_unique.append(ans)
        mu_unique.append(mu)
        # append to dictionary
        # check if key exists. if not, create it with I as the value, otherwise append to the list
        if (ans, mu) not in I_dict:
            I_dict[(ans, mu)] = [I]
        else:
            # add each element of I to the list
            I_dict[(ans, mu)].append(I)

    # get unique values
    ans_unique = list(set(ans_unique))
    # sort ans small to large
    ans_unique = sorted(ans_unique, key=lambda x: int(x))
    mu_unique = list(set(mu_unique))
    # sort mu small to large
    mu_unique = sorted(mu_unique, key=lambda x: int(x))
    num_ans = len(ans_unique)
    num_mu = len(mu_unique)


    # plot a 4x4 grid of ansatz vs mu
    fig, axs = plt.subplots(num_ans, num_mu, figsize=(25, 25))
    for i, ans in enumerate(ans_unique):
        for j, mu in enumerate(mu_unique):
            I_ls = I_dict[(ans, mu)]
            for I in I_ls:
                x = np.linspace(0, 10, len(I))
                
                if standardize:
                    # Standardize I
                    I_mean = np.mean(I)
                    I_std = np.std(I)
                    I_standardized = (I - I_mean) / I_std
                    # Plot standardized I
                    axs[i, j].plot(x, I_standardized, marker='o')
                else:
                    # Plot original I
                    axs[i, j].plot(x, I, marker='o')
            axs[i, j].set_title(f'Ansatz {ans}, $\mu = {mu}$', fontsize=15)
            axs[i, j].set_xlabel('Time', fontsize=12)
            axs[i, j].set_ylabel('Mutual Information', fontsize=12)

    plt.tight_layout()
    # plt.suptitle('Mutual Information for VQE Benchmark')
    plt.savefig(f'results_new/mi_benchmark_{subdir0}_{standardize}.pdf')

def plot_H(N_m):
    '''Plots H_LR'''
    # make a plot of an H matrix
    H_LR, H_L, H_R = get_H_LR(N_m)
    H = H_LR.to_matrix()
    # get both the real and imaginary parts
    H_real = np.abs(H)
    H_imag = np.angle(H)
    H_imag = np.where(H_real > 1e-10, H_imag, 0)
    # plot the real part with color bars
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    a0 = ax[0].imshow(H_real, cmap='viridis')
    fig.colorbar(a0 , ax=ax[0])
    ax[0].set_title('Magnitude')
    ax1 = ax[1].imshow(H_imag, cmap='viridis')
    ax[1].set_title('Angle')
    fig.colorbar(a0, ax=ax[1])
    plt.tight_layout()
    # timestamp
    timestamp = int(time.time())
    plt.savefig(f'results_new/H_matrix_{N_m}_{timestamp}.pdf')        

if __name__ == '__main__':
    N_m=10
    # benchmark_vqe(N_m, num_iter=100, display_circs=False)
    # benchmark_mi(N_m, num_reps=100)
    # reconstruct_total()
    # run_reconstruction(simulate=True, resilience_level=3)
    # plot_angles()
    # for _ in trange(5):
    #     full_protocol(N_m, ans=4, mu = 0, display_circs=False)
    # repeat_full_protocol(N_m, num_reps=2)
    # plot_reconstruct()
    # plot_MI_benchmark(subdir='1709029808.6135924', standardize=False)
    # plot_MI_benchmark(subdir='1709029808.6135924', standardize=True)
    # plot_H(N_m)

    test_learned_hamiltonian()

    # simplify_H(N_m, left=True, method=2, num_rand=1000000, parallelize=False, binned=True, temp_init=0.1, lambda_fix = 0.7)
    # test_simplify_H('H_True_1709631423_None.npy', 'params_True_1709631423_None.npy')