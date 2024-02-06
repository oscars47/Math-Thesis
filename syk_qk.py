# use qiskit to simulate SYK hamiltoninans and wormhole protocol
import numpy as np
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Operator, DensityMatrix, partial_trace
from qiskit.opflow import I, X, Y, Z, PauliSumOp, PauliOp
import matplotlib.pyplot as plt
import os, time
from math import factorial
from scipy.linalg import expm

## ------ visualization ------- ##
def print_matrix(matrix, show=False, save=False, save_name=None):
    '''Nicely prints out matrix with color coded mag and phase.'''
    mag = np.abs(matrix)
    phase = np.angle(matrix)

    # wherever mag is 0, make phase 0
    phase = np.where(mag == 0, 0, phase)

    # plot both matrices with color bar; add colorbar
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im = ax[0].imshow(mag, cmap='RdBu_r')
    fig.colorbar(im, ax=ax[0])
    im2 = ax[1].imshow(phase, cmap='RdBu_r')
    fig.colorbar(im2, ax=ax[1])
    if save:
        # add to 'syk_qiskit' folder
        path = os.path.join(os.getcwd(), 'syk_qiskit')
        if not os.path.exists(path):
            os.makedirs(path)
        # timestamp
        if save_name is None:
            plt.savefig(os.path.join(path, f'{time.time()}.pdf'))
        else:
            plt.savefig(os.path.join(path, save_name+str(time.time())+'.pdf'))
    if show:
        plt.show()
    
## ------ qiskit helpers ------- ##
def num_gates_pso(pauli_sum_op):
    '''Count number of gates for PauliSumOp object.'''
    gate_count = 0
    # Extract the SparsePauliOp
    sparse_pauli_op = pauli_sum_op.primitive

    # Iterate over each Pauli string in SparsePauliOp
    for pauli_string in sparse_pauli_op.paulis:
        # Count non-identity operators in the Pauli string
        gate_count += np.sum(pauli_string != 'I')

    return gate_count

#### ------ WORMHOLE PROTOCOL ------- ####
def majorana_to_qubit_op(l,  num_qubits, left=True):
    '''Performs Jordan Wigner transformation from Majorana to qubit operators, as in Jafferis and Gao et al.

    Params:
        l (int): index of Majorana operator
        num_qubits (int): number of qubits
        left (bool): whether the Majorana operator is on the left or right side 
    
    '''
    j = (l + 1) // 2
    
    # start with identity operator for all qubits
    operator_chain = [I] * num_qubits
    
    # apply (ZX)^(j-1)
    for idx in range(j - 1):
        operator_chain[idx] = Z if idx % 2 == 0 else X
    
    if left:
        # apply the XX or YX at j-th position
        if l % 2 == 0:  # for 2j-1, we apply XX, which is just X because X^2 = I
            operator_chain[j - 1] = X
        else:  # for 2j, we apply YX, which simplifies to iZ
            operator_chain[j - 1] = Y  # Apply Y at the j-th position

         # if not the last qubit, apply X on the next qubit for the 2j-1 case
        if l % 2 == 0 and j < num_qubits:  # check if there's a qubit to apply X on
            operator_chain[j] = X

    else:
        # apply I j-th position
        operator_chain[j - 1] = I

        # if not the last qubit, apply Y if odd or Z if even on the next qubit for the 2j-1 case
        if l % 2 == 0 and j < num_qubits:
            operator_chain[j] = Z
        elif l % 2 == 1 and j < num_qubits:
            operator_chain[j] = Y

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
                        term = majorana_to_qubit_op(i, n_qubits, left=left) @ majorana_to_qubit_op(j, n_qubits, left=left) @ majorana_to_qubit_op(k, n_qubits, left=left) @ majorana_to_qubit_op(l, n_qubits, left=left)
                        H += J[i, j, k, l] * term

    return H

## ------ TROTTER-SUZUKI ------- ##
def trotter_suzuki_circuit(pauli_sum_op, time, steps):
    '''Perform the Trotter-Suzuki approximation of the time evolution operator.'''
    n_qubits = pauli_sum_op.num_qubits
    qc = QuantumCircuit(n_qubits)

    # First-order Trotter-Suzuki approximation
    delta_t = time / steps
    for _ in range(steps):
        for term in pauli_sum_op:
            # print('term:', term.primitive.coeffs[0] )
            # need to check if delta_t is real or imaginary
            if np.isreal(delta_t):
                angle = -1j * np.abs(term.primitive.coeffs[0]) * delta_t # sometimes Qiskit will autoconvert YX = iZ thus adding a phase, so we need to take the abs
            else:
                angle = np.abs(term.primitive.coeffs[0]) * delta_t
            angle = np.imag(angle)
            # print('angle:', angle)
            pauli_strings = term.primitive.paulis.to_labels()
            for qubit_idx, gate_coeff in enumerate(pauli_strings):
                    for i, gate in enumerate(gate_coeff):
                        # print('qubit_idx:', i)
                        # print('gate_char:', gate)
                        if gate == 'X':
                            qc.rx(2 * angle, i)
                        elif gate == 'Y':
                            qc.ry(2 * angle, i)
                        elif gate == 'Z':
                            qc.rz(2 * angle, i)
                        # 'I' does nothing
    return qc

def time_evolve(H, tf=1, steps=20, benchmark=False, save=False, save_name=None):
    '''Time evolves the Hamiltonian H from t0 to tf in steps.'''
    print('Time evolving...')
    # get the circuit 
    H_circ = trotter_suzuki_circuit(H, tf, steps)

    # log the total number of gates in the qc
    num_gates_initial_dict = H_circ.count_ops()
    num_gates_initial = sum(num_gates_initial_dict.values())
    print('Number of gates in initial circuit:', num_gates_initial)

    # run transpiler
    H_opt_circ = transpile(H_circ, optimization_level=1)
    print('Transpiled...')

    # log the number of gates in the optimized qc
    num_gates_opt_dict = H_opt_circ.count_ops()
    num_gates_opt = sum(num_gates_opt_dict.values())
    print('Number of gates in optimized circuit:', num_gates_opt)

    if benchmark:
        # compute fidelity
        # get matrix representation of the circuit
        H_circ_matrix = Operator(H_circ).data
        print('Converted to matrix')
        H_opt_circ_matrix = Operator(H_opt_circ).data

        # fidelity
        fidelity = np.linalg.norm(H_circ_matrix - H_opt_circ_matrix)

        # log gate speedup
        gate_speedup = num_gates_initial / num_gates_opt

        print('Fidelity:', fidelity)
        print('Gate speedup:', gate_speedup)

        # print out both matrices
        print_matrix(H_circ_matrix, save=True, save_name='H_circ')
        print_matrix(H_opt_circ_matrix, save=True, save_name='H_opt_circ')

        if save:
            # add to 'syk_qiskit' folder
            path = os.path.join(os.getcwd(), 'syk_qiskit')
            if not os.path.exists(path):
                os.makedirs(path)
            # timestamp
            if save_name is None:
                np.save(os.path.join(path, f'H_circ_{time.time()}'), H_circ_matrix)
                np.save(os.path.join(path, f'H_opt_circ_{time.time()}'), H_opt_circ_matrix)
            else:
                np.save(os.path.join(path, save_name+f'H_circ_{time.time()}'), H_circ_matrix)
                np.save(os.path.join(path, save_name+f'H_opt_circ_{time.time()}'), H_opt_circ_matrix)

        return fidelity, gate_speedup
    else:
        return H_opt_circ # just return the optimized circuit

def get_TFD(H, beta=4, steps=20):
    '''Takes in pauli sum op and returns the TFD state. Assumes time reversal applied first. Performs computation as quantum circuit. Applies Trotter-Suzuki with imag time evolve the Hamiltonian.

    Params:
        H (PauliSumOp): Hamiltonian
        beta (float): inverse temperature
        steps (int): number of steps for Trotter-Suzuki
        '''

    # exponentiate using trotter suzuki
    tfd_i = trotter_suzuki_circuit(H, -1j*beta, steps)

    H_exp = -1j*beta/4 * H

    # use trotter suzuki to get the circuit
    expH = time_evolve(H_exp, steps)

    # create a circuit with n_qubits on the left and n_qubits on the right, appended together
    ent = QuantumCircuit(2*H.num_qubits)

    # entangle the two halves
    for i in range(H.num_qubits):
        ent.h(i)
        ent.cx(i, i+H.num_qubits)

    # apply expH to both halves; need to shift the qubit indices by the desired amount
    expH2 = QuantumCircuit(2*H.num_qubits)
    for gate in expH.data:
        qubits = [q.index for q in gate[1]]
        expH2.append(gate[0], qubits)
        qubits = [q + H.num_qubits for q in qubits]
        expH2.append(gate[0], qubits)
    
    # now compose
    tfd = expH2.compose(ent)

    # transpile
    opt_tfd = transpile(tfd, optimization_level=1)

    return opt_tfd # is already normalized bc quantum circuit

def get_bell_pair(n_qubits):
    '''Returns the bell state in N qubit hilbert space using quantum circuit'''
    qc = QuantumCircuit(2*n_qubits) # account for L and R
    qc.h(0)
    qc.cx(0, n_qubits)
    return qc
    
## ------ Apply potential ------- ##
def get_expV(n_majorana, mu =-12, steps=20):
    '''Computes the exponential of the potential V for n_majorana fermions for interaction param mu'''

    n_qubits = n_majorana // 2

    # initialize the potential as pauli sum op
    V = PauliSumOp.from_list([("I" * n_qubits, 0.0)])
    
    for j in range(n_majorana):
        term = majorana_to_qubit_op(j, n_qubits, left=True) @ majorana_to_qubit_op(j, n_qubits, left=False)
        V += 1/(4*n_majorana) * term

    # scale by mu
    V = mu * V

    # compute exponential with trotter suzuki
    expV = time_evolve(V, tf=1, steps=steps)
    return expV

## ------ main function to implment the protocol ------- ##
def implement_protocol(n_majorana, tmin=0, tmax=10, steps = 20, mu=-12, beta=4):
    '''Computes the correlation K for n_majorana fermions at time t, interaction param mu, and inverse temperature beta'''

    # get the Hamiltonian
    H = get_SYK(n_majorana) # pauli sum op


    # initialize list of mutual infos
    I_ls = []

    delta_t = (tmax - tmin) / steps

    for t in np.arange(tmin, tmax, delta_t):
        ## STEP 1: generate TFD and apply negative time evolution on L
        # make tfd
        tfd = get_TFD(H, beta) # quantum circuit

        # apply backwards time evolution to L part of tfd
        tev_nt0 = time_evolve(H, tf=-t, steps=steps) # quantum circuit
        # apply this circuit to the tfd, but only left
        tev_nt0_big = QuantumCircuit(2*H.num_qubits)
        for gate in tev_nt0.data:
            qubits = [q.index for q in gate[1]]
            tev_nt0_big.append(gate[0], qubits)

        # now compose with tfd
        tfd = tfd.compose(tev_nt0_big)
        # print(tfd)

        ## STEP 2: swap in R part of bell pair into tfd
    # get bell pair P and Q
        # swap in R part of bell pair into tfd
        # # get the last half of the qubits in bell pair
        # R_part_bp = QuantumCircuit(2*H.num_qubits)
        # # start indexing from H.num_qubits
        # for i in range(H.num_qubits, 2*H.num_qubits):
        #     gate = bell_pair.data[i]
        #     qubits = [q.index for q in gate[1]]
        #     R_part_bp.append(gate[0], qubits)

        # # apply the swap
        # # get only the R part of the tfd
        # R_part_tfd = QuantumCircuit(H.num_qubits)
        # for i in range(H.num_qubits, 2*H.num_qubits):
        #     gate = tfd.data[i]
        #     qubits = [q.index for q in gate[1] + H.num_qubits] # start applying at the right
        #     R_part_tfd.append(gate[0], qubits)
        
        bell_pair = get_bell_pair(H.num_qubits) # quantum circuit

        # apply swap gate
        swap_c = QuantumCircuit(2*H.num_qubits)
        for i in range(H.num_qubits):
            swap_c.swap(i + H.num_qubits, i) # R swapped to L

        # print(swap_c)

        # now compose
        R_part_bp = bell_pair.compose(swap_c)

        # now compose with tfd
        tfd_swapped = R_part_bp.compose(tfd)
        # print(tfd_swapped)

        ## STEP 3: apply forward time evolution to L part of tfd
        # apply forward time evolution to L part of tfd
        tev_pt0 = time_evolve(H, tf=t, steps=steps) # quantum circuit
        # apply this circuit to the tfd, but only left
        tev_pt0_big = QuantumCircuit(2*H.num_qubits)
        for gate in tev_pt0.data:
            qubits = [q.index for q in gate[1]]
            tev_pt0_big.append(gate[0], qubits)

        # now compose with tfd
        tfd_pt0 = tfd_swapped.compose(tev_pt0_big)

        ## STEP 4: apply potential to both parts of tfd
        expV = get_expV(n_majorana, mu, steps) # quantum circuit
        # apply to both
        expV2 = QuantumCircuit(2*H.num_qubits)
        for gate in expV.data:
            qubits = [q.index for q in gate[1]]
            expV2.append(gate[0], qubits)
            qubits = [q + H.num_qubits for q in qubits]
            expV2.append(gate[0], qubits)

        # now compose
        tfd_expV = tfd_pt0.compose(expV2)

        ## STEP 5: apply forward time evolution by t1 to R part of tfd
        # apply forward time evolution to R part of tfd
        tev_pt1 = time_evolve(H, tf=t, steps=steps) # quantum circuit
        # apply this circuit to the tfd, but only right
        tev_pt1_big = QuantumCircuit(2*H.num_qubits)
        for gate in tev_pt1.data:
            qubits = [q.index + H.num_qubits for q in gate[1]]
            tev_pt1_big.append(gate[0], qubits)

        # now compose with tfd
        tfd_pt1 = tfd_expV.compose(tev_pt1_big)
        print(tfd_pt1)

        ## STEP 6: swap out R part of tfd
        # # get the first half of the qubits in bell pair
        # L_part_bp = QuantumCircuit(2*H.num_qubits)
        # for i in range(H.num_qubits):
        #     gate = bell_pair.data[i]
        #     qubits = [q.index for q in gate[1]]
        #     L_part_bp.append(gate[0], qubits)

        # # get R part of tfd
        # R_part_tfd_pt1 = QuantumCircuit(2*H.num_qubits)
        # for i in range(H.num_qubits, 2*H.num_qubits):
        #     gate = tfd_pt1.data[i]
        #     qubits = [q.index for q in gate[1] + H.num_qubits] # start applying at the right
        #     R_part_tfd_pt1.append(gate[0], qubits)

        # # compose
        # tfd_final = L_part_bp.compose(R_part_tfd_pt1)

        # apply swap gate
        swap_c = QuantumCircuit(2*H.num_qubits)
        for i in range(H.num_qubits):
            swap_c.swap(i, i +H.num_qubits) # L swapped to R

        # print(swap_c)
        
        # now compose
        tfd_swap_out = tfd_pt1.compose(swap_c)  # quantum circuit
        
        # optimize
        tfd_final = transpile(tfd_swap_out, optimization_level=1)
        print(tfd_final)
        print('number of gates:', tfd_final.count_ops())

        ## STEP 7: compute the mutual info
        # I = S(L) + S(R) - S(LR)
        # S(L) = S(R) = -Tr(ρ_L log ρ_L) = -Tr(ρ_R log ρ_R) = -Tr(ρ_LR log ρ_LR)

        # get the density matrix
        tfd_final_dens = DensityMatrix(tfd_final)
        # convert to DensityMatrix
        tfd_final_dens = tfd_final_dens.data


        # get the reduced density matrices
        rho_L = partial_trace(state = tfd_final_dens, qargs = range(H.num_qubits))
        rho_R = partial_trace(state = tfd_final_dens, qargs = range(H.num_qubits, 2*H.num_qubits))

        # get the joint density matrix
        rho_LR = tfd_final_dens

        # compute the mutual info
        I = -np.trace(rho_L @ np.log(rho_L)) - np.trace(rho_R @ np.log(rho_R)) + np.trace(rho_LR @ np.log(rho_LR))
        print('Mutual info:', I)
        I_ls.append(I)

    # plot the mutual info
    plt.plot(np.arange(tmin, tmax, delta_t), I_ls)
    plt.xlabel('Time')
    plt.ylabel('Mutual Info')
    plt.show()
    
    return I_ls

## ------- testing ------- ##
def benchmark_SYK(num, n_majorana):
    '''Simulates num times of n_majorana SYK Hamiltonians and logs the average and sem fidelity and gate speedup.'''
    fidelities = []
    gate_speedups = []
    for _ in range(num):
        fidelity, gate_speedup = time_evolve(get_SYK(n_majorana), benchmark=True)
        fidelities.append(fidelity)
        gate_speedups.append(gate_speedup)
    avg_fidelity = np.mean(fidelities)
    sem_fidelity = np.std(fidelities) / np.sqrt(num)
    avg_gate_speedup = np.mean(gate_speedups)
    sem_gate_speedup = np.std(gate_speedups) / np.sqrt(num)

    print(f'Average fidelity: {avg_fidelity} ± {sem_fidelity}')
    print(f'Average gate speedup: {avg_gate_speedup} ± {sem_gate_speedup}')

if __name__ == "__main__":
    implement_protocol(12, steps = 50)