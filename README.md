# Math-Thesis
Repo for all the code used in my math thesis at Pomona College ('24) with Prof. Ami Radunskaya.

## Major updates

## 2/16/24
* found SuzukiTrotter method from Qiskit. puts in CNOT -- which makes sense why the final MI could change under these circumstances.
** biggest concern** what choice of VQE ansatz? what configuration of time evolution? how to sparsify since the CNOTs ruin the party?
** what i am going to do ** run benchmark for variety of configurations and see the impact on mutual info
* things to consider: 
** ansatz for TFD: need rigorous comparison ground state energy of ansatz to target
** what time evolution to use? [using PauliEvolutionGate, default synthesis is the Lie-Trotter product formula with a single repetition.] need more efficient: counting 333 cx, 171 u3 gates with 
** idea: once we have an understanding of what the mutual info should look like in the full mega gate setup, create a test circuit similar to the original one i was playing around with and try to match the mutual info dynamics **
--> motivation is there seems to be a lot of redudancy with the CNOT gates because of the way time evolution is implemented. would test if it also can learn the other dynamics we want to measure to prove teleportation

code: 
* wrote benchmark_vqe
* added ans param to full_protocol(). 
* for conveinience added get_H_LR() to prepare H_L, H_R, H_LR in the combined N_m qubit space.

## 2/15/24
* fixed majorana_to_qubit() 
* working on VQE: number of CZ = 2*N_m - 1. but how to perform VQE since the number of qubits required for H is only N_m/2. but: what about since we know H_TFD = H_L + H_R + V really means (H_L \otimes I_{N_q}) + (I_{N_q} \otimes H_R) + V. same implicit tensor product for V on L and R: i.e., V = 1/qN \sum_j (\psi_{j_L} \otimes I_{N_q/2}) (I_{N_q/2}  \otimes \psi_{j_R}).
** got VQE working! seems to have min eigenvalue of around -1.2. resulting parameters are the same even after re-running
* added trotter-suzuki and rest of protocol from previous code
* result is still 0 mutual info....
* changed ansatz to entangle neighbors. have non-0 MI, but doesn't as the later time evolution changes -- why would it? need to put the rotations *before* the conditional -- how is their 
* tried test idea of all CNOT chained together


## 2/13/24
* need to use a different jordan wigner transformation for the majorana, as well as vqe for SYK: https://learning.quantum.ibm.com/tutorial/variational-quantum-eigensolver.
* started file syk_qk_redo.py

## 2/12/24
* still nothing on the swap gates. tried changing them to cx but no luck.
* need to take H = Hl + Hr: doesn't fix anything....
* wrote AZ for code. need to understand what the circuit should actually look like and why the teleportation doesn't work
** maybe there is a tensor product in the V term between L and R? no, wouldn't make sense based on definitions.
* actually calculating explicit swap gates based on dirac operators -- their definition of swap doesn't make sense at all dimensionally...
* fixed definition of trotter suzuki so we multiply the angle by the step number

## 2/9/24
* trying to figure out the issue with TFD. I believe issue was how I was checking unitarity before. using Operator(circ).is_unitary() (added the new function, is_unitary(circ)). confirmed that TFD with imaginary time evolution is unitary.
* need to fix swap function
** was thinking to only swap the second qubit in bell pair to the first in tfd
** BUT, watched video on the teleportation, and they define the SWAP operation in terms of dirac fermions: https://www.youtube.com/watch?v=ZenQViOgxDM
** note from Jafferis and Gao: "Note that this is the ordinary SWAP operator"
** correctly using Aer engine to simulate the measurement of the quantum state resulting from the circuit
** issue is that while rho_L and rho_R are PSD, rho_LR is not PSD
** confirmed the density matrices are valid and calculated mutual info using qiskit.quantum_info.entropy(). got a sensible value, so now testing full protocol.
* got same value of I_pt the entire time. realized need to make P, Q, T as separate registers from the rest of the N + N which represent the two entangled systems for the tfd. 
** implemented this, but now getting mutual info = 0.
** realized need a minus sign on mu so that e^i mu V is implemented w trotter suzuki and not e^-imu V. doesn't solve issue of 0 mut info tho.
** fixed SWAP to only apply to qubits 1, 2 and total_num-2, total_num-1. still mutual info = 0.
** tried changing J but still no info
** tried inc trotter steps, no change
** realized from circuit diagrams the swap happens BEFORE ALL cnot gates
** WHERE TO PUT THE SWAP GATES??

## 2/7/24
* testing implement_protocol() with variety of changes:
** running 10, 200 revealed a kind of noisy cloud, so:
** added option to fix t0 param to 2.8 like in wormhole paper
** updated time_evolve to take time_step argument in place of steps so the time step is fixed, so at later time values we are still precise
** i think i was using the wrong register for the Ipt: should only be 0 and 2*N-1? no, what
** selective swapping of only the bell term part
* realized rhoL, rhoR, rhoLR not PSD. circuit not unitary. tried removing the swap circuits and still not unitary. in particular TFD is *not* unitary. both the expH and ent are not unitary. 
** issue was i was using DensityMatrix() on the circuit, but need to use Operator().data. need to revisit and fix.
** no duh, the expH won't be unitary bc no factor of 1j. 
 

## 2/5-6/24
* finished rest of wormhole protocol using strictly QC representation (up until computing I)
** added get_TFD(), get_bell_pair(), get_expV(), and implement_protocol() which puts it all together
* added zlokapa_code.py which has snippets from Zlokapa's thesis (ended up not using)

## 2/4/24
* reorganizing syk_qk.py so that get_SYK returns the H pauli_sum_op object.
* added get_TFD(), which depends on get_bell_pair(), time_reverse(), time_reverse_op().
* realized need to add both the L and R definitions for the jordan-wigner because in the potential term otherwise they all become I (added param, ```left```).
* changing from Zlokapa's definition so all values are in terms of Qiskit QC representation.
** for expV, we use 20 trotter steps and imaginary time (so the exponent is real)
** left the matrix versions of these in comments for future refer.
* fixed trotter_suzuki_circuit to accept imaginary time steps


## 1/30/24
* built syk_qk.py, which has the function get_SYK() to generate SYK hamiltonian for any given number of majorana fermions. automatically runs trotterization. was able to convert time evolution circuit with 20 trotter steps from 0 to 1 from about 580000 gates to 10, for an avg speedup of 58036 += 0 and avg fidelity to original of 6.087174651948914e-12 ± 1.2044172115152807e-12, for 10 hamiltonians. (built function to benchmark_SYK to automatically run these computations).

## 1/18/24
* realized making a CCNOT gate is impossible w EA2.1 bc it only allows CNOT between neighbors. making a convolutional transformer to decompose circuit structure for N<=10 qubits.

* sphynx_gen.py creates the random matrices and decoded structure to be used in training; sphynx.py will have the actual code for the transformer.

## 1/17/24
* Was able to run SWAP, CSWAP. not able to solve CCNOT or CCCNOT. why? let's debug this.

* hide print statements

* log performance over time, as well as final loss as a function of number of gates used to construct the same circuit, ran several times --> gets at somewhat of the solution space

* try out hilbert schmidt norm?

* get plan for measurement

To do:
* correct CCNOT

* Read more doc of Qiskit transpile

* Implement computation of measurements for chaos so we can directly compare w theory. Need to figure out pricing. --> measure correlations between all qubits? what would be an effective measurement that shows we've learned the circuits?

* get benchmark for EA2.1 and qiskit on **(3,5), (3,15), (6,5), (6,15)**, (8, 5), (8, 15), (10, 5), (10, 15). log avg time to complete, avg acc, sem acc. 

* do qiskit simulation, real thing!!

* Once we have figured out the above, go back to SYK model in cirq -- + test w model 2.1!!


## 1/16/23
* added 3 levels of ```debug``` to find_params
* tested on model 2.1: Hadamard, X, Y, Z, CZ, SWAP, CCNOT (Toffoli), CSWAP (Fredkin), CCCNOT (Toffoli w 2 control). 
** problem with final RP prune not preserving the actual RP sequence: incorrectly pasing the sample circuit using the test_seq to the run. fixed this.

 * also separate issue with CZ: realized that the current framework of EA2.1 does not allow simultaneous operations. fixed this by changing the framework to account for operations to be applied in simultaneous moments. 

## 1/15/24
Realized Qiskit has a module, transpile, which can convert an arbitrary unitary into decomposed circuit. BUT: much less efficient in terms of number of gates used compared to my algorithm: using a random_circuit(3,5) EA2.1 used ~6 gates whereas Qiskit transpile used ~60-100. Need rigorous comparison


### 1/13/24
Troubleshooting model 0; it doesn't seem possible to simplify as you go, so will create model 2.1 which essentially does model 2 but then finds all the stretches of RxRyRzP and simplifies after the fact. Implemented this. Increased num times to run trabbit from 1 to 5.

Benchmarking model 2 first: tested on depth of 5 up to 6 qubits, up to depth of 15 on 3 qubits. Handles great on my laptop and will run rigrous test with model 2.1 simplification when I get back from Germany. 

Realized major impracticality: adds more RP blocks even if no CNOTs: only add if 1) there are no gates yet or 2) there is a CNOT currently as the last item in the qubit list. Fixed this, and this increases speed by quite a bit.

Reminder: all in elegans_adapt2.py

### 1/9/24
Fixed issue with sample circuits 4-9. Model 2 can now effectively handle [some rotation] CNOT [some rotation] and CNOT [rotation] in any order on the qubits. Will implement simplification on each individual RP section tomorrow.
 

### 1/6/24
Wrote new Circuit class in circuit.py to handle storage, updating of genes and conversion to unitary. Implemented in elegans_adapt2.py for the iterative learning circuit. Started test_circuits.py which is a framework for simulating the circuits and actually experimentally testing them using Qiskit.

### 1/3/24
Developed "learning circuit" in elegans_new.py using the Pauli spin, P, CNOT. Also built a genetic algorithm in ga.py. Adapting elegans_new.py to cirq in elegans_cirq.py to better study the circuits with CNOTs--since if we do not allow any CNOTs, a circuit of only 4N for N qubits can learn a circuit of arbitrary depth.


### 11/6/23
As of the LO, the code I am using is ```syk.py``` to generate the Majorana operators and SYK Hamiltonians. ```interact_jg.py``` adapts Jafferis and Gao et al to implement the wormhole teleportation protocol. 
