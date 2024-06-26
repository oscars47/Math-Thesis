# Math-Thesis
Repo for all the code used in my math thesis at Pomona College ('24) with Prof. Ami Radunskaya.

## Major updates

## 4/12/24
* also tried ```L_BFGS_B``` optimizer with multiple simulataneous evaluations, and still the result is inconsistent with the expected behavior.

## 4/7/24
* implementing their hamiltonian. result is nonsensical:not enough mutual information difference. tried doing the optimizer 1000 times instead of 100, and now basically 0 mutual information. 

## 4/2/24
* looking into error mitigation
    - https://qiskit-extensions.github.io/qiskit-experiments/manuals/measurement/readout_mitigation.html
    - not quite sure how to implement for state tomography
    - can try to do a parallel experiment tho

## 3/4/24
* added metropolis hastings alg for ```simplify_H```. seems to be working well so far; temp_init of ~ 0.4 seems best
    * did not work. even though with temp_init = 0.1 was able to achieve 
        Best loss for H_True: 8.219160996247865e-08. Number of non-zero terms: 10000
    when calculating difference directly, no ridge, got 
        Distance between actual and reconstructed Hamiltonians: 15.807895907647413
    I think this is because the lambda term was negative, so even though the loss function was always >= 0, a lot of stuff was allowed to cancel inside the abs val
    * will fix temp_init at 0.1, then try variety of lambda: 0.01, 0.1, 0.5, 0.7

## 3/2/24
* found note on error correction: https://docs.quantum.ibm.com/run/configure-error-mitigation. i want to re-run calcultions for MI on Kyoto in March (next IBM cycle) using resilience level 2 or 3.
* doesn't appear it's possible to set ```resilience_level``` for StateTomography

## 2/27/24
* realized the mi plots i made before were for N_m = 6, so remade them for N_m = 10, also made font on the titles slightly larger
* trying binned approach for simplify_H generating params -- actually it's quite slow....
* results of 1 million random param:
    Best loss for H_True: 0.8749393635473162. Number of non-zero terms: 210
    Best lambda: 0.030888040317091803

    Distance between actual and reconstructed Hamiltonians: 0.8103448462028633

## 2/26/24
* parallelization crashed again bc can't pickle partialized function...
* result from 10^6 random with setting 10% of terms to 0 is 
    Best loss for H_True: 1.293806055364993. Number of non-zero terms: 29
* realized there's a bit of redudnancy there bc the lambda term should help us remove out terms so this may just be not helpful; rerunning.
** for N_m = 10, about 9 iterations / s.


## 2/25/24
* realized using genetic algorithm by randomly generating params is MUCH faster than gd
* updated ```random_h_coeff``` to randomly set all but 10% of coefficients to 0
* running 3 experiments: 1. grad descent with lambda_fix = .01, 2. random with lambda_fix = 0.01, 3. random with no lambda_fix. hypothesis for gd is that optimizing for lambda as well was taking too long; updated ```loss_h``` accordingly
** lambda_fix = None and genetic had best performance in terms of speed
** added parallelization which doubled speed
* want to investigate quantum scarring --> analog of periodic orbits *
* if we have a valid simplified hamiltonian then can do the other test measures?

## 2/24/24
* did standardization on each MI output; now much clearer to see.
* adding ```simplify_H()``` code; idea is to perform ridge regression on the coefficients of the terms in the pauli sum
** to do this, added ```get_SYK_params()``` and ```get_random_SYK_params()``` as separate 
** each run is about 0.01 seconds

## 2/22/24
* got MI 100
* ran MI 1000
* ran vqe 100

## 2/21/24
* VQE/MI still running; MI crashed
* running 2000 shots x 5 for ibmq_qasm in order to compare better to hardware results

## 2/20/24
* BIG DAY!! added compute_mi_actual(), which uses  
* total time between submission to ibm_kyoto and execution is ~10 mins, but note that can only execute 3 instances at a time, so just running in series so it is easy to identify which job is which if the local device crashes since queue to execution time is reasonable.
* ran several instances of noisy simulation using IBMQ ibmq_qasm_simulator backend using 10000 shots repeated 5 times, took avg and sem and saved in results_new/I_3_1708411432.npy and analogous path with sem.
** result is good agreement with theory!!
* currently running hardware data 
* for future reference, to initialize Qiskit IBMQ, need to run a python file with the following code once:
    from qiskit_ibm_provider import IBMProvider
    IBMProvider.save_account(API_KEY, overwrite=True)
* wrote up chapter 3 of thesis to explain the simulation process
* realized that ref. 3 used 2*N_m-1 CZs -- so I will add ansatz 4. 
** problem is realized that when mu = 0 still have MI > 0... also shape of peak is still not consistent.  
* ran benchmark of VQE with 1000 circuits
* running benchmark of MI with 1000 circuits


## 2/19/24
* adding code to simulate with noise the reconstructed circuit and to run the result on eagle processor

## 2/18/24
* ran reconstruct_total() on the analytic large N MI data on 3 qubit circuit, saving both the loss and angles. result for 15 different points is:  
    Avg loss: 1.0282881953334784e-09
    SEM dev loss: 3.3446982547455856e-10
* will run reconstruct_total() but on actual IBM eagle processor so we can get those results using the saved angle configs

## 2/17/24
* ran benchmark of vqe with 20 iterations per ansatz
* generating mutual info over time for mu in [-12, -6, 0, 6, 12] and each ansatz 5 times (added parallelization) --> will combine these into one plot, which will be main result for this implementation of the protocol. actually decided not to combine all of them bc the scaling is so off...could for each config (ansatz, mu) will calculate avg, sem per datapoint and plot each of these. CLEARER ARGUMENT: should be no mutual info transmitted at mu = 0. only ans 0 does this. but then the resulting shapes for other mu for ans 0 don't agree (not consistent), and the ground state energy is off 0.1491039549729666 +- 0.09610106055094844, which is abs val diff relative to abs val of true
* required number of gates for full implementation: 
    [('cx', 333), ('u3', 171)]
* extracted MI data from jafferis et al 2022 paper
* added parametrized model to replicate the MI dynamics: will train on the extracted data using trabbit
* fundamental philosophical question: if i can recreate the MI data to arbitrary precision, what does this mean from a simulation perspective?


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
* wrote benchmark_vqe  --  + added parallelization of inner loop checking the different ansatzes
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
