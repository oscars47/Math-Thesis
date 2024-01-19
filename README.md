# Math-Thesis
Repo for all the code used in my math thesis at Pomona College ('24) with Prof. Ami Radunskaya.

## Major updates

BIG PICTURE GOAL FOR END OF WEEK, 1/19/24 
* research: have rigorous benchmarks on performance of EA2.1 and initial simulation and experiental results.
* writing: rough draft intro chapter

## 1/18/23
* realized making a CCNOT gate is impossible w EA2.1 bc it only allows CNOT between neighbors. making a convolutional transformer to decompose circuit structure for N<=10 qubits.

* sphynx_gen.py creates the random matrices and decoded structure to be used in training; sphynx.py will have the actual code for the transformer.

## 1/17/23
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

## 1/15/23
Realized Qiskit has a module, transpile, which can convert an arbitrary unitary into decomposed circuit. BUT: much less efficient in terms of number of gates used compared to my algorithm: using a random_circuit(3,5) EA2.1 used ~6 gates whereas Qiskit transpile used ~60-100. Need rigorous comparison


### 1/13/23
Troubleshooting model 0; it doesn't seem possible to simplify as you go, so will create model 2.1 which essentially does model 2 but then finds all the stretches of RxRyRzP and simplifies after the fact. Implemented this. Increased num times to run trabbit from 1 to 5.

Benchmarking model 2 first: tested on depth of 5 up to 6 qubits, up to depth of 15 on 3 qubits. Handles great on my laptop and will run rigrous test with model 2.1 simplification when I get back from Germany. 

Realized major impracticality: adds more RP blocks even if no CNOTs: only add if 1) there are no gates yet or 2) there is a CNOT currently as the last item in the qubit list. Fixed this, and this increases speed by quite a bit.

Reminder: all in elegans_adapt2.py

### 1/9/23
Fixed issue with sample circuits 4-9. Model 2 can now effectively handle [some rotation] CNOT [some rotation] and CNOT [rotation] in any order on the qubits. Will implement simplification on each individual RP section tomorrow.
 

### 1/6/23
Wrote new Circuit class in circuit.py to handle storage, updating of genes and conversion to unitary. Implemented in elegans_adapt2.py for the iterative learning circuit. Started test_circuits.py which is a framework for simulating the circuits and actually experimentally testing them using Qiskit.

### 1/3/23
Developed "learning circuit" in elegans_new.py using the Pauli spin, P, CNOT. Also built a genetic algorithm in ga.py. Adapting elegans_new.py to cirq in elegans_cirq.py to better study the circuits with CNOTs--since if we do not allow any CNOTs, a circuit of only 4N for N qubits can learn a circuit of arbitrary depth.


### 11/6/23
As of the LO, the code I am using is ```syk.py``` to generate the Majorana operators and SYK Hamiltonians. ```interact_jg.py``` adapts Jafferis and Gao et al to implement the wormhole teleportation protocol. 
