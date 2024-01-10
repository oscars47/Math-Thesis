# Math-Thesis
Repo for all the code used in my math thesis at Pomona College ('24) with Prof. Ami Radunskaya.

## Major updates

### 1/9/23
Fixed issue with sample circuits 4-9. Model 2 can now effectively handle [some rotation] CNOT [some rotation] and CNOT [rotation] in any order on the qubits. Will implement simplification on each individual RP section tomorrow.
 

### 1/6/23
Wrote new Circuit class in circuit.py to handle storage, updating of genes and conversion to unitary. Implemented in elegans_adapt2.py for the iterative learning circuit. Started test_circuits.py which is a framework for simulating the circuits and actually experimentally testing them using Qiskit.

### 1/3/23
Developed "learning circuit" in elegans_new.py using the Pauli spin, P, CNOT. Also built a genetic algorithm in ga.py. Adapting elegans_new.py to cirq in elegans_cirq.py to better study the circuits with CNOTs--since if we do not allow any CNOTs, a circuit of only 4N for N qubits can learn a circuit of arbitrary depth.


### 11/6/23
As of the LO, the code I am using is ```syk.py``` to generate the Majorana operators and SYK Hamiltonians. ```interact_jg.py``` adapts Jafferis and Gao et al to implement the wormhole teleportation protocol. 
