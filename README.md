# Math-Thesis
Oscar Scholin, Pomona College 2024

Repo for all the code used in my math thesis at Pomona College with Professor Ami Radunskaya.

For full documentation of the science updates and particulars of the code, see ```README-sci.md```. 

## At a glance

* The most relevant file is ```syk_qk_redo.py```. It contains all the main code used in my thesis, which aims to implement the process of wormhole teleportation from Jafferis et al. 2022 (https://www.nature.com/articles/s41586-022-05424-3). It is written in Python and uses Qiskit version 0.45.2 (there is a known incompatability with Qiskit 1.0, and I am planning on resolving this issue soon). There is a known conflict in the results of the simulation with what is expected in Jafferis et al. Read chapter 3 of my thesis if you are curious to learn more.
    - The function ```full_protocol``` implements the simulation from $t=0$ to $t_f$ for a given number of time steps and mu val, and $t_0$.
    - By default ```full_protocol``` uses ```compute_mi_actual```, which takes in an argument for the backend, which can enable either the simulation environment or a real quantum computer.
        - ```get_SYK``` calculates the SYK Hamiltonians in qubit basis, converted from the fermionic by ```majorana_to_qubit_op``` given some list of parameters for each term
        - ```get_random_SYK_params``` generates these random parameters according to the normal distribution specified in the paper
    - there are several dedicated functions for benchmarking the performance of the VQE (```benchmark_vqe```), and ```benchmark_mi``` to run multiple simulations in parallel
    - the functions ```get_ansatz```, ```learn_point```, ```reconstruct_total```, and ```run_reconstruction``` are for specifically the gedankenexperiment I explored in which we can use a simplified circuit, specified in ```get_ansatz```, and learn the gate parameters in ```learn_point``` in order to match a desired mutual information timeseries. In ```run_reconstruction``` we use the learned angles to create states which we measure using the desired backend
    - ```loss_h```, ```random_h_coeff```, ```metropolis_step```, ```mcmc_optimize```, ```simplify_H```, ```test_simplify_H``` are for simplifying the hamiltonians. ```simplify_H``` has code for SGD as well as MCMC---SGD is too slow given that the hamiltonians must be explicitly calculated in matrix form to calculate the loss function, which is the frobenius norm to the target hamiltonian with a ridge parameter. 
* I have also explored the more general problem of decomposing arbitrary unitaries in terms of quantum circuits; this code can be found primarily in ```elegans_adapt2.py```. It iteratively builds a layer of single qubit rotation and CNOT gates and compares the loss against the target unitary. The problem is this gets VERY slow VERY fast due to figuring out the optimal placement for the conditional logic gates. The ```sphynx``` files are for a neural network-based reconstruction of the unitaries based on their eigenvalues, but this had poor results.