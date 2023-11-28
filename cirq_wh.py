# file to create the SYK hamitlonian and run the simulation using Cirq and OpenFermion
import cirq, openfermion, scipy.sparse, math, datetime, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def SYK(N=10, J2=2, save=True):
    '''Generates the SYK Hamiltonian for N fermions and J2 true variance.
    
    Params:
        N (int): number of qubits
        J2 (float): true variance
        save (bool): save the argument of the Hamiltonian to a text file

    Returns:
        H (openfermion.QubitOperator): SYK Hamiltonian
    
    '''
    N *= 2 # convert to majorana fermions
    # initialize Hamiltonian
    H = openfermion.FermionOperator()
    for i in range(1,N+1):
        for j in range(i+1, N):
            for l in range(j+1, N-1):
                for k in range(l+1, N-2):
                    # get random coupling coefficient
                    J_ijkl = J2 * np.random.normal(loc=0.0, scale=math.factorial(3)*J2/(2**(N)))
                    # add product of 4 majorana fermion terms corresponding to the indices to Hamiltonian
                    H += J_ijkl * openfermion.majorana_operator((i,int(i%2==0) )) * openfermion.majorana_operator((j,int(j%2==0) )) * openfermion.majorana_operator((k,int(k%2==0) )) * openfermion.majorana_operator((l,int(l%2==0) ))
    # convert to qubit operator
    H = openfermion.transforms.jordan_wigner(H)
    # save the text file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if save:
        N //= 2 # convert back to fermions
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # create a separate directory for each N
        if not(os.path.isdir(f'ham_cirq/{N}')):
            os.mkdir(f'ham_cirq/{N}')
        with open(f'ham_cirq/{N}/{timestamp}.txt', 'w') as f:
            f.write(str(H))
    return H, timestamp

def load_SYK(N, timestamp):
    '''Loads the SYK Hamiltonian for N fermions and J2 true variance.
    
    Params:
        N (int): number of fermions
        timestamp (str): timestamp of the saved Hamiltonian

    Returns:
        H (openfermion.QubitOperator): SYK Hamiltonian
    
    '''
    # load the text file
    with open(f'ham_cirq/{N}/{timestamp}.txt', 'r') as f:
        H = f.read()
    # convert to qubit operator
    H = openfermion.QubitOperator(H)
    return H

def num_qubits(qubit_operator):
    '''Returns the number of qubits required to make the qubit operator.'''
    max_qubit_index = -1
    for term in qubit_operator.terms:
        for qubit, _ in term:
            max_qubit_index = max(max_qubit_index, qubit)
    return max_qubit_index + 1

def convert_matrix(timestamp, N, H = None, print_mat=True):
    '''Converts the Hamiltonian to a matrix. CAUTION: this can be very memory-intensive for large systems!! Don't use above N=6 (10 qubits) unless you want to fry some eggs.
    
    Params:
        timestamp (str): timestamp of the saved Hamiltonian
        N (int): number of fermions
        H (openfermion.QubitOperator): SYK Hamiltonian
        print_mat (bool): print the matrix representation of the Hamiltonian as matplotlib plot

    Returns:
        H_mat (np.array): matrix representation of the Hamiltonian
    
    '''
    if H is None:
        H = load_SYK(N, timestamp)

    # convert the qubit operator to a sparse matrix
    n_qubits = num_qubits(H)
    sparse_matrix = openfermion.get_sparse_operator(H, n_qubits=n_qubits)

    # convert the sparse matrix to a dense matrix for printinge
    dense_matrix = scipy.sparse.csr_matrix.todense(sparse_matrix)
    
    if H is None:
        if print_mat:
            title = f'Hamiltonian for {n_qubits} qubits, {timestamp}'
            plot_matrix(dense_matrix, title, N)
    return dense_matrix, n_qubits

def plot_matrix(dense_matrix, title, N):
    '''Plots the matrix representation of the Hamiltonian.'''
    # separate the mag and phase components of the matrix
    mag = np.abs(dense_matrix)
    phase = np.angle(dense_matrix)
    # set phase to 0 when mag is 0
    phase = np.where(np.isclose(mag,0, 1e-19), 0, phase)

    # plot both matrices as a heatmap, with legend for the heatmap
    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    im0 = ax[0].imshow(mag, cmap='viridis')
    im1 = ax[1].imshow(phase, cmap='viridis')

    # create an axes divider for the first subplot
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)

    # create an axes divider for the second subplot
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    # create colorbars
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar1 = fig.colorbar(im1, cax=cax1)

    # set colorbar labels
    cbar0.set_label('Magnitude')
    cbar1.set_label('Phase')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'ham_cirq/{N}/{title}.pdf')
    plt.show()

def plot_eigenvalues(timestamp, N):
    '''Plots the eigenvalues of the Hamiltonian.
    
    Params:
        H (openfermion.QubitOperator): SYK Hamiltonian
        print_mat (bool): print the matrix representation of the Hamiltonian as matplotlib plot

    Returns:
        H_mat (np.array): matrix representation of the Hamiltonian
    
    '''
    # convert the qubit operator to a sparse matrix
    dense_matrix, n_qubits = convert_matrix(timestamp, N, print_mat=False)

    # get eigenvalues
    eigvals = np.linalg.eigvalsh(dense_matrix)

    # plot eigenvalues
    plt.hist(eigvals, bins=100)
    plt.title(f'Eigenvalues for {n_qubits} qubits, {timestamp}')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'ham_cirq/{N}/{timestamp}_eigvals.pdf')
    plt.show()

def get_TFD(H, N, beta = 4, plot=False):
    '''Returns the thermal field double state for Hamiltonian H. Requires us to convert the Hamiltonian to a dense matrix, since Openfermion doesn't have a built-in function to get the TFD'''
    # get dense matrix
    H_mat, n_qubits = convert_matrix(None, None, H, print_mat=False)
    # get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H_mat)
    # get partition function
    Z = np.sum(np.exp(-beta*eigvals))
    # get density matrix
    rho = np.zeros(H_mat.shape, dtype=np.complex128)
    for i in range(len(eigvals)):
        rho += np.exp(-beta*eigvals[i])/Z * np.outer(eigvecs[:,i], np.conj(eigvecs[:,i]))

    if plot:
        plot_matrix(rho, f'TFD for {n_qubits} qubits, beta={beta}', N)
    
    return rho


if __name__ == '__main__':
    # H, timestamp = SYK(6)
    # plot_eigenvalues('20231127-160358', 6)
    H  = load_SYK(6, '20231127-160358')
    print(get_TFD(H, 6, plot=True))
    


    # H = load_SYK(10, '20210503-171524')

    # print(openfermion.transforms.jordan_wigner(openfermion.majorana_operator((0, 1))))
