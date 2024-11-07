import numpy as np
import sys
from scipy.linalg import expm
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from IPython.display import display


def create_single_hopping_fermionic_operator(i, j, N):
    if i > j:
        raise ValueError("i must be smaller or equal to j")
    else:
        L = 2**N
        H = np.zeros((L,L))
        if i!=j:
            for k in range(L):
                state = list(np.binary_repr(k, N))
                if state[i] == "0" and state[j] == "1":
                    counter = 1
                    for l in range(i, j):
                        cur_str = state[l]
                        if cur_str == "1":
                            counter *= -1
                    new_state = state.copy()
                    new_state[i] = "1"
                    new_state[j] = "0"
                    new_state = "".join(new_state)
                    new_state = int(new_state, 2)
                    H[new_state, k] = counter
        else:
            for k in range(L):
                state = list(np.binary_repr(k, N))
                if state[i] == "1":
                    H[k, k] = 1
    H = H - H.transpose()
    return H

def create_single_hopping_fermionic_operator_test(i, j, N):
    if i > j:
        raise ValueError("i must be smaller or equal to j")
    else:
        L = 2**N
        H = np.zeros((L,L))
        if i!=j:
            for k in range(L):
                state = list(np.binary_repr(k, N))
                "(a^+_i a_j)"
                if state[i] == "0" and state[j] == "1":
                    counter = 1
                    for l in range(i, j):
                        cur_str = state[l]
                        if cur_str == "1":
                            counter *= -1
                    new_state = state.copy()
                    new_state[i] = "1"
                    new_state[j] = "0"
                    new_state = "".join(new_state)
                    new_state = int(new_state, 2)
                    H[new_state, k] = counter
                "-(a^+_j a_i)"
                if state[j] == "0" and state[i] == "1":
                    counter = -1
                    for l in range(i+1, j):
                        cur_str = state[l]
                        if cur_str == "1":
                            counter *= -1
                    
                    new_state = state.copy()
                    new_state[i] = "0"
                    new_state[j] = "1"
                    new_state = "".join(new_state)
                    # print(state, "old")
                    # print(new_state, "new")
                    # print(counter)
                    # print("=====")
                    new_state = int(new_state, 2)
                    H[new_state, k] = counter
        else:
            for k in range(L):
                state = list(np.binary_repr(k, N))
                if state[i] == "1":
                    H[k, k] = 1
    return H

def create_single_hopping_qubit_operator(i,j,N):
    if i>j:
        raise ValueError("i must be smaller or equal to j")
    L = 2**N
    H = np.zeros((L,L))
    for k in range(L):
        state = list(np.binary_repr(k, N))
        if state[i] == "0" and state[j] == "1":
            counter = 1
            for l in range(i, j):
                cur_str = state[l]
                if cur_str == "1":
                    counter *= -1
            new_state = state.copy()
            new_state[i] = "1"
            new_state[j] = "0"
            new_state = "".join(new_state)
            new_state = int(new_state, 2)
            H[new_state, k] = counter
    H = H - H.transpose()
    return H

def create_single_hopping_qubit_exponent(N, i, j, t):
    y = np.array([[0, -1j], [1j, 0]])
    x = np.array([[0, 1], [1, 0]])
    z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    A = 1
    B = 1
    for k in range(N):
        if k < i:
            A = np.kron(A, I)
            B = np.kron(B,I)
        elif k == i:
            A = np.kron(A, -t*1j*y/2)
            B = np.kron(B, t*1j*x/2)
        elif k < j:
            A = np.kron(A, z)
            B = np.kron(B, z)
        elif k == j:
            A = np.kron(A, x)
            B = np.kron(B, y)
        else:
            A = np.kron(A, I)
            B = np.kron(B,I)
    A_test = np.kron(-1j*y/2,x)
    return np.matmul(expm(A), expm(B))


def construct_circuit(i,j,N,t):
    # Construc before hand gates
    qc = QuantumCircuit(N)
    qc.h(N-j-1)
    
    qc.sdg(N - i - 1)
    qc.h(N - i - 1)

    for k in range(i,j):
        qc.cx(N - k - 1, N - (k+1) - 1)
    qc.rz(t, N-j-1)
    for k in range(j,i,-1):
        qc.cx(N-k, N-k-1)

    qc.h(N-i-1)    
    qc.s(N-i-1)
    qc.h(N-j-1)
    
    # Construc the second term
    qc.h(N-i-1)
    
    qc.sdg(N-j-1)
    qc.h(N-j-1)

    for k in range(i,j):
        qc.cx(N-k-1, N-k-2)
    qc.rz(-t, N-j-1)
    for k in range(j,i,-1):
        qc.cx(N-k, N-k-1)

    qc.h(N-j-1)    
    qc.s(N-j-1)
    qc.h(N-i-1)
    return qc