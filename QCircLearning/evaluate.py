from typing import Mapping
import numpy as np

from qiskit.quantum_info import Operator, DensityMatrix, process_fidelity
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator

simulator = AerSimulator(method='density_matrix')

Direct_measure = True

def init_state(n:int):
    # Initialize state
    mixed = DensityMatrix(np.diag(np.ones(4**n)/4**n))
    pure = DensityMatrix.from_label('0')

    return mixed.tensor(pure)

def pre_processing_ori(n:int):
    # Pre processing circuit
    qc = QuantumCircuit(2*n+1,1)
    qc.set_density_matrix(init_state(n))
    qc.h(0)
    for i in range(n):
        qc.cswap(0,i+1,n+1+i)
    qc.barrier()
    
    return qc

def post_processing_ori(n:int):
    # Post processing circuit
    qc = QuantumCircuit(2*n+1,1)
    qc.barrier()
    for i in range(n):
        qc.cswap(0,i+1,n+1+i)
    qc.h(0)

    qc.save_probabilities([0])
    return qc

def fid_circuit(U:QuantumCircuit,V:QuantumCircuit):
    N = U.num_qubits
    if V.num_qubits != N:
        raise ValueError('Circuits must have the same number of qubits.')

    qc0 = pre_processing_ori(N)
    qc1 = post_processing_ori(N)

    qlist0 = list(range(1,N+1))
    qlist1 = list(range(N+1,2*N+1))

    return qc0.compose(U,qlist0).compose(V,qlist1).compose(qc1)

def gate_fidelity_def(U:Operator,V:Operator):
    return np.abs((U.adjoint() @ V).to_matrix().trace())**2 / U.dim[0] / U.dim[1]

def gate_fidelity_qiskit(U:Operator,V:Operator):
    return process_fidelity(U,V)

def gate_fidelity_circuit(U:QuantumCircuit,V:QuantumCircuit):
    circ = transpile(fid_circuit(U,V),simulator)
    result = simulator.run(circ).result()
    return result.data()['probabilities'][0]*2-1

def gate_fidelity_para(full_qc:QuantumCircuit,para:Mapping):
    circ = transpile(full_qc.assign_parameters(para),simulator)
    result = simulator.run(circ).result()
    return result.data()['probabilities'][0]*2-1

class Evaluator():
    def __init__(self,method:str='para',**kwargs):
        self.method = method
        if method == 'para':
            if 'target' not in kwargs:
                raise ValueError(f'`target` must be provided for "{method}" method.')
            if 'vqc' not in kwargs:
                raise ValueError(f'`vqc` must be provided for "{method}" method.')
            self.target = kwargs['target']
            self.vqc = kwargs['vqc']
            self.qc = fid_circuit(self.target,self.vqc)

            self.evaluate = lambda pa: gate_fidelity_para(self.qc,pa)
        elif method == 'circuit':
            self.evaluate = gate_fidelity_circuit
        elif method == 'direct':
            self.evaluate = gate_fidelity_qiskit
        else:
            raise ValueError('Invalid method.')
