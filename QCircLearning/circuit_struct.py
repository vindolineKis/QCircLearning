from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister,ClassicalRegister, Parameter, ParameterVector
import numpy as np


BIAS_STRENGTH = 1

def simple_circ(qc:QuantumCircuit):
    qubits = qc.qubits

    n = qc.num_qubits
    para = ParameterVector('θ', 2*n).params

    for i in range(n):
        qc.ry(para[i],qubits[i])
    qc.barrier()
    for i in range(n//2):
        qc.cx(qubits[i*2],qubits[i*2+1])
    for i in range((n-1)//2):
        qc.cx(qubits[i*2+1],qubits[i*2+2])
    qc.barrier()
    for i in range(n):
        qc.ry(para[i+n],qubits[i])

    return qc, para

def one_layer(qc:QuantumCircuit):
    qc, para = multi_layer(qc, 1)

    return qc, para

def addition_layer(qc:QuantumCircuit, para = None, bias_strength=BIAS_STRENGTH):
    qubits = qc.qubits
    print(f"bias_strengt at circuits: {bias_strength}")
    n = qc.num_qubits
    if para is None:
        para = ParameterVector('θ', 2*n).params

    for i in range(n//2):
        qc.cx(qubits[i*2],qubits[i*2+1])
    for i in range((n-1)//2):
        qc.cx(qubits[i*2+1],qubits[i*2+2])
    qc.barrier()
    for i in range(n):
        epsilon_ry = np.random.normal(0, bias_strength)
        epsilon_rz = np.random.normal(0, bias_strength*0.0001)
        epsilon_rx = np.random.normal(0, bias_strength)
        qc.rz(epsilon_rz,qubits[i])#rz noise
        qc.ry(para[i*2]+epsilon_ry,qubits[i]) #noisy ry
        qc.rx(epsilon_rx,qubits[i])# rx noise

        qc.ry(epsilon_ry, qubits[i])
        qc.rz(para[i*2+1]+epsilon_rz,qubits[i])
        qc.rx(epsilon_rx, qubits[i])


    return qc, para

def multi_layer(qc:QuantumCircuit, layers:int=1, bias_strength=BIAS_STRENGTH):
    print(f"bias_strength at circuits: {bias_strength}")

    qubits = qc.qubits
    n = qc.num_qubits
    para = ParameterVector('θ', 2*n*layers + 3*n).params

    for i in range(n):
        # add rx, ry noise to noisy rz gate
        epsilon_rx1 = np.random.normal(0, bias_strength)
        epsilon_ry1 = np.random.normal(0, bias_strength)
        epsilon_rz1 = np.random.normal(0, bias_strength*0.0001)
        qc.rx(epsilon_rx1,qubits[i])#rx noise
        qc.ry(epsilon_ry1,qubits[i]) #noisy ry
        qc.rz(para[i*3]+epsilon_rz1,qubits[i])

        epsilon_rx2 = np.random.normal(0, bias_strength)
        epsilon_ry2 = np.random.normal(0, bias_strength)
        epsilon_rz2 = np.random.normal(0, bias_strength*0.0001)
        qc.rx(epsilon_rx2,qubits[i])#rx noise
        qc.rz(epsilon_rz2,qubits[i])#rz noise
        qc.ry(para[i*3+1]+epsilon_ry2,qubits[i])
        
        epsilon_rx3 = np.random.normal(0, bias_strength)
        epsilon_ry3 = np.random.normal(0, bias_strength)
        epsilon_rz3 = np.random.normal(0, bias_strength*0.0001)
        qc.rx(epsilon_rx3,qubits[i])#rx noise
        qc.ry(epsilon_ry3,qubits[i]) #noisy ry
        qc.rz(para[i*3+2]+epsilon_rz3,qubits[i])
    qc.barrier()

    for i in range(layers):
        qc, qubits = addition_layer(qc, para[i*2*n + 3*n:(i+1)*2*n + 3*n])

    para = qc.parameters

    return qc, para

def one_layer_pure(qc:QuantumCircuit):
    qc, para = multi_layer_pure(qc, 1)

    return qc, para

def addition_layer_pure(qc:QuantumCircuit, para = None):
    qubits = qc.qubits

    n = qc.num_qubits
    if para is None:
        para = ParameterVector('θ', 2*n).params

    for i in range(n//2):
        qc.cx(qubits[i*2],qubits[i*2+1])
    for i in range((n-1)//2):
        qc.cx(qubits[i*2+1],qubits[i*2+2])
    qc.barrier()
    for i in range(n):
        qc.ry(para[i*2],qubits[i])
        qc.rz(para[i*2+1],qubits[i])

    return qc, para

def multi_layer_pure(qc:QuantumCircuit, layers:int=1):

    qubits = qc.qubits
    n = qc.num_qubits
    para = ParameterVector('θ', 2*n*layers + 3*n).params

    for i in range(n):
        qc.rz(para[i*3],qubits[i])
        qc.ry(para[i*3+1],qubits[i])
        qc.rz(para[i*3+2],qubits[i])
    qc.barrier()

    for i in range(layers):
        qc, qubits = addition_layer_pure(qc, para[i*2*n + 3*n:(i+1)*2*n + 3*n])

    para = qc.parameters

    return qc, para

_predefined_ansatz = {
    'simple': simple_circ,
    'one_layer': one_layer,
    'efficient_circ': multi_layer
}

class VCircuitConstructor:
    def __init__(self, n:int, ansatz:str='efficient_circ'):
        """
        A constructor for variational quantum circuit.
        It generates a dictionary containing the quantum circuit, quantum register, and parameters.

        Args:
            n (int): number of qubits
            ansatz (str|function): ansatz of the circuit. It can be either a predefined ansatz or a user-defined
                ansatz. If it is a predefined ansatz, it should be a string. Otherwise, it should be a function.

        Returns:
            dict: A dictionary containing the quantum circuit, quantum register, and parameters.

        Use example:

        ```python
        vcirc = VCircuitConstructor.get_vqc(4, 'simple')
        ```
        """
        self.n = n
        if isinstance(ansatz, str):
            if ansatz in _predefined_ansatz:
                self.ansatz = _predefined_ansatz[ansatz]
            else:
                raise ValueError('Invalid ansatz. Please define your own ansatz.')
        elif callable(ansatz):
            self.ansatz = ansatz
        else:
            raise ValueError('Invalid ansatz.')
    
    def get_circuit(self, *args, **kwargs):
        qubits = QuantumRegister(self.n)
        qc,para = self.ansatz(QuantumCircuit(qubits), *args, **kwargs)

        return {
            'circuit': qc,
            'qubits': qubits,
            'para': para,
    
        }
    
    @staticmethod
    def get_vqc(n:int, ansatz:str='efficient_circ', *args, **kwargs):

        vcirc = VCircuitConstructor(n, ansatz)
        return vcirc.get_circuit(*args, **kwargs)
