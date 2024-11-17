from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister,ClassicalRegister, Parameter, ParameterVector

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

def addition_layer(qc:QuantumCircuit, para = None):
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

def multi_layer(qc:QuantumCircuit, layers:int=1):

    qubits = qc.qubits
    n = qc.num_qubits
    para = ParameterVector('θ', 2*n*layers + 3*n).params

    for i in range(n):
        qc.rz(para[i*3],qubits[i])
        qc.ry(para[i*3+1],qubits[i])
        qc.rz(para[i*3+2],qubits[i])
    qc.barrier()

    for i in range(layers):
        qc, qubits = addition_layer(qc, para[i*2*n + 3*n:(i+1)*2*n + 3*n])

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
        elif isinstance(ansatz, callable):
            self.ansatz = ansatz
        else:
            raise ValueError('Invalid ansatz.')
    
    def get_circuit(self, *args, **kwargs):
        qubits = QuantumRegister(self.n)
        qc,para = self.ansatz(QuantumCircuit(qubits), *args, **kwargs)

        return {
            'circuit': qc,
            'qubits': qubits,
            'para': para
        }
    
    @staticmethod
    def get_vqc(n:int, ansatz:str='efficient_circ', *args, **kwargs):

        vcirc = VCircuitConstructor(n, ansatz)
        return vcirc.get_circuit(*args, **kwargs)
