from pytest import approx

from .evaluate import gate_fidelity_circuit,gate_fidelity_def,gate_fidelity_qiskit

from qiskit.quantum_info import random_unitary
from qiskit.circuit import QuantumCircuit

# %%
U = random_unitary(4)
V = random_unitary(4)

qcU = QuantumCircuit(2)
qcU.append(U,[0,1])
qcV = QuantumCircuit(2)
qcV.append(V,[0,1])

print(gate_fidelity_def(U,V))
print(gate_fidelity_qiskit(U,V))
print(gate_fidelity_circuit(qcU,qcV))
print(gate_fidelity_def(U,U))
print(gate_fidelity_qiskit(U,U))
print(gate_fidelity_circuit(qcU,qcU))
# %%

def test_fidelity():
    assert gate_fidelity_qiskit(U,V) == approx(gate_fidelity_def(U,V))
    assert gate_fidelity_circuit(qcU,qcV) == approx(gate_fidelity_def(U,V))

    assert gate_fidelity_qiskit(U,U) == approx(1)
    assert gate_fidelity_circuit(qcU,qcU) == approx(1)