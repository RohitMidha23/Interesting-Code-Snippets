#!/usr/bin/env python
# coding: utf-8

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute


def make_bell_pair(circuit,q1,q2):
    circuit.h(q1)
    circuit.cx(q1,q2)


coupling_map = [[0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2]]
backend = BasicAer.get_backend("qasm_simulator")


q = QuantumRegister(3,"q")
c0 = ClassicalRegister(1,"c0")
c1 = ClassicalRegister(1,"c1")
c2 = ClassicalRegister(1,"c2")

qc = QuantumCircuit(q, c0, c1, c2, name="teleport")

qc.u3(0.3, 0.2, 0.1, q[0])
make_bell_pair(qc,q[1],q[2])
qc.barrier(q)

qc.cx(q[0],q[1])
qc.h(q[0])
qc.measure(q[0],c0)
qc.measure(q[1],c1)

qc.barrier(q)

qc.x(q[2]).c_if(c1,1)
qc.z(q[2]).c_if(c0,1)
qc.measure(q[2],c2)
qc.draw()


initial_layout = {q[0]: 0,
                  q[1]: 1,
                  q[2]: 2}
job = execute(qc, backend=backend, coupling_map=None, shots=1024)



result = job.result()
print(result.get_counts(qc))
