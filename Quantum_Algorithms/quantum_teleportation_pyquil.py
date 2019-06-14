import os
from pyquil.gates import *
from pyquil.quil import Program
from pyquil.api import QVMConnection
API_KEY = "YOUR_KEY"
USER_ID = "YOUR_ID"

with open(os.path.expanduser('~/.pyquil_config'), 'w') as f:
    f.write(PYQUIL_CONFIG)

def make_bell_pair(q1,q2):
    p  = Program(H(q1), CNOT(q1,q2))
    return p

def teleportation(alice,mid,bob):
    program = make_bell_pair(mid,bob)

    classical_registers = program.declare('classical_registers', memory_size=3)


    program.inst(CNOT(alice,mid))
    program.inst(H(alice))

    program.measure(alice,classical_registers[0])
    program.measure(mid,classical_registers[1])

    program.if_then(classical_registers[1],X[2])
    program.if_then(classical_registers[0],Z[2])

    program.measure(bob,classical_registers[2])

    print(program)
    return program


if __name__ == '__main__':

    qvm = api.QVMConnection()

    # initialize qubit 0 in |1>
    teleport_demo = Program(X(0))
    teleport_demo += teleport(0, 1, 2)
    print("Teleporting |1> state: {}".format(qvm.run(teleport_demo, [2])))

    # initialize qubit 0 in |0>
    teleport_demo = Program()
    teleport_demo += teleport(0, 1, 2)
    print("Teleporting |0> state: {}".format(qvm.run(teleport_demo, [2])))

    # initialize qubit 0 in |+>
    teleport_demo = Program(H(0))
    teleport_demo += teleport(0, 1, 2)
    print("Teleporting |+> state: {}".format(qvm.run(teleport_demo, [2], 10)))

    # initialize qubit 0 in |->
    teleport_demo = Program(H(X(0)))
    teleport_demo += teleport(0, 1, 2)
    print("Teleporting |-> state: {}".format(qvm.run(teleport_demo, [2], 10)))
