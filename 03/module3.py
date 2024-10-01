from logic_gate import LogicGate # importing LogicGate class from logic_gate file


logic_gate = LogicGate() # creating an object instance
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)] # defining inputs

for x1, x2 in inputs: #  Looping through all the inputs
    print(f"   Inputs: ({x1}, {x2})")
    print(f"   AND Gate of ({x1},{x2}): {logic_gate.and_gate(x1, x2)}") # for AND gate
    print(f"   NAND Gate of ({x1},{x2}): {logic_gate.nand_gate(x1, x2)}") # for NAND gate
    print(f"   OR Gate of ({x1},{x2}) : {logic_gate.or_gate(x1, x2)}") # for OR gate
    print(f"   NOR Gate of ({x1},{x2}): {logic_gate.nor_gate(x1, x2)}") # for NOR gate
    print(f"   XOR Gate of ({x1},{x2}): {logic_gate.xor_gate(x1, x2)}") # for XOR gate
    print("||------------------------||")



