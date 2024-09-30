from logic_gate import LogicGate


logic_gate = LogicGate()
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

for x1, x2 in inputs:
    print(f"   Inputs: ({x1}, {x2})")
    print(f"   AND Gate of ({x1},{x2}): {logic_gate.and_gate(x1, x2)}")
    print(f"   NAND Gate of ({x1},{x2}): {logic_gate.nand_gate(x1, x2)}")
    print(f"   OR Gate of ({x1},{x2}) : {logic_gate.or_gate(x1, x2)}")
    print(f"   NOR Gate of ({x1},{x2}): {logic_gate.nor_gate(x1, x2)}")
    print(f"   XOR Gate of ({x1},{x2}): {logic_gate.xor_gate(x1, x2)}")
    print("||------------------------||")



