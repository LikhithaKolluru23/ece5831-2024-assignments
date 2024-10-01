import numpy as np
# creating a class called LogicGate where all gates like AND, NAND, OR, NOR and XOR gates
class LogicGate:
    def __init__(self):
        pass

    def and_gate(self, x1, x2): # Defining AND Gate
        w = np.array([0.5, 0.5]) # weights 
        b = -0.7  # bias
        x = np.array([x1, x2]) # inputs
        
        y = np.sum(w * x) + b # output calculation

        if y > 0:
            return 1
        else:
            return 0
        
        
    def nand_gate(self, x1, x2): # Defining NAND Gate
        w = np.array([-0.5, -0.5]) # weights 
        b = 0.7 # bias
        x = np.array([x1, x2]) # inputs
        
        y = np.sum(w * x) + b # output calculation

        if y > 0:
            return 1
        else:
            return 0
        
    def or_gate(self, x1, x2): # Defining OR Gate
        w = np.array([1.0, 1.0]) # weights 
        b = -0.9 # bias
        x = np.array([x1, x2]) # inputs
        
        y = np.sum(w * x) + b # output calculation
        
        if y > 0:
            return 1
        else:
            return 0
        
    def nor_gate(self, x1, x2): # Defining NOR Gate
        w = np.array([-1.0, -1.0]) # weights 
        b = 0.9 # bias
        x = np.array([x1, x2]) # inputs
        
        y = np.sum(w * x) + b # output calculation
        if y > 0:
            return 1
        else:
            return 0
        
    def xor_gate(self, x1, x2): # Defining XOR Gate using AND of OR and NAND
        y1 = self.or_gate(x1, x2)
        y2 = self.nand_gate(x1, x2)
        return self.and_gate(y1, y2)



   


if __name__ == "__main__":
     help_message = """

This program demonstrates the use of various logic gates (AND, NAND, OR, NOR, XOR) implemented using NumPy.

LogicGate Class:
    - and_gate(x1, x2) : Returns AND of x1 and x2
    - nand_gate(x1, x2): Returns NAND of x1 and x2
    - or_gate(x1, x2)  : Returns OR of x1 and x2
    - nor_gate(x1, x2) : Returns NOR of x1 and x2
    - xor_gate(x1, x2) : Returns XOR of x1 and x2

Example:
    To use the defined logic gates:
    
    from logic_gate import LogicGate
    logic_gate = LogicGate()
    
    # For AND Gate
    result = logic_gate.and_gate(1, 1)
    print("AND Gate(1, 1):", result)
    """
     print(help_message)
   
