from multilayer_percetron import MultiLayerPerceptron #importing the class MultiLayerPerceptron 
import numpy as np
mlp=MultiLayerPerceptron()  # Creating an instance of MultiLayerPerceptron
mlp.init_network() #calling this method to initialize the network
x = np.array([0.5, 0.6]) # Input vector
# Forward pass
output = mlp.forward(x)
print(f"Output of {x}  as an input: {output}")

xn=np.zeros((2,))
results=[]
for i in range(1, 11):  # from 1 to 10
    # Inner loop to increment the second element
    for j in range(1, 11):  # from 1 to 10
        xn = np.array([i * 0.1, j * 0.1])  # Create array with both elements
        results.append(xn)

for index,value in enumerate(results):
    output=mlp.forward(value)
    print(f"Output of {value} as an input: {output}")