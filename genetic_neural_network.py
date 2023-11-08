import torch.nn as nn
import copy
import numpy as np
import random
import torch

class GeneticNeuralNetwork(nn.Module):
    # initialise a pytorch neural network with random weights and biases - 10 inputs, 4 outputs, 2 hidden layers
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(10, 16)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(16, 10)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(10, 4)
        self.act_output = nn.Sigmoid()
        self.category = 'RANDOM'
        self.mut_power = ' '
        self.mut_resist = ' '

    # internal method to define output tensor
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


    def clone(self):
        """ returns a mutable deep copy of the instance """
        clone = copy.deepcopy(self)
        return clone


    def mutate(self, mutation_power):
        """ applies a mild peturbation to all weights and biases """
        random_mutation_power = random.uniform(mutation_power*2, mutation_power/2)
        # sum all weights and biases with random standardised-distribution tensors of same shape
        for tensor in self.parameters():
            tensor.data += torch.randn_like(tensor) * random_mutation_power 
        # add instance attributes
        self.category = 'MUTANT'
        self.mut_power = round(random_mutation_power, 2)


    def speciate(self, speciation_power, speciation_resistance):
        """ applies a strong peturbation to a fixed percentage of weights and biases """
        random_speciation_power = random.uniform(speciation_power * 1.5, speciation_power * 0.5)
        random_speciation_resistance = random.uniform(1, speciation_resistance * 0.75)
        # create a uniformly distributed random tensor with same dimensions as each parameter tensor
        for tensor in self.parameters(): 
            rnd_tensor = torch.rand_like(tensor)
            # iterate through array to create boolean mask to prevent some parameters being mutated
            mask_tensor = np.array(rnd_tensor)
            for num in np.nditer(mask_tensor, op_flags=['readwrite']):
                if num > random_speciation_resistance:
                    # modify num in place
                    num[...] = True   
                else:
                    num[...] = False
            # sum unmasked weights and biases with random standardised-distribution tensors of same shape
            tensor.data += (torch.randn_like(tensor) * mask_tensor) * random_speciation_power
        # update instance attributes
        self.category = 'NEWSPEC'
        self.mut_power = round(random_speciation_power, 2)
        self.mut_resist = round(random_speciation_resistance, 2)


    def cross(self, neural_network_2):
        """ returns a child neural network by calculating mean parameter values with a second network """
        child = copy.deepcopy(self)
        # iterate through child's tensors and sum with matching tensor from other parent
        for i, tensor3 in enumerate(child.parameters()):
            for j, tensor2 in enumerate(neural_network_2.parameters()):
                if j == i:
                    tensor3.data += tensor2.data 
                    # divide tensors by two to create average of parents
                    tensor3.data = tensor3.data / 2  
        # update instance attributes
        child.category = 'CROSS'
        return child