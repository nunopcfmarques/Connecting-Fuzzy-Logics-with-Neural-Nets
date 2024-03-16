import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class ReLUNetwork:
    def __init__(self, weights: list[np.matrix], biases: list[np.ndarray]):
        self.weights = weights
        self.biases = biases
        self.numLayers = len(weights)


class ReLUNetworkOperations:
    @staticmethod
    def compose_bias(bias_array1: np.ndarray, bias_array2: np.ndarray) -> np.ndarray:
        return np.concatenate((bias_array1, bias_array2))

    @staticmethod
    def compose_weights(weight_matrix1: np.matrix, weight_matrix2: np.matrix) -> np.matrix:
        num_inputs_mat1, num_weights_mat1 = weight_matrix1.shape[1], weight_matrix1.shape[0]
        num_inputs_mat2, num_weights_mat2 = weight_matrix2.shape[1], weight_matrix2.shape[0]

        B = np.zeros((num_weights_mat1, num_inputs_mat2))
        C = np.zeros((num_weights_mat2, num_inputs_mat1))
    
        return np.asmatrix(np.block([[weight_matrix1, B],
                                     [C, weight_matrix2]]))

    @staticmethod
    def compose_vertically(ReLU1: ReLUNetwork, ReLU2: ReLUNetwork) -> ReLUNetwork:
        ReLU3 = ReLUNetwork(weights=[], biases=[])
        ReLU3.numLayers = ReLU1.numLayers
        for i in range(0, ReLU1.numLayers):
            ReLU3.weights.append(ReLUNetworkOperations.compose_weights(ReLU1.weights[i], ReLU2.weights[i]))
            ReLU3.biases.append(ReLUNetworkOperations.compose_bias(ReLU1.biases[i], ReLU2.biases[i]))
        
        return ReLU3

    @staticmethod
    def compose_horizontally(ReLU1: ReLUNetwork, ReLU2: ReLUNetwork) -> ReLUNetwork:
        ReLU3 = ReLUNetwork(weights=[], biases=[])
        for i in range(0, ReLU1.numLayers):
            ReLU3.weights.append(ReLU1.weights[i])
            ReLU3.biases.append(ReLU1.biases[i])

        for i in range(0, ReLU2.numLayers):
            ReLU3.weights.append(ReLU2.weights[i])
            ReLU3.biases.append(ReLU2.biases[i])

        ReLU3.numLayers = ReLU1.numLayers + ReLU2.numLayers
        
        return ReLU3


class ReLUNetworkTorch(nn.Module):
    def __init__(self, weights, biases):
        super().__init__()
        self.layers = nn.ModuleList()
        numLayers = len(weights)
        for i in range(0, numLayers):
            input_size, output_size = np.shape(weights[i])[1], np.shape(weights[i])[0]
            
            layer = nn.Linear(input_size, output_size)
            layer.weight.data = torch.from_numpy(weights[i]).to(dtype=torch.float32)
            layer.bias.data = torch.from_numpy(biases[i]).to(dtype=torch.float32)
            self.layers.append(layer)
            
            if i != numLayers - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
