import numpy as np
import torch.nn as nn
import torch as torch

class ReLUNetwork:
    def __init__(self, weights: list[np.matrix] = [], biases: list[np.ndarray] = []) -> None:
        self.weights = weights
        self.biases = biases
        self.numLayers = len(weights)

    @staticmethod
    def vertically_append_biases(bias_array1: np.ndarray, bias_array2: np.ndarray) -> np.ndarray:
        return np.concatenate((bias_array1, bias_array2))

    @staticmethod
    def vertically_append_weights(weight_matrix1: np.matrix, weight_matrix2: np.matrix) -> np.matrix:
        num_inputs_mat1, num_outputs_mat1 = weight_matrix1.shape[1], weight_matrix1.shape[0]
        num_inputs_mat2, num_outputs_mat2 = weight_matrix2.shape[1], weight_matrix2.shape[0]

        B = np.zeros((num_outputs_mat1, num_inputs_mat2))
        C = np.zeros((num_outputs_mat2, num_inputs_mat1))
    
        return np.asmatrix(np.block([[weight_matrix1, B],
                                     [C, weight_matrix2]]))
    
    def compose_vertically(self, ReLU2):
        if self.numLayers != ReLU2.numLayers:
            raise ValueError("Number of layers in the two networks must be the same for vertical composition.")
        self.weights = [self.vertically_append_weights(w1, w2) for w1, w2 in zip(self.weights, ReLU2.weights)]
        self.biases = [self.vertically_append_biases(b1, b2) for b1, b2 in zip(self.biases, ReLU2.biases)]
        
        return self
    
    def compose_horizontally(self, ReLU2):
        if ReLU2.weights[-1].shape[0] != self.weights[0].shape[1]:
            raise ValueError("Number of outputs of the second network must match the number of inputs of the first network.")
        self.weights = ReLU2.weights + self.weights
        self.biases = ReLU2.biases + self.biases
        self.numLayers += ReLU2.numLayers

        return self
    
class ReLUNetworkTorch(nn.Module):
    def __init__(self, weights: list[np.matrix] = [], biases: list[np.ndarray] = []):
        super().__init__()
        self.layers = nn.ModuleList()
        numLayers = len(weights)
        for i in range(0, numLayers):
            input_size, output_size = np.shape(weights[i])[1], np.shape(weights[i])[0]
                
            layer = nn.Linear(input_size, output_size)
            layer.weight.data = torch.from_numpy(weights[i]).to(dtype=torch.float64)
            layer.bias.data = torch.from_numpy(biases[i]).to(dtype=torch.float64)
            self.layers.append(layer)
                
            if i != numLayers - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x