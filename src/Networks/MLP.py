import torch as torch
import torch.nn as nn
from copy import deepcopy

def sigma_activation(x: torch.tensor):
    return torch.clamp(x, min=0, max=1)

# Class for CReLU activation function
class CReLU(nn.Module):
    def __init__(self): 
        super(CReLU, self).__init__() 
          
    def forward(self, x: torch.tensor):
        return sigma_activation(x)

# Class for constructions on multi-layer percepetrons
class MLP(nn.Module):
    def __init__(self, weights: list[torch.tensor] = [], biases: list[torch.tensor] = []) -> None:
        super().__init__()
        self.weights = weights
        self.biases = biases
        self.num_layers = len(weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    @staticmethod
    def vertically_append_biases(bias_tensor1: torch.tensor, bias_tensor2: torch.tensor) -> torch.tensor:
        return torch.cat((bias_tensor1, bias_tensor2), dim=0)
    
    @staticmethod
    def vertically_append_weights(weight_tensor1: torch.tensor, weight_tensor2: torch.tensor) -> torch.tensor:
        num_inputs_mat1, num_outputs_mat1 = weight_tensor1.shape[1], weight_tensor1.shape[0]
        num_inputs_mat2, num_outputs_mat2 = weight_tensor2.shape[1], weight_tensor2.shape[0]

        B = torch.zeros((num_outputs_mat1, num_inputs_mat2), dtype=weight_tensor1.dtype)
        C = torch.zeros((num_outputs_mat2, num_inputs_mat1), dtype=weight_tensor1.dtype)
    
        return torch.cat((torch.cat((weight_tensor1, B), dim=1), torch.cat((C, weight_tensor2), dim=1)), dim=0)

    def vertically_append_ReLUs(self, ReLU2: 'MLP') -> 'MLP':
        if self.num_layers != ReLU2.num_layers:
            raise ValueError("Number of layers in the two networks must be the same for vertical composition.")
        self.weights = [self.vertically_append_weights(w1, w2) for w1, w2 in zip(self.weights, ReLU2.weights)]
        self.biases = [self.vertically_append_biases(b1, b2) for b1, b2 in zip(self.biases, ReLU2.biases)]
        
        return self
    
    def horizontally_append_ReLUs(self, ReLU2: 'MLP') -> 'MLP':
        self.weights = ReLU2.weights + self.weights
        self.biases = ReLU2.biases + self.biases
        self.num_layers += ReLU2.num_layers

        return self
    

class ReLUNetwork(MLP):
    def construct_layers(self) -> None:
        self.layers = nn.ModuleList()
        for i in range(0, self.num_layers):
            input_size, output_size = self.weights[i].shape[1], self.weights[i].shape[0]

            layer = nn.Linear(input_size, output_size)
            layer.weight.data = self.weights[i]
            layer.bias.data = self.biases[i]
            self.layers.append(layer)

            if i != self.num_layers - 1:
                self.layers.append(nn.ReLU())

class CReLUNetwork(MLP):
    def construct_layers(self) -> None:
        self.layers = nn.ModuleList()
        for i in range(0, self.num_layers):
            input_size, output_size = self.weights[i].shape[1], self.weights[i].shape[0]

            layer = nn.Linear(input_size, output_size)
            layer.weight.data = self.weights[i]
            layer.bias.data = self.biases[i]
            self.layers.append(layer)

            if i != self.num_layers - 1:
                self.layers.append(CReLU())


def transform_ReLU_to_CReLU(ReLU: ReLUNetwork) -> CReLUNetwork:
    CReLUNet = CReLUNetwork([], [])
    CReLUNet.num_layers = ReLU.num_layers

    for i in range(ReLU.num_layers - 1):
        ReLU_weight = ReLU.weights[i]
        ReLU_bias = ReLU.biases[i]
        num_neurons = ReLU_weight.shape[0]

        CReLU_weight, CReLU_bias, updated_ReLU_weight = [], [], []

        for j in range(num_neurons):
            contributing_weight = ReLU_weight[j]
            contributing_bias = ReLU_bias[j]
            max_weighted_input = int(torch.ceil(torch.sum(contributing_weight[contributing_weight > 0]) + contributing_bias).item())

            for u in range(max(1, max_weighted_input)):
                CReLU_weight.append(contributing_weight)
                CReLU_bias.append(contributing_bias.item() - u)

                if len(updated_ReLU_weight) == 0:
                    updated_ReLU_weight = ReLU.weights[i + 1][:, j].view(-1, 1)
                else:
                    updated_ReLU_weight = torch.hstack((updated_ReLU_weight, ReLU.weights[i + 1][:, j].view(-1, 1)))

        CReLUNet.weights.append(torch.vstack(CReLU_weight))
        CReLUNet.biases.append(torch.tensor(CReLU_bias, dtype=torch.float64))
        ReLU.weights[i + 1] = updated_ReLU_weight

    CReLUNet.weights.append(ReLU.weights[-1])
    CReLUNet.biases.append(ReLU.biases[-1])

    return CReLUNet
        