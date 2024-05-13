import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from src.ReLUNetwork import ReLUNetwork


#CReLU activation function
class CReLU(nn.Module):
    def __init__(self): 
        super(CReLU, self).__init__() 
          
    def forward(self, x):
        clipped_x = torch.clamp(x, min=0, max=1)
        return clipped_x


class CReLUNetwork(nn.Module):
    def __init__(self, weights: list[torch.tensor] = [], biases: list[torch.tensor] = []) -> None:
        super().__init__()
        self.weights = weights
        self.biases = biases
        self.num_layers = len(weights)

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

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def transformReLUToCReLU(ReLU: ReLUNetwork) -> 'CReLUNetwork':
    CReLUNet = CReLUNetwork()
    CReLUNet.num_layers = ReLU.num_layers
    # Last layer doesn't need to be transformed!
    for i in range(ReLU.num_layers - 1):
        ReLU_weight = ReLU.weights[i]
        ReLU_bias = ReLU.biases[i]
        num_neurons = ReLU_weight.shape[0]

        CReLU_weight = None
        CReLU_bias = None
        updated_ReLU_weight = None

        for j in range(num_neurons):
            contributing_weight = ReLU_weight[j]
            contributing_bias = ReLU_bias[j]
            max_weighted_input = int(torch.ceil(torch.sum(contributing_weight[contributing_weight > 0]) + contributing_bias).item())  # bound of the maximum output in the ReLU Network  # bound of the maximum output in the ReLU Network
            # if the max_weighted_input exceeds one then we have to add "neurons", meaning we have to expand the weight matrix and updated the bias
            if max_weighted_input > 1:
                for u in range(0, max_weighted_input):

                    if CReLU_weight is None:
                        CReLU_weight = contributing_weight
                    else:
                        CReLU_weight = torch.vstack((CReLU_weight, contributing_weight))

                    if CReLU_bias is None:
                        CReLU_bias = torch.tensor([contributing_bias.item()], dtype=torch.float64)
                    else:
                        CReLU_bias = torch.hstack((CReLU_bias, torch.tensor([contributing_bias.item() - u], dtype=torch.float64)))

                    if updated_ReLU_weight is None:
                        updated_ReLU_weight = ReLU.weights[i + 1][:, j].view(-1, 1)
                    else:
                        updated_ReLU_weight = torch.hstack((updated_ReLU_weight, ReLU.weights[i + 1][:, j].view(-1, 1)))
            else:
                if CReLU_weight is None:
                    CReLU_weight = contributing_weight
                else:
                    CReLU_weight = torch.vstack((CReLU_weight, contributing_weight))

                if CReLU_bias is None:
                    CReLU_bias = torch.tensor([contributing_bias.item()], dtype=torch.float64)
                else:
                    CReLU_bias = torch.hstack((CReLU_bias, torch.tensor([contributing_bias.item()], dtype=torch.float64)))

                if updated_ReLU_weight is None:
                    updated_ReLU_weight = ReLU.weights[i + 1][:, j].view(-1, 1)
                else:
                    updated_ReLU_weight = torch.hstack((updated_ReLU_weight, ReLU.weights[i + 1][:, j].view(-1, 1)))

        CReLUNet.weights.append(CReLU_weight)
        CReLUNet.biases.append(CReLU_bias)  # Corrected the variable name
        ReLU.weights[i + 1] = updated_ReLU_weight
    
    CReLUNet.weights.append(ReLU.weights[-1])
    CReLUNet.biases.append(ReLU.biases[-1])

    return CReLUNet