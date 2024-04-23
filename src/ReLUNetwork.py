import torch as torch
import torch.nn as nn

class ReLUNetwork(nn.Module):
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
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def set_weights(self, weights: list[torch.tensor]) -> 'ReLUNetwork':
        self.weights = weights
        return self
    
    def set_biases(self, biases: list[torch.tensor]) -> 'ReLUNetwork':
        self.biases = biases
        return self
    
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

    def vertically_append_ReLUs(self, ReLU2: 'ReLUNetwork') -> 'ReLUNetwork':
        if self.num_layers != ReLU2.num_layers:
            raise ValueError("Number of layers in the two networks must be the same for vertical composition.")
        self.weights = [self.vertically_append_weights(w1, w2) for w1, w2 in zip(self.weights, ReLU2.weights)]
        self.biases = [self.vertically_append_biases(b1, b2) for b1, b2 in zip(self.biases, ReLU2.biases)]
        
        return self
    
    def horizontally_append_ReLUs(self, ReLU2: 'ReLUNetwork') -> 'ReLUNetwork':
        self.weights = ReLU2.weights + self.weights
        self.biases = ReLU2.biases + self.biases
        self.num_layers += ReLU2.num_layers

        return self