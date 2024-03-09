import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class ReLUNetwork(nn.Module):
    def __init__(self, weights, biases):
        super().__init__()
        self.layers = nn.ModuleList()
        numLayers = len(weights)
        for i in range(0, numLayers):
            input_size, output_size = np.shape(weights[i])[1], np.shape(weights[i])[0]
            print(weights[i].shape)
            
            layer = nn.Linear(input_size, output_size)
            layer.weight.data = torch.from_numpy(weights[i])
            layer.bias.data = torch.from_numpy(biases[i])
            self.layers.append(layer)
            
            if i != numLayers - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def compose(self, ReLU2):
        lastLayer1 = self.layers[-1]
        firstLayer2 = ReLU2.layers[0]
        d2 = lastLayer1.out_features

        Id2 =  torch.eye(d2)
        print(Id2)
        TL1 =  torch.cat((Id2, -Id2), 0)
        T1  =  torch.cat((Id2, -Id2), 1)

        wL1 = torch.matmul(TL1, lastLayer1.weight.data)
        bL1 = torch.matmul(TL1, lastLayer1.bias.data)

        w1 = torch.matmul(firstLayer2.weight.data, T1)
        b1 = torch.matmul(firstLayer2.bias.data, T1)

        layerL1 = nn.Linear(lastLayer1.in_features, 2*d2)
        layerL1.weight.data = wL1
        layerL1.bias.data = bL1
        
        self.layers.pop(len(self.layers) - 1)
        self.layers.append(layerL1)
        self.layers.append(nn.ReLU())

        layer1 = nn.Linear(2*d2, firstLayer2.out_features)
        layer1.weight.data = w1
        layer1.bias.data = b1

        self.layers.append(layer1)

        for i in range(1, len(ReLU2.layers)):
            self.layers.append(ReLU2.layers[i])

        return self
