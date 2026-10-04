import torch.nn as nn
import numpy as np 

class PositionalProbe(nn.Module):

    def __init__(self, input_dim, hidden_dim = 1024, output_dim = 2048):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x

  


    
    