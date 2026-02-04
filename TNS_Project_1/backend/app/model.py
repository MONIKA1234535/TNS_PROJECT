import torch
import torch.nn as nn

class ManufacturingModel(nn.Module):
    def __init__(self):
        super(ManufacturingModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(17, 64),   # Input layer (17 features)
            nn.ReLU(),
            nn.Linear(64, 32),   # Hidden layer
            nn.ReLU(),
            nn.Linear(32, 1)    # Output layer
        )
    
    def forward(self, x):
        return self.layers(x)