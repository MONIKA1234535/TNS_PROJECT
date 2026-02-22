import torch.nn as nn

class ManufacturingModel(nn.Module):
    def __init__(self):
        super(ManufacturingModel, self).__init__()
        # Layer 1: Matches your checkpoint (20 inputs -> 64 hidden)
        self.fc1 = nn.Linear(20, 64) 
        
        # Layer 2: Matches your checkpoint (64 hidden -> 2 outputs)
        # The error said the checkpoint has size [2, 64] and [2]
        self.fc2 = nn.Linear(64, 2) 
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # No fc3 here because the checkpoint doesn't have it
        x = self.fc2(x) 
        return x