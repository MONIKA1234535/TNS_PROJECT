import torch
import torch.nn as nn

class ManufacturingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(17, 64)
        self.fc2 = nn.Linear(64, 2)    # 🔥 2 outputs

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
