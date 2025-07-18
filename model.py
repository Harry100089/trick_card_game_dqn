import torch
import torch.nn as nn
import torch.optim as optim

class CardGameModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CardGameModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)          # Second hidden layer
        self.fc3 = nn.Linear(64, output_size)  # Output layer (Q-values for each action)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x