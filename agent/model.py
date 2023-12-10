import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # Define the final layer that will output the action values
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        # Pass the state through the first layer and apply ReLU activation
        x = F.relu(self.fc1(state))
        # Pass the result through the second layer and apply ReLU activation
        x = F.relu(self.fc2(x))
        # Pass the result through the final layer to get the action values
        actions = self.fc3(x)
        return actions
