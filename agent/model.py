import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # Define the third fully connected layer
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        # Define the final layer that will output the action values
        self.fc4 = nn.Linear(fc3_dims, n_actions)
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # Define loss function
        self.loss = nn.MSELoss()
        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)
        return actions
