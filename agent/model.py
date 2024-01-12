import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        # Learning rate for the optimizer
        self.alpha = alpha
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # Define the final layer that will output the action values
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        # Define loss function, Mean Squared Error
        self.loss = nn.MSELoss()
        # Determine to use GPU (cuda) or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # Forward method defines how the model processes input data
    def forward(self, state):
        # Apply ReLU activation function after each fully connected layer
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions
