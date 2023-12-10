import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent.model import Model


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.lr = lr  # learning rate
        self.input_dims = input_dims  # input dimensions
        self.batch_size = batch_size  # batch size
        self.n_actions = n_actions  # number of actions
        self.max_mem_size = max_mem_size  # maximum memory size
        self.eps_min = eps_end  # minimum value of epsilon
        self.eps_dec = eps_dec  # decrement of epsilon per step

        self.action_space = [i for i in range(n_actions)]  # action space
        self.mem_cntr = 0  # memory counter
        self.iter_cntr = 0  # iteration counter
        self.memory = self.init_memory()  # initializing memory

        self.Q_eval = Model(input_dims, 256, 256, n_actions)  # neural network model
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=self.lr)  # optimizer
        self.loss = nn.MSELoss()  # loss function

    def init_memory(self):
        # Initialize memory
        mem = {'state_memory': np.zeros((self.max_mem_size, self.input_dims)),
               'new_state_memory': np.zeros((self.max_mem_size, self.input_dims)),
               'action_memory': np.zeros(self.max_mem_size, dtype=np.int32),
               'reward_memory': np.zeros(self.max_mem_size),
               'terminal_memory': np.zeros(self.max_mem_size, dtype=np.bool_)}
        return mem

    def store_transition(self, state, action, reward, state_, done):
        # Store transition in the replay buffer
        index = self.mem_cntr % self.max_mem_size
        self.memory['state_memory'][index] = state
        self.memory['new_state_memory'][index] = state_
        self.memory['action_memory'][index] = action
        self.memory['reward_memory'][index] = reward
        self.memory['terminal_memory'][index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        # Choose an action based on the current policy
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        # Learning process of the agent
        if self.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.memory['state_memory'][batch], dtype=torch.float).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.memory['new_state_memory'][batch], dtype=torch.float).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.memory['reward_memory'][batch], dtype=torch.float).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory['terminal_memory'][batch], dtype=torch.bool).to(self.Q_eval.device)
        action_batch = self.memory['action_memory'][batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.loss(q_eval, q_target).to(self.Q_eval.device)
        loss.backward()
        self.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
