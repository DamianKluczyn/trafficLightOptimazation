import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent.model import Model


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, fc1_dims, fc2_dims, batch_size, n_actions, junctions,
                 max_mem_size=100000, eps_dec=5e-4, eps_end=0.01):
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.lr = lr  # learning rate
        self.input_dims = input_dims  # input dimensions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size  # batch size
        self.n_actions = n_actions  # number of actions
        self.junctions = junctions  # junctions ids
        self.max_mem_size = max_mem_size  # maximum memory size
        self.eps_min = eps_end  # minimum value of epsilon
        self.eps_dec = eps_dec  # decrement of epsilon per step

        self.action_space = [i for i in range(n_actions)]  # action space
        self.mem_counter = 0  # memory counter
        self.iter_counter = 0  # iteration counter
        self.memory = self.init_memory(junctions)  # initializing memory

        self.Q_eval = Model(self.lr, self.input_dims, self.fc1_dims, self.fc2_dims,
                            self.n_actions)  # neural network model

    def init_memory(self, junctions):
        # Initialize memory
        memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros((self.max_mem_size, self.input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((self.max_mem_size, self.input_dims), dtype=np.float32),
                "reward_memory": np.zeros(self.max_mem_size, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem_size, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem_size, dtype=np.bool_),
                "mem_counter": 0,
                "iter_counter": 0,
            }
        return memory

    def store_transition(self, state, state_, action, reward, done, junction):
        # Store transition in the replay buffer
        index = self.memory[junction]["mem_counter"] % self.max_mem_size
        self.memory[junction]['state_memory'][index] = state
        self.memory[junction]['new_state_memory'][index] = state_
        self.memory[junction]['action_memory'][index] = action
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]['memory_counter'] += 1

    def choose_action(self, observation):
        # Choose an action based on the current policy
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self, junction):
        # Learning process of the agent
        self.Q_eval.optimizer.zero_grad()

        batch = np.arange(self.memory[junction]['mem_counter'], dtype=np.int32)

        state_batch = torch.tensor(self.memory[junction]['state_memory'][batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.memory[junction]['new_state_memory'][batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]['action_memory'][batch]

        q_eval = self.Q_eval.forward(state_batch)[batch, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_counter += 1
        self.epsilon = (self.epsilon - self.eps_dec
                        if self.epsilon > self.eps_min
                        else self.eps_min)

    def reset(self, junction_numbers):
        for junction_number in junction_numbers:
            self.memory[junction_number]["mem_counter"] = 0

    def save(self, model_name):
        torch.save(self.Q_eval.state.dict(), f'models/{model_name}.bin')
