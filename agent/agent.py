import numpy as np
import torch
from agent.model import Model
import utils.variables as var


class Agent:
    def __init__(self, gamma, epsilon, alpha, input_dims, fc1_dims, fc2_dims, batch_size, n_actions, junctions,
                 max_mem_size=100000, eps_dec=var.eps_dec, eps_end=var.eps_end):
        # Discount factor for future reward
        self.gamma = gamma
        # Exploration rate for choosing random actions.
        self.epsilon = epsilon
        self.alpha = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        # Junction ids
        self.junctions = junctions
        # Memory parameters for experience replay
        self.max_mem_size = max_mem_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        # List of possible actions
        self.action_space = [i for i in range(n_actions)]
        # Counter of experiences stored
        self.mem_counter = 0
        # Counter of learning iterations
        self.iter_counter = 0
        # Initialize memory for each junction
        self.memory = self.init_memory(junctions)

        self.Q_eval = Model(self.alpha, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions)

    # Initialize memory for storing experience for each junction
    def init_memory(self, junctions):
        memory = dict()
        for junction in junctions:
            memory[junction] = {
                "state_memory": np.zeros((self.max_mem_size, self.input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((self.max_mem_size, self.input_dims), dtype=np.float32),
                "reward_memory": np.zeros(self.max_mem_size, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem_size, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem_size, dtype=np.bool_),
                "mem_counter": 0,
                "iter_counter": 0,
            }
        return memory

    # Store a transition (experience) in memory for a specific junction
    def store_transition(self, state, state_, action, reward, done, junction):
        # Store transition in the replay buffer
        index = self.memory[junction]["mem_counter"] % self.max_mem_size
        self.memory[junction]['state_memory'][index] = state
        self.memory[junction]['new_state_memory'][index] = state_
        self.memory[junction]['action_memory'][index] = action
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]['mem_counter'] += 1

    # Choose an action based on the current policy and exploration rate
    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    # The learning process of the agent, updating the neural network
    def learn(self, junction):
        self.Q_eval.optimizer.zero_grad()

        batch = np.arange(self.memory[junction]['mem_counter'], dtype=np.int32)

        # Prepare batches of data for learning
        state_batch = torch.tensor(self.memory[junction]['state_memory'][batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.memory[junction]['new_state_memory'][batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]['action_memory'][batch]

        # Calculate the Q values for the current and next state
        q_eval = self.Q_eval.forward(state_batch)[batch, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # Calculate the target Q values and loss
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # Update counters and epsilon for exploration
        self.iter_counter += 1
        self.epsilon = (self.epsilon - self.eps_dec
                        if self.epsilon > self.eps_min
                        else self.eps_min)

    # Reset memory counters for specified junction
    def reset(self, junction_numbers):
        for junction_number in junction_numbers:
            self.memory[junction_number]["mem_counter"] = 0

    # Save the state of the neural network model
    def save(self, model_name, best_epoch):
        torch.save({
            'state_dict': self.Q_eval.state_dict(),
            'best_epoch': best_epoch,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'eps_dec':self.eps_dec
        }, f'models/{model_name}.pt')

