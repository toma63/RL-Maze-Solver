import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MazeEnvironment:
    def __init__(self, maze_file, config):
        with open(maze_file, 'r') as f:
            self.maze = json.load(f)
        self.goal = [cell for cell in self.maze if cell['is_goal']]
        assert len(self.goal) == 1, "Maze must have exactly one goal"
        self.goal = self.goal[0]
        self.config = config

    def state_to_tensor(self, state):
        x, y = state
        tensor_state = torch.tensor([x, y], dtype=torch.float32)
        return tensor_state

    def step(self, state, action):
        x, y = state
        cell = next(cell for cell in self.maze if cell['x'] == x and cell['y'] == y)
        moves = cell['moves']
        if action == 'n' and 'n' in moves:
            y -= 1
        elif action == 's' and 's' in moves:
            y += 1
        elif action == 'e' and 'e' in moves:
            x += 1
        elif action == 'w' and 'w' in moves:
            x -= 1
        else:
            return state, self.config['rewards']['illegal'], False
        
        new_state = (x, y)
        if new_state == (self.goal['x'], self.goal['y']):
            reward = self.config['rewards']['goal']
        else:
            reward = self.config['rewards']['legal']

        done = new_state == (self.goal['x'], self.goal['y'])
        return new_state, reward, done

def train(model, environment, optimizer, config, epochs=1000):
    epsilon = config['epsilon']
    gamma = config['gamma']
    for epoch in range(epochs):
        state = (0, 0)  # Assuming starting point
        done = False
        while not done:
            state_tensor = environment.state_to_tensor(state)
            q_values = model(state_tensor)
            if random.random() < epsilon:
                action = random.choice(['n', 's', 'e', 'w'])
            else:
                action = torch.argmax(q_values).item()
            next_state, reward, done = environment.step(state, action)
            next_state_tensor = environment.state_to_tensor(next_state)
            next_q_values = model(next_state_tensor)
            max_next_q = torch.max(next_q_values)
            target_q = reward + gamma * max_next_q

            loss = nn.MSELoss()(q_values[action], target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            epsilon *= config['epsilon_decay']  # Update epsilon

# Load configuration
with open('path_to_hyperparameters_file.json', 'r') as f:
    config = json.load(f)

input_size = 2
hidden_size = 64
output_size = 4
model = QNetwork(input_size, hidden_size, output_size)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=config['alpha'])
environment = MazeEnvironment('path_to_your_maze_file.json', config)
train(model, environment, optimizer, config)
