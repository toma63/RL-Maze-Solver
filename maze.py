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
            maze_data = json.load(f)
        
        self.maze = {(cell['x'], cell['y']): cell for cell in maze_data}
        self.config = config

    def state_to_tensor(self, state):
        x, y = state
        tensor_state = torch.tensor([x, y], dtype=torch.float32)
        return tensor_state

    def step(self, state, action):
        x, y = state
        cell = self.maze.get((x, y))
        if not cell:
            return state, self.config['rewards']['illegal'], False

        if cell['legal']['action']:
            if action == 'n':
                y -= 1
            elif action == 's':
                y += 1
            elif action == 'e':
                x += 1
            elif action == 'w':
                x -= 1
        else:
            return state, self.config['rewards']['illegal'], False

        new_state = (x, y)
        if self.maze[new_state]['goal']:
            reward = self.config['rewards']['goal']
            done = True
        else:
            reward = self.config['rewards']['legal']
            done = False

        return new_state, reward, done

def train(model, environment, optimizer, config, device, epochs=1000):
    epsilon = config['epsilon']
    gamma = config['gamma']
    for epoch in range(epochs):
        state = (0, 0)  # Assuming starting point
        done = False
        total_reward = 0
        while not done:
            state_tensor = environment.state_to_tensor(state).to(device)
            q_values = model(state_tensor)
            if random.random() < epsilon:
                action = random.choice(['n', 's', 'e', 'w'])
            else:
                action = torch.argmax(q_values).item()
            next_state, reward, done = environment.step(state, action)
            total_reward += reward
            next_state_tensor = environment.state_to_tensor(next_state).to(device)
            next_q_values = model(next_state_tensor)
            max_next_q = torch.max(next_q_values)
            target_q = reward + gamma * max_next_q

            loss = nn.MSELoss()(q_values[action], target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            epsilon *= config['epsilon_decay']  # Update epsilon

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Reward = {total_reward}")

def extract_policy(model, environment, device):
    policy = {}
    for (x, y), cell in environment.maze.items():
        state = (x, y)
        state_tensor = environment.state_to_tensor(state).to(device)
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()
        policy[state] = ['n', 's', 'e', 'w'][action]
    return policy

# Load configuration
with open('path_to_hyperparameters_file.json', 'r') as f:
    config = json.load(f)

input_size = 2
hidden_size = 64
output_size = 4
model = QNetwork(input_size, hidden_size, output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=config['alpha'])
environment = MazeEnvironment('path_to_your_maze_file.json', config)
train(model, environment, optimizer, config, device)

# Extract and print the learned policy
policy = extract_policy(model, environment, device)
print("Learned Policy:")
for state, action in policy.items():
    print(f"State {state}: Best Action = {action}")
