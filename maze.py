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
    def __init__(self, maze_file):
        with open(maze_file, 'r') as f:
            maze_def = json.load(f)
        
        self.maze = {(cell['x'], cell['y']): cell for cell in maze_def['cell_info']}
        self.config = maze_def['rlhp']
        self.starting_states = [state for state in self.maze.keys() if not self.maze[state]['goal']]
        self.action_map = ['n', 's', 'e', 'w']
        self.epsilon_map = {state: self.config['epsilon'] for state in self.maze.keys()}

    def state_to_tensor(self, state):
        x, y = state
        tensor_state = torch.tensor([x, y], dtype=torch.float32)
        return tensor_state

    def step(self, state, action_index):
        action = self.action_map[action_index]
        x, y = state
        cell = self.maze.get((x, y))
        if not cell:
            return state, self.config['rIllegal'], False

        if cell['legal'][action]:
            if action == 'n':
                y -= 1
            elif action == 's':
                y += 1
            elif action == 'e':
                x += 1
            elif action == 'w':
                x -= 1
        else:
            return state, self.config['rIllegal'], False

        new_state = (x, y)
        if self.maze[new_state]['goal']:
            reward = self.config['rGoal']
            done = True
        else:
            reward = self.config['rLegal']
            done = False

        return new_state, reward, done
    
    def get_random_start_state(self):
        return random.choice(self.starting_states)
    
    def get_epsilon(self, state):
        return self.epsilon_map[state]

    def update_epsilon(self, state):
        self.epsilon_map[state] = max(self.config['min_epsilon'], self.epsilon_map[state] * self.config['epsilon_decay'])


def train(model, environment, optimizer, device, epochs=1000):
    config = environment.config
    epsilon = config['epsilon']
    epsilon_decay = config['epsilon_decay']
    min_epsilon = config['min_epsilon']
    gamma = config['gamma']
    for epoch in range(epochs):
        state = environment.get_random_start_state()
        done = False
        total_reward = 0
        while not done:
            state_tensor = environment.state_to_tensor(state).to(device)
            q_values = model(state_tensor)
            epsilon = environment.get_epsilon(state)
            if random.random() < epsilon:
                action_index = random.randint(0, 3)  # Random action index (0 to 3)
            else:
                action_index = torch.argmax(q_values).item()
            next_state, reward, done = environment.step(state, action_index)
            #print(f"action index: {action_index}, next state: {next_state}, reward: {reward}. q_values: {q_values}")
            total_reward += reward
            next_state_tensor = environment.state_to_tensor(next_state).to(device)
            next_q_values = model(next_state_tensor)
            max_next_q = torch.max(next_q_values)
            target_q = reward + gamma * max_next_q

            # Only update the Q-value for the taken action
            target = q_values.clone().detach()
            target[action_index] = target_q

            loss = nn.MSELoss()(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            environment.update_epsilon(state)
            state = next_state

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Reward = {total_reward}")
            print(f"action index: {action_index}, epsilon: {epsilon}, reward: {reward}. q_values: {q_values}")

def extract_policy(model, environment, device):
    policy = {}
    for (x, y), cell in environment.maze.items():
        state = (x, y)
        state_tensor = environment.state_to_tensor(state).to(device)
        q_values = model(state_tensor)
        action_index = torch.argmax(q_values).item()
        policy[state] = environment.action_map[action_index]
    return policy

input_size = 2
output_size = 4
environment = MazeEnvironment('maze.json')
model = QNetwork(input_size, environment.config['hiddenSize'], output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=environment.config['alpha'])
train(model, environment, optimizer, device)

# Extract and print the learned policy
policy = extract_policy(model, environment, device)
print("Learned Policy:")
for state, action in policy.items():
    print(f"State {state}: Best Action = {action}")
