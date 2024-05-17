import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.initialize_weights()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

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

def update_target_network(target, source):
    target.load_state_dict(source.state_dict())

def train(model, target_model, environment, optimizer, replay_buffer, device, epochs=1000, batch_size=64, target_update_frequency=100):
    config = environment.config
    gamma = config['gamma']
    for epoch in range(epochs):
        state = environment.get_random_start_state()
        done = False
        total_reward = 0
        steps = 0  # Debugging steps per epoch
        while not done:
            state_tensor = environment.state_to_tensor(state).to(device)
            q_values = model(state_tensor)
            epsilon = environment.get_epsilon(state)
            if random.random() < epsilon:
                action_index = random.randint(0, 3)  # Random action index (0 to 3)
            else:
                action_index = torch.argmax(q_values).item()
            next_state, reward, done = environment.step(state, action_index)
            total_reward += reward
            next_state_tensor = environment.state_to_tensor(next_state).to(device)

            replay_buffer.push(state, action_index, reward, next_state, done)

            state = next_state
            steps += 1  # Increment step count for debugging
            environment.update_epsilon(state)  # Update epsilon for the current state

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
                actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)

                current_q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(next_states_tensor).max(1)[0]
                target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

                loss = nn.MSELoss()(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Reward = {total_reward}")
            print(f"Steps taken in epoch {epoch}: {steps}")
            print(f"Epsilon: {epsilon}")

        if epoch % target_update_frequency == 0:
            update_target_network(target_model, model)

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
target_model = QNetwork(input_size, environment.config['hiddenSize'], output_size)
update_target_network(target_model, model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
target_model.to(device)

optimizer = optim.Adam(model.parameters(), lr=environment.config['alpha'])
replay_buffer = ReplayBuffer(10000)
train(model, target_model, environment, optimizer, replay_buffer, device)

# Extract and print the learned policy
policy = extract_policy(model, environment, device)
print("Learned Policy:")
for state, action in policy.items():
    print(f"State {state}: Best Action = {action}")
