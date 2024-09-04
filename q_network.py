import torch
import random
from collections import deque
import numpy as np
from game2048 import Game2048
from visuals import Visual


class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

input_dim = 16  # 4x4 board flattened
output_dim = 4  # 4 possible moves (up, down, left, right)
q_network = QNetwork(input_dim, output_dim)

class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
    
def train_q_network(env, q_network, episodes=1000, batch_size=16, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
    criterion = torch.nn.HuberLoss()
    replay_buffer = ReplayBuffer()
    epsilon = epsilon_start

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1, 2, 3])
            else:
                q_values = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = torch.argmax(q_values).item()

            next_state, reward = env.move(['up', 'down', 'left', 'right'][action])
            total_reward += reward
            done = env.is_game_over()

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            if len(replay_buffer) >= batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(batch_size)
                
                # Ensure batch_state requires gradients
                batch_state.requires_grad_(True)
                
                q_values = q_network(batch_state)
                batch_action_unsqueezed = batch_action.unsqueeze(1).long()
                q_value = q_values.gather(1, batch_action_unsqueezed).squeeze(1)
                next_q_values = q_network(batch_next_state)
                next_q_value = next_q_values.max(1)[0]

                expected_q_value = batch_reward + gamma * next_q_value * (1 - batch_done)
                
                # Ensure expected_q_value does not require gradients
                expected_q_value = expected_q_value.detach()
                
                loss = criterion(q_value, expected_q_value)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

        # Example: Train the Q-network
env = Game2048()
train_q_network(env, q_network)
torch.save(q_network.state_dict(), 'q_network.pth')
env.reset()
visual = Visual()
visual.model_play(q_network, env)


