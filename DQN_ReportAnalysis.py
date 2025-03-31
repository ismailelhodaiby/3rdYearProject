import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import tracemalloc  # for tracking CPU memory usage
import psutil      

# --- Environment Setup ---
grid_size = 10
goal_state = (1, 8)
obstacles_state = {(0, 3), (0, 2), (1,3), (1,2), (2,3), (2,2), (3,3), (3,2), (2,6), (2,7), (2,8)}  # Set of obstacle coordinates
points_state = (3, 1)
num_actions = 4  # Up, Down, Left, Right

# Rewards
reward_goal = 50
reward_default = -1
reward_obstacles = -15
reward_points = 0

# Action mapping
actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

def is_valid_position(pos):
    return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size

def take_action(state, action):
    row, col = state
    dr, dc = actions[action]
    new_state = (row + dr, col + dc)
    return new_state if is_valid_position(new_state) else state

def get_reward(state):
    if state == goal_state:
        return reward_goal
    elif state in obstacles_state:
        return reward_obstacles
    # elif state == points_state:
    #     return reward_points
    else:
        return reward_default

# --- DQN and Replay Buffer ---
class DQN(nn.Module):
    def __init__(self, input_dim=2, output_dim=4):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# convert state tuple to tensor (normalized)
def state_to_tensor(state, device):
    return torch.tensor([state[0] / (grid_size - 1), state[1] / (grid_size - 1)],
                        dtype=torch.float32, device=device).unsqueeze(0)

# --- Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 32
gamma = 0.9
learning_rate = 0.001
epsilon_start = 1.0
epsilon_final = 0.05
epsilon_decay = 300  # decay over episodes
num_episodes = 2500
max_steps_per_episode = 100  # cap steps to avoid infinite loops
target_update_freq = 50  # update target network every N episodes

# --- Initialize Networks and Optimizer ---
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # target network in evaluation mode

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

replay_buffer = ReplayBuffer(10000)

# --- Training Loop ---
episode_rewards = []
episode_steps = []

# Start overall timer and CPU memory tracking
total_start_time = time.perf_counter()
tracemalloc.start()

# Record GPU memory (if using CUDA) by resetting peak stats:
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

episode_times = []
episode_cpu_peak = []  # record peak memory usage per episode
episode_gpu_peak = []  # record peak GPU memory usage per episode (if applicable)


for episode in range(num_episodes):
    episode_start_time = time.perf_counter()
    snapshot_before = tracemalloc.take_snapshot()

    state = (0, 0)
    total_reward = 0
    done = False
    steps = 0

    # Decay epsilon over time (exponential decay)
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * episode / epsilon_decay)

    while not done and steps < max_steps_per_episode:
        steps += 1
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            with torch.no_grad():
                q_vals = policy_net(state_to_tensor(state, device))
            action = int(torch.argmax(q_vals).item())
        # with torch.no_grad():
        #     q_vals = policy_net(state_to_tensor(state, device))
        # action = int(torch.argmax(q_vals).item())

        new_state = take_action(state, action)
        reward = get_reward(new_state)
        total_reward += reward
        done = (new_state == goal_state)
        
        replay_buffer.push(state, action, reward, new_state, done)
        state = new_state

        # Sample mini-batch and train if we have enough samples
        if len(replay_buffer) >= batch_size:
            states, actions_b, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # Convert batches to tensors
            states_tensor = torch.tensor([[s[0] / (grid_size-1), s[1] / (grid_size-1)] for s in states],
                                         dtype=torch.float32, device=device)
            next_states_tensor = torch.tensor([[s[0] / (grid_size-1), s[1] / (grid_size-1)] for s in next_states],
                                              dtype=torch.float32, device=device)
            actions_tensor = torch.tensor(actions_b, dtype=torch.long, device=device).unsqueeze(1)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
            
            # Compute current Q values
            current_q_values = policy_net(states_tensor).gather(1, actions_tensor).squeeze(1)
            
            # Compute target Q values using target network
            with torch.no_grad():
                max_next_q_values = target_net(next_states_tensor).max(1)[0]
                target_q_values = rewards_tensor + gamma * max_next_q_values * (1 - dones_tensor)
            
            loss = criterion(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

    
    # Record CPU memory usage after episode
    snapshot_after = tracemalloc.take_snapshot()
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    # Ssum up the memory usage differences (in bytes)
    peak_memory = max([stat.size_diff for stat in stats] or [0])
    episode_cpu_peak.append(peak_memory / (1024**2))  # Convert to MB
    # Record GPU peak memory (if using CUDA)
    if device.type == "cuda":
        gpu_peak = torch.cuda.max_memory_allocated() / 1024**2  # in MB
        episode_gpu_peak.append(gpu_peak)
    
    episode_end_time = time.perf_counter()
    episode_duration = episode_end_time - episode_start_time
    episode_times.append(episode_duration)
    
    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    
    print(f'Episode {episode} terminated | Time: {episode_duration:.2f}s, Reward: {total_reward}, Steps: {steps}')

    # Update target network periodically
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")


# Stop overall timing and memory tracking
total_end_time = time.perf_counter()
total_duration = total_end_time - total_start_time

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Total training time: {total_duration:.2f} seconds")
print(f"Peak CPU memory usage during training: {peak / (1024**2):.2f} MB")

if device.type == "cuda":
    overall_gpu_peak = max(episode_gpu_peak) if episode_gpu_peak else 0
    print(f"Peak GPU memory usage during training: {overall_gpu_peak:.2f} MB")

# Compute average time per episode
avg_episode_time = np.mean(episode_times)
print(f"Average time per episode: {avg_episode_time:.2f} seconds")

plt.figure(figsize=(14, 8))

# plt.subplot(2, 1, 1)
plt.plot(episode_times, label="Episode Duration (s)")
plt.xlabel("Episode")
plt.ylabel("Time (seconds)")
plt.title("Episode Computation Time")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(episode_cpu_peak, label="CPU Peak Memory (MB)", color="green")
# if device.type == "cuda":
#     plt.plot(episode_gpu_peak, label="GPU Peak Memory (MB)", color="red")
plt.xlabel("Episode")
plt.ylabel("Memory (MB)")
plt.title("CPU Memory Usage Per Episode")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
# plt.plot(episode_cpu_peak, label="CPU Peak Memory (MB)", color="green")
if device.type == "cuda":
    plt.plot(episode_gpu_peak, label="GPU Peak Memory (MB)", color="red")
plt.xlabel("Episode")
plt.ylabel("Memory (MB)")
plt.title("GPU Memory Usage Per Episode")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


peak_reward = max(episode_rewards)
print("Peak Reward Reached:", peak_reward)

# --- Plotting Performance Metrics ---
plt.figure(figsize=(14, 6))  # Make the figure wider for clarity

### First subplot: Total Rewards ###
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label="Total Reward per Episode")

# Mark the last point (reward)
last_episode_reward = len(episode_rewards) - 1
last_reward = episode_rewards[-1]
plt.scatter(last_episode_reward, last_reward, color='red', zorder=5)

# Draw guide line for the last reward level
plt.axhline(y=last_reward, color='gray', linestyle='--', linewidth=0.8)

# Update yticks so that the last reward appears
yticks_rewards = plt.yticks()[0].tolist()
if last_reward not in yticks_rewards:
    yticks_rewards.append(last_reward)
yticks_rewards = sorted(yticks_rewards)
plt.yticks(yticks_rewards)

# Axis limits
plt.xlim(left=0, right=last_episode_reward + 1)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward Over Episodes")
plt.legend()
plt.grid()

### Second subplot: Steps Taken ###
plt.subplot(1, 2, 2)
plt.plot(episode_steps, label="Steps per Episode", color="orange")

# Mark the last point (steps)
last_episode_steps = len(episode_steps) - 1
last_steps = episode_steps[-1]
plt.scatter(last_episode_steps, last_steps, color='red', zorder=5)

# Draw guide line for the last steps level
plt.axhline(y=last_steps, color='gray', linestyle='--', linewidth=0.8)

# Update yticks so that the last steps value appears
yticks_steps = plt.yticks()[0].tolist()
if last_steps not in yticks_steps:
    yticks_steps.append(last_steps)
yticks_steps = sorted(yticks_steps)
plt.yticks(yticks_steps)

# Axis limits
plt.xlim(left=0, right=last_episode_steps + 1)
plt.xlabel("Episode")
plt.ylabel("Steps Taken")
plt.title("Steps Taken Over Episodes")
plt.legend()
plt.grid()

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# --- Evaluate Policy for Visualization ---
# Create a Q-value table for the grid for plotting
q_table_approx = np.zeros((grid_size, grid_size, num_actions))
for i in range(grid_size):
    for j in range(grid_size):
        with torch.no_grad():
            q_vals = policy_net(state_to_tensor((i, j), device)).cpu().numpy()[0]
        q_table_approx[i, j] = q_vals

# --- Plotting ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.max(q_table_approx, axis=2), cmap="coolwarm", interpolation="none")
plt.colorbar(label="Value of Best Action per State")
plt.title("Learned Q-Values (Best Action in Each State)")
plt.xlabel("X Coordinate (Columns)")
plt.ylabel("Y Coordinate (Rows)")

# Mark obstacles, goal, and start on the Q-values heat map
for i in range(grid_size):
    for j in range(grid_size):
        if (i, j) in obstacles_state:
            plt.text(j, i, 'X', ha='center', va='center', color='red', fontsize=15)  # Mark obstacles
        elif (i, j) == goal_state:
            plt.text(j, i, 'G', ha='center', va='center', color='green', fontsize=15)  # Mark goal
        elif (i, j) == (0, 0):
            plt.text(j, i, 'S', ha='center', va='center', color='blue', fontsize=15)  # Mark start

plt.xlim(-0.5, grid_size - 0.5)
plt.ylim(-0.5, grid_size - 0.5)
plt.gca().invert_yaxis()
plt.grid(True)
plt.xticks(range(grid_size))
plt.yticks(range(grid_size))

plt.subplot(1, 2, 2)
for i in range(grid_size):
    for j in range(grid_size):
        if (i, j) in obstacles_state:
            plt.text(j, i, 'X', ha='center', va='center', color='red', fontsize=15)
        elif (i, j) == goal_state:
            plt.text(j, i, 'G', ha='center', va='center', color='green', fontsize=15)
        elif (i, j) == (0, 0):
            plt.text(j, i, 'S', ha='center', va='center', color='blue', fontsize=15)
        else:
            best_action = np.argmax(q_table_approx[i, j])
            if best_action == 0:
                plt.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif best_action == 1:
                plt.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif best_action == 2:
                plt.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif best_action == 3:
                plt.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Visualize the agent's path
max_path_steps = 1000
state = (0, 0)
path = [state]
steps = 0
while state != goal_state and steps < max_path_steps:
    best_action = np.argmax(q_table_approx[state[0], state[1]])
    new_state = take_action(state, best_action)
    # If no progress is made (stuck), break
    if new_state == state:
        break
    path.append(new_state)
    state = new_state
    steps += 1


if state != goal_state:
    print("Policy did not reach the goal within the step limit.")
else:
    print("Policy reached the goal!")

path_x = [pos[1] for pos in path]
path_y = [pos[0] for pos in path]
plt.plot(path_x, path_y, marker='o', color='purple', markersize=6, linewidth=2, label="Path")
plt.legend()
plt.xlim(-0.5, grid_size - 0.5)
plt.ylim(-0.5, grid_size - 0.5)
plt.gca().invert_yaxis()
plt.grid(True)
plt.xticks(range(grid_size))
plt.yticks(range(grid_size))
plt.title("Learned Policy with Path from Start to Goal and Obstacles")
plt.xlabel("X Coordinate (Columns)")
plt.ylabel("Y Coordinate (Rows)")
plt.show()

# --- Plot Training Reward ---
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward per Episode")
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)

# Draw grid arrows and special markers
for i in range(grid_size):
    for j in range(grid_size):
        if (i, j) in obstacles_state:
            plt.text(j, i, 'X', ha='center', va='center', color='red', fontsize=15)
        elif (i, j) == goal_state:
            plt.text(j, i, 'G', ha='center', va='center', color='green', fontsize=15)
        elif (i, j) == (0, 0):
            plt.text(j, i, 'S', ha='center', va='center', color='blue', fontsize=15)
        else:
            best_action = np.argmax(q_table_approx[i, j])
            if best_action == 0:
                plt.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif best_action == 1:
                plt.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif best_action == 2:
                plt.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif best_action == 3:
                plt.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Visualize the agent's path (even if goal not reached)
max_path_steps = 1000
state = (0, 0)
path = [state]
steps = 0
while steps < max_path_steps:
    best_action = np.argmax(q_table_approx[state[0], state[1]])
    new_state = take_action(state, best_action)
    # If no progress is made (stuck) or if it cycles, break out of loop
    if new_state == state or new_state in path:
        break
    path.append(new_state)
    state = new_state
    steps += 1

if state != goal_state:
    reached_text = "Policy did not reach the goal within the step limit."
    print(reached_text)
else:
    reached_text = "Policy reached the goal!"
    print(reached_text)

path_x = [pos[1] for pos in path]
path_y = [pos[0] for pos in path]
plt.plot(path_x, path_y, marker='o', color='purple', markersize=6, linewidth=2, label="Path")

plt.legend()
plt.xlim(-0.5, grid_size - 0.5)
plt.ylim(-0.5, grid_size - 0.5)
plt.gca().invert_yaxis()
plt.grid(True)
plt.xticks(range(grid_size))
plt.yticks(range(grid_size))
plt.xlabel("X Coordinate (Columns)")
plt.ylabel("Y Coordinate (Rows)")
plt.title("Learned Policy with Path\n" + reached_text)
plt.show()

