import numpy as np
import random
import matplotlib.pyplot as plt
import time
import tracemalloc  # For CPU memory tracking

grid_size = 10

#Actions
possible_actions = 4 #Up/Down/Right/Left
actions = [(0,1),  (0,-1), (1,0), (-1,0)] #Right/Left/Down/Up

#States
state_current = (0,0)
state_goal = (1,8)
state_obstacle = {(0, 3), (0, 2), (1,3), (1,2), (2,3), (2,2), (3,3), (3,2), (2,6), (2,7), (2,8)}  # Set of obstacle coordinates

#Rewards
reward_goal = 50
reward_move = -1
reward_obstacle = -15

#Q-learning parameters
alpha = 0.5 #learning rate
gamma = 0.9 #discount factor
epsilon = 0.9 #exploration rate
epsilon_decay = 0.999 #epsilon decay
min_epsilon = 0.05 #minimum epsilon
episodes = 2500 #number of episodes to run

#Initialize q-table
q_table = np.zeros((grid_size, grid_size, possible_actions))

#Get the reward funciton
def get_reward(state):
    if state == state_goal:
        return reward_goal
    elif state == state_obstacle:
        return reward_obstacle
    else:
        return reward_move

#Check if state is possible
def is_state_possible(state):
    if 0 <= state[0] < grid_size and 0 <= state[1] < grid_size and state not in state_obstacle:
        return True
    else:
        return False

#Take an action and transmit to next state
def take_action(action, state):
    row, col = state
    row_delta, col_delta = action
    new_state = (row + row_delta, col + col_delta)
    if is_state_possible(new_state):
        return new_state
    else:
        return state

# Initialize tracking metrics for visualization
episode_rewards = []
episode_steps = []
episode_times = []       # Per-episode computation time
episode_cpu_peak = []    # Peak CPU memory (in MB) per episode

# Start overall profiling
total_start_time = time.perf_counter()
tracemalloc.start()

#Start Q-learning
for ep in range(episodes):
    episode_start_time = time.perf_counter()
    snapshot_before = tracemalloc.take_snapshot()  # Memory snapshot before episode

    state_current = (0,0)
    done =  False
    total_reward = 0
    steps = 0

    while not done:
        # if (random.random() < epsilon):
        #     action = random.randint(0,3)
        # else:
        #     action = np.argmax(q_table[state_current[0], state_current[1]]) #get the best action for the current state
        
        action = np.argmax(q_table[state_current[0], state_current[1]]) #get the best action for the current state
        state_next = take_action(actions[action], state_current) #take the best action and get to the next state

        reward_new_state = get_reward(state_next) #get the reward of next state

        total_reward+=reward_new_state

        q_value_current = q_table[state_current[0], state_current[1], action]
        q_best_value_next = np.max(q_table[state_next[0], state_next[1]]) #get the max value in the next state
        q_table[state_current[0], state_current[1], action] = ((1-alpha)*q_value_current) + \
            (alpha*(reward_new_state + gamma*q_best_value_next))

        #go to the new state
        state_current = state_next

        #exit the while loop is terminal state is reached
        if state_current == state_goal:
            done = True

        steps+=1

    # Record episode-level profiling metrics
    snapshot_after = tracemalloc.take_snapshot()
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    peak_memory = max([stat.size_diff for stat in stats] or [0])
    episode_cpu_peak.append(peak_memory / (1024**2))  # Convert bytes to MB

    episode_end_time = time.perf_counter()
    episode_duration = episode_end_time - episode_start_time
    episode_times.append(episode_duration)

    # Record metrics for visualization
    episode_rewards.append(total_reward)
    #print(f"Last steps: {steps}")
    episode_steps.append(steps)

    # Apply epsilon decay
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


# End overall profiling
total_end_time = time.perf_counter()
total_duration = total_end_time - total_start_time
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("\n--- Profiling Report ---")
print(f"Total training time: {total_duration:.4f} seconds")
print(f"Peak CPU memory usage during training: {peak / (1024**2):.2f} MB")
avg_episode_time = np.mean(episode_times)
print(f"Average time per episode: {avg_episode_time:.4f} seconds")

# --- Visualization of Performance Metrics ---

plt.figure(figsize=(14, 10))

# Episode time plot
plt.plot(episode_times, label="Episode Duration (s)")
plt.xlabel("Episode")
plt.ylabel("Time (seconds)")
plt.title("Computation Time per Episode")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(14, 10))
# CPU Memory usage plot
plt.plot(episode_cpu_peak, label="CPU Peak Memory (MB)", color="green")
plt.xlabel("Episode")
plt.ylabel("Memory (MB)")
plt.title("CPU Memory Usage per Episode")
plt.legend()
plt.grid()
plt.show()

# # Plot total rewards per episode
# plt.subplot(1, 2, 1)
# plt.plot(episode_rewards, label="Total Reward per Episode")

# # Annotate the last point (no arrow)
# last_episode = len(episode_rewards) - 1
# last_reward = episode_rewards[-1]
# plt.scatter(last_episode, last_reward, color='red')  # Highlight the point

# plt.text(
#     last_episode,
#     last_reward + (max(episode_rewards) * 0.1),  # Offset the text above the point
#     f'{last_reward:.2f}',
#     fontsize=9,
#     ha='center'
# )

# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("Total Reward Over Episodes")
# plt.legend()
# plt.grid()



# # Plot steps per episode
# plt.subplot(1, 2, 2)
# plt.plot(episode_steps, label="Steps per Episode", color="orange")
# plt.xlabel("Episode")
# plt.ylabel("Steps Taken")
# plt.title("Steps Taken to Reach Goal")
# plt.legend()
# plt.grid()


# plt.tight_layout()
# plt.show()


plt.figure(figsize=(14, 6))  # Optional: Make the figure wider for clarity

### First subplot: Total Rewards ###
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label="Total Reward per Episode")

# Last point (reward)
last_episode_reward = len(episode_rewards) - 1
last_reward = episode_rewards[-1]

plt.scatter(last_episode_reward, last_reward, color='red', zorder=5)

# Guide lines
plt.axhline(y=last_reward, color='gray', linestyle='--', linewidth=0.8)
# plt.axvline(x=last_episode_reward, color='gray', linestyle='--', linewidth=0.8)

# Ticks update
#xticks_rewards = plt.xticks()[0].tolist()
yticks_rewards = plt.yticks()[0].tolist()

# if last_episode_reward not in xticks_rewards:
#     xticks_rewards.append(last_episode_reward)
if last_reward not in yticks_rewards:
    yticks_rewards.append(last_reward)

# xticks_rewards = sorted(xticks_rewards)
yticks_rewards = sorted(yticks_rewards)

# plt.xticks(xticks_rewards)
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

# Last point (steps)
last_episode_steps = len(episode_steps) - 1
last_steps = episode_steps[-1]

plt.scatter(last_episode_steps, last_steps, color='red', zorder=5)

# Guide lines
plt.axhline(y=last_steps, color='gray', linestyle='--', linewidth=0.8)
#plt.axvline(x=last_episode_steps, color='gray', linestyle='--', linewidth=0.8)

# Ticks update
# xticks_steps = plt.xticks()[0].tolist()
yticks_steps = plt.yticks()[0].tolist()

# if last_episode_steps not in xticks_steps:
#     xticks_steps.append(last_episode_steps)
if last_steps not in yticks_steps:
    yticks_steps.append(last_steps)

# xticks_steps = sorted(xticks_steps)
yticks_steps = sorted(yticks_steps)

# plt.xticks(xticks_steps)
plt.yticks(yticks_steps)

# Axis limits
plt.xlim(left=0, right=last_episode_steps + 1)

plt.xlabel("Episode")
plt.ylabel("Steps Taken")
plt.title("Steps Taken to Reach Goal")
plt.legend()
plt.grid()

plt.tight_layout()  # Adjust spacing between subplots
plt.show()

# Visualization
# 1. Plot the Q-values (value landscape)
plt.figure(figsize=(12, 5))

# Subplot 1: Heat map of Q-values
plt.subplot(1, 2, 1)
plt.imshow(np.max(q_table, axis=2), cmap="coolwarm", interpolation="none")
plt.colorbar(label="Value of Best Action per State")
plt.title("Learned Q-Values (Best Action in Each State)")
plt.xlabel("X Coordinate (Columns)")
plt.ylabel("Y Coordinate (Rows)")

# Mark obstacles, goal, and start on the Q-values heat map
for i in range(grid_size):
    for j in range(grid_size):
        if (i, j) in state_obstacle:
            plt.text(j, i, 'X', ha='center', va='center', color='red', fontsize=15)  # Mark obstacles
        elif (i, j) == state_goal:
            plt.text(j, i, 'G', ha='center', va='center', color='green', fontsize=15)  # Mark goal
        elif (i, j) == (0, 0):
            plt.text(j, i, 'S', ha='center', va='center', color='blue', fontsize=15)  # Mark start

plt.xlim(-0.5, grid_size - 0.5)
plt.ylim(-0.5, grid_size - 0.5)
plt.gca().invert_yaxis()
plt.grid(True)
plt.xticks(range(grid_size))
plt.yticks(range(grid_size))

# Subplot 2: Policy visualization with path and obstacles
plt.subplot(1, 2, 2)
for i in range(grid_size):
    for j in range(grid_size):
        if (i, j) in state_obstacle:
            plt.text(j, i, 'X', ha='center', va='center', color='red', fontsize=15)  # Mark obstacles
        elif (i, j) == state_goal:
            plt.text(j, i, 'G', ha='center', va='center', color='green', fontsize=15)  # Mark goal
        elif (i, j) == (0, 0):
            plt.text(j, i, 'S', ha='center', va='center', color='blue', fontsize=15)  # Mark start
        else:
            best_action = np.argmax(q_table[i, j])
            if best_action == 3:  # Up
                plt.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif best_action == 2:  # Down
                plt.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif best_action == 1:  # Left
                plt.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
            elif best_action == 0:  # Right
                plt.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Path Visualization: Overlay the agent's path
state = (0, 0)
path = [state]
while state != state_goal:
    action = np.argmax(q_table[state[0], state[1]])
    state = take_action(actions[action], state)
    path.append(state)

# Extract x and y coordinates for path plotting
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

print("Path from Start to Goal:", path)
