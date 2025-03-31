import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import pandas as pd  # Import pandas for saving data to CSV

# ----------------------------------------------------------------------------------------------
# COPPELIA INITIALIZATION
# ----------------------------------------------------------------------------------------------
client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

# Initialize joints
rightJointHandle = sim.getObject("/joint_right")
leftJointHandle = sim.getObject("/joint_left")
sim.setObjectInt32Param(rightJointHandle, 2000, 1)
sim.setObjectInt32Param(leftJointHandle, 2000, 1)

# Initialize chassis and target
chassisHandle = sim.getObject("/Chassis")
targetHandle = sim.getObject("/Target")

#Speed limits
speedMax = 20 #deg/s
speedMin = 0

#Simultion limits
max_possible_reward = 150
max_sim_time_s = 120
max_steps = 5000

actor_gradients_plots = []
critic_gradients_plots = []

# ----------------------------------------------------------------------------------------------
# NEURAL NETWORK INITIALIZATION
# ----------------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights_actor(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=0.01)  # Reduce gain to avoid extreme values
        nn.init.constant_(m.bias, 0.0)  # Keep bias small

def init_weights_critic(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization
        nn.init.zeros_(m.bias)  # Zero biases (or use small positive value)

#Actor Network (policy network)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Outputs a deterministic action:
         - Linear velocity: Use softplus so it is positive.
         - Angular velocity: Use tanh scaled to desired range in deg/s.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Separate heads for linear and angular velocities:
        self.linear_head = nn.Linear(hidden_dim, 1)
        self.angular_head = nn.Linear(hidden_dim, 1)

        self.apply(init_weights_actor) #Initialize actor network
        
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        linear = 2.5 * torch.nn.functional.softplus(self.linear_head(x))
        #linear = 2.5 * 0.5 * (torch.tanh(self.linear_head(x)) + 1)
        #linear = 2.5 * torch.sigmoid(self.linear_head(x))
        # Angular velocity (both positive and negative), scaled to desired range.
        angular = 1 * torch.tanh(self.angular_head(x)) 
        #print(f'Actor linear {linear} , Actor angular: {angular}')
        # Concatenate actions:
        return torch.cat((angular, linear), dim=-1)  # Returns a single tensor
    
#Critic Network (Q-value prediction)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        input_dim = state_dim + action_dim
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.value_stream = nn.Linear(16, 1)

        self.apply(init_weights_critic) #Initialize critic network

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        #critic_q_value = torch.relu(self.value_stream(x)) #Edited 
        #critic_q_value = self.value_stream(x)
        critic_q_value = torch.nn.functional.softplus(self.value_stream(x))
        return critic_q_value
    
# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = []
        self.capacity = capacity

    def add(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove oldest experience
        self.buffer.append(transition)

    def sample(self, batch_size=64):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
    
#Agent class
class Agent:
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 2

        self.wheel_radius = 0.2  # example: 5cm radius
        self.axle_length = 0.3    # example: distance between wheels is 30cm

        #Neural network initialiation
        self.actor = Actor(self.state_dim, self.action_dim).to(DEVICE)
        self.critic = Critic(self.state_dim, self.action_dim).to(DEVICE)

        self.target_actor = Actor(self.state_dim, self.action_dim).to(DEVICE) #Initialize target actor
        self.target_actor.load_state_dict(self.actor.state_dict()) 

        self.target_critic = Critic(self.state_dim, self.action_dim).to(DEVICE) #Initialize target critic
        self.target_critic.load_state_dict(self.critic.state_dict()) 

        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=5e-4) #(Edited)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)

        self.gamma = 0.99

        self.step = 0

        #Loggging variables
        self.actor_losses_train = []  # Stores actor network loss
        self.actor_losses_negative_train = []  # Stores actor network loss
        self.critic_losses_train = []  # Stores critic network loss
        self.td_errors_train = []  # Tracks TD error values
        self.actual_q_values_train = []  # Stores actual Q-values (TD target)
        self.predicted_q_values_train = []  # Stores critic-predicted Q-values
        self.mean_rewards_per_episode = []  # Stores critic-predicted Q-values
        self.episode_returns = []       # Monte Carlo returns per episode
        self.episode_initial_qs = []      # Q-value predictions for the initial state
        self.episode_td_targets = []      # TD target for the initial state (if desired)
        self.episode_correlations = []    # Correlation between Q-values and rewards per episode

        #Data storage variables
        self.buffer = ReplayBuffer(capacity=50000) #Create replay buffer
        self.episode_transitions = []  # Store transitions for the current episode

    def store_transition(self, state, action, reward, next_state):
        self.episode_transitions.append((state, action, reward, next_state))

    #Actor's code for selecting action
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        agent_action = self.actor(state_tensor).squeeze(0).detach().cpu().numpy()

        omega, v = agent_action
        #print(f'Omega: {omega}, V: {v}')

        left_wheel_velocity = float((v - (self.axle_length / 2.0) * omega) / self.wheel_radius)
        right_wheel_velocity = float((v + (self.axle_length / 2.0) * omega) / self.wheel_radius)

        #print(f'Left Speed before noise: {left_wheel_velocity}')
        #print(f'Right Speed before noise: {right_wheel_velocity}')

        noise_left = np.random.normal(0, 0.5)
        noise_right = np.random.normal(0, 0.5)
        #print(f'Left noise: {noise_left}')

        left_wheel_velocity += noise_left
        right_wheel_velocity += noise_right
        #print(f'Left Speed after noise: {left_wheel_velocity}')
        #print(f'Right Speed after noise: {right_wheel_velocity}')
        #agent_action = agent_action + noise #Add noise for exploration

        return left_wheel_velocity, right_wheel_velocity
    
    #Soft update for the target network
    def soft_update(self, target, source, tau=0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau*param.data + (1.0 - tau) * target_param.data)

    #Saving networks' parameters (weights and biases)
    def save_checkpoint(self, filename="checkpoint.pth", episode=0):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "episode": episode,  # Save the current episode number

            # Random States
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate()
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at episode {episode}.")

    #Loading networks' parameters (weights and biases)
    def load_checkpoint(self, filename="checkpoint.pth"):
        try:
            checkpoint = torch.load(filename)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            

            # Restore random states for reproducibility
            torch.set_rng_state(checkpoint["torch_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            random.setstate(checkpoint["python_rng_state"])
            
            episode = checkpoint["episode"]
            print(f"Checkpoint loaded. Resuming from episode {episode}.")
            return episode
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")
            return 0
        
    #def flat_grad(self, loss, parameters, retain_graph=True):
        # Compute gradients for all parameters and flatten them into a single vector
        #grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
        #return torch.cat([grad.contiguous().view(-1) for grad in grads if grad is not None])

    #Function for updating the networks
    def optimize_model(self):
        if self.buffer.size() < 64:
            return
        
        self.step += 1
        
        #Sample a random minibatch from replay buffer
        transitions = self.buffer.sample(64)
        states, actions, rewards, next_states = zip(*transitions)

        states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)

        #Compute TD Target for each transition in mini batch using target critic
        with torch.no_grad():
            next_action = self.target_actor(next_states)
            next_q_value = self.target_critic(next_states, next_action).view(-1)

        td_target = rewards + self.gamma * next_q_value #Todo this online. calculate td error each step

        #Critic Operation
        q_value = self.critic(states, actions).view(-1) #Compute the q-value using the Critic
        critic_loss = nn.MSELoss()(q_value, td_target) #Compute the Critic Loss using MSE of target and prediction

        #if self.step == 10:  # Only every 10 steps (Edited)
        self.critic_optimizer.zero_grad() #Clear previous gradients
        critic_loss.backward() #Compute gradients of the Critic loss using backpropagation
        track_gradients(self.critic, critic_gradients_plots) #Track the Critic's gradients for plotting
        self.critic_optimizer.step() #Updates the Critic network's weights using the computed gradient
        self.step = 0
    
        #Actor Operation
        predicted_action = self.actor(states) #Compute the action using the current policy
        actor_loss = (-self.critic(states, predicted_action)).mean() #Compute the actor loss function (-Q(s,a)) (Edited)

        # Compute the flat gradient Edited
        actor_params = list(self.actor.parameters())
        #flat_gradients = self.flat_grad(actor_loss, actor_params)

        actor_loss_negative = -actor_loss 

        track_gradients(self.actor, actor_gradients_plots) #Track the Actor's gradients for plotting

        self.actor_optimizer.zero_grad() #Clear previous gradients

        # Compute gradients normally (to populate each parameter's .grad)
        #grads = torch.autograd.grad(actor_loss, actor_params, retain_graph=True)
        #for param, grad in zip(actor_params, grads):
        #    param.grad = grad

        actor_loss.backward() #Compute gradients of the Actor loss using backpropagation
        self.actor_optimizer.step() #Update the Actor's weights using the computed gradient

        #Store losses and metrics for logging
        self.critic_losses_train.append(critic_loss.item())
        self.actor_losses_train.append(actor_loss.item())
        self.actor_losses_negative_train.append(actor_loss_negative.item())
        self.td_errors_train.append((td_target - q_value).mean().detach().item())
        self.actual_q_values_train.append(td_target.mean().item())
        self.predicted_q_values_train.append(q_value.mean().item())
        #print(f'Printed: {q_value.item()}')
        self.mean_rewards_per_episode.append(rewards.mean().item())

        #Soft update target critic and actor
        self.soft_update(self.target_critic, self.critic)
        self.soft_update(self.target_actor, self.actor)


    def clear_transition(self):
         self.episode_transitions = []  #Clear transitions


# ----------------------------------------------------------------------------------------------
# Extra functions
# ----------------------------------------------------------------------------------------------
def track_gradients(model, gradient_list):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(2)  # L2 norm of gradients
            total_norm += param_norm.item() ** 2
    gradient_list.append(total_norm ** 0.5)  # Store the total L2 norm

# Function to save data to CSV
def save_data_to_csv(data, filename, header):
    df = pd.DataFrame(data, columns=header)
    df.to_csv(filename, index=False)
    #print(f"Saved {filename}")


# Function to save plots
def save_plot(fig, filename):
    fig.savefig(filename)
    print(f"Saved {filename}")

# ----------------------------------------------------------------------------------------------
# MAIN TRAINING LOOP
# ----------------------------------------------------------------------------------------------

def main():

    agent = Agent()

    #Load previous model if available
    #agent.load_checkpoint()

    num_episodes = 350
    reward_per_episode = []
    action_log = []
    all_trajectories = []
    all_heading_errors = [] # Global log of heading error per episode
    all_cumulative_rewards = []  # Store cumulative rewards for all episodes
    all_cumulative_q_value_per_episode = []
    all_q_value_per_episode = []
    all_cumulative_tdtarget_per_episode = []
    all__tdtarget_per_episode = []
    tracked_episodes = list(range(0, num_episodes, 10))

    actor_loss_per_episode = {ep: [] for ep in tracked_episodes}
    critic_loss_per_episode = {ep: [] for ep in tracked_episodes}
    q_value_per_tracked_episode = {ep: [] for ep in tracked_episodes}
    td_error_per_episode = {ep: [] for ep in tracked_episodes}

    for episode in range(num_episodes):
        sim.startSimulation()
        start_time = sim.getSimulationTime()
        cumulative_rewards = []  # Reset for each episode
        cumulative_q_values = []
        q_values_per_episode = []
        cumulative_tdtarget = []
        td_target_per_epiosde = []
        total_reward = 0
        total_q_value = 0
        total_tdtarget = 0

        trajectory = []      # Log positions per episode
        heading_errors = []  # Log heading errors per episode

        #Get initial states
        chassis_position = sim.getObjectPosition(chassisHandle, sim.handle_world)
        target_position = sim.getObjectPosition(targetHandle, sim.handle_world)
        chassis_orientation = sim.getObjectOrientation(chassisHandle, sim.handle_world)
        robot_heading = chassis_orientation[2]

        x, y = chassis_position[:2]
        state = [x, y, robot_heading]
        prev_distance = math.sqrt((target_position[0] - x) ** 2 + (target_position[1] - y) ** 2)

        step = 0  # Initialize step counter

        while True:
            #Take an action
            actions = agent.select_action(state)
            sim.setJointTargetVelocity(leftJointHandle, float(actions[0]))
            sim.setJointTargetVelocity(rightJointHandle, float(actions[1]))

            step += 1

            #if step % 3 == 0:  # Only step every 3rd iteration
            #Take a simulation step
            sim.step()

            #step = 0  # Initialize step counter

            #Obtain the new states after the action has been taken s_t+1
            x_next, y_next, _ = sim.getObjectPosition(chassisHandle, sim.handle_world)
            theta_next = sim.getObjectOrientation(chassisHandle, sim.handle_world)
            next_state = [x_next, y_next, theta_next[2]]
            next_state_np = np.array([x_next, y_next, theta_next[2]])
            trajectory.append(next_state_np[:2])

            new_distance = math.sqrt((target_position[0] - x_next) ** 2 + (target_position[1] - y_next) ** 2)

            #Compute the reward
            reward = (1/(new_distance+ 1e-6))*10
            # reward = -new_distance

            target_angle = math.atan2(target_position[1] - y_next, target_position[0] - x_next)
            angle_difference = abs(target_angle - theta_next[2])
            angle_difference = min(angle_difference, 2 * math.pi - angle_difference) # Normalize angle difference between 0 and π  

            heading_errors.append(angle_difference)

            state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)
            actions_tensor = torch.tensor(actions, dtype=torch.float32).to(DEVICE)

            # Ensure batch dimension
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)  
            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(0)  
            if next_state_tensor.dim() == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)  

            #Compute td-target and q-value for mointoring
            with torch.no_grad():
                next_action = agent.target_actor(next_state_tensor)

                q_value_monitor = agent.critic(state_tensor, actions_tensor)  # Get Q-values
                total_q_value += q_value_monitor.cpu().numpy().item()
                #cumulative_q_values.append(q_value_monitor.cpu().numpy().item())  # Store for later analysis
                cumulative_q_values.append(total_q_value)  # Store for later analysis
                q_values_per_episode.append(q_value_monitor.cpu().numpy().item())

                next_q_value_monitor = agent.target_critic(next_state_tensor, next_action) # Get Q-values
                td_target_monitor = reward + agent.gamma * next_q_value_monitor
                total_tdtarget += td_target_monitor.cpu().numpy().item()
                #cumulative_tdtarget.append(td_target_monitor.cpu().numpy().item())  # Store for later analysis
                cumulative_tdtarget.append(total_tdtarget)  # Store for later analysis
                td_target_per_epiosde.append(td_target_monitor.cpu().numpy().item())


            # Encourage the robot to face the target
            #reward += (1 - (angle_difference / math.pi)) * 5  # Higher reward for facing the target (Edited)

            # Encourage forward movement
            #forward_motion = (x_next - x) * math.cos(theta_next[2]) + (y_next - y) * math.sin(theta_next[2])
            #reward += max(forward_motion, 0) * 10  # Reward only positive forward movement (Edited)

            # Extra incentive: Boost reward if within a small radius of the target
            if new_distance < 1.0:
                reward += 100  # Extra boost when getting very close
            
            #Check if robot reached the target
            #result, _ = sim.checkCollision(chassisHandle, targetHandle)
            #if result:
            #    reward+=100

            #Add the experience to the replay buffer and transitions storage
            agent.buffer.add((state, actions, reward, next_state)) #Store transition to replay buffer
            agent.store_transition(state, actions, reward, next_state)

            #if step % 3 == 0:  # Only step every 3rd iteration
                #Update the networks online (Edited)
            agent.optimize_model()

            agent.clear_transition() #Edited

            total_reward += reward
            cumulative_rewards.append(total_reward)
            action_log.append(actions)

            state = next_state

            #Track parameters 'tracked_episodes'
            if (episode + 1) in tracked_episodes:
                td_error_per_episode[episode + 1].append(agent.td_errors_train[-1])
                q_value_per_tracked_episode[episode + 1].append(agent.predicted_q_values_train[-1])
                if len(agent.actor_losses_train) > 0:
                    actor_loss_per_episode[episode + 1].append(agent.actor_losses_train[-1])
                if len(agent.critic_losses_train) > 0:
                    critic_loss_per_episode[episode + 1].append(agent.critic_losses_train[-1])

            #Check if simulation completion conditions have been met
            #if new_distance > 100:
            #    print(f"Episode {episode + 1} ended due to being too far with reward {total_reward}.")
            #    sim.stopSimulation()
            #    while sim.getSimulationState() != sim.simulation_stopped:
            #        time.sleep(0.5)  # Ensure simulation fully stops before restarting
            #    break

            if sim.getSimulationTime() - start_time > max_sim_time_s:
            #if step >= max_steps:
                #print(f"Episode {episode + 1} ended due to time limit with reward {total_reward}.")
                print(f"Episode {episode + 1} ended due to max steps of {step} with reward {total_reward}.")
                sim.stopSimulation()
                while sim.getSimulationState() != sim.simulation_stopped:
                    time.sleep(0.5)  # Ensure simulation fully stops before restarting
                break

            #if result:
            #    print(f"Episode {episode + 1} ended due to collision. with reward {total_reward}")
            #    sim.stopSimulation()
            #    while sim.getSimulationState() != sim.simulation_stopped:
            #        time.sleep(0.5)  # Ensure simulation fully stops before restarting
            #    break
            
            #End of this step

        
        #Update the network at the end of the episode
        #agent.optimize_model()
        reward_per_episode.append(total_reward)
        all_trajectories.append(np.array(trajectory))
        all_heading_errors.append(np.array(heading_errors))
        agent.mean_rewards_per_episode.append(np.mean(reward_per_episode[-100:]))  # Moving
        all_cumulative_rewards.append(cumulative_rewards)  # Store this episode's rewards
        all_cumulative_q_value_per_episode.append(cumulative_q_values)
        all_q_value_per_episode.append(q_values_per_episode)
        all_cumulative_tdtarget_per_episode.append(cumulative_tdtarget)
        all__tdtarget_per_episode.append(td_target_per_epiosde)

        # Compute the final discounted return for the episode
        final_return = cumulative_rewards[-1]
        # Get the predicted Q-value for the initial state (first element of q_values_per_episode)
        predicted_q_initial = q_values_per_episode[0] 
        agent.episode_returns.append(final_return)
        agent.episode_initial_qs.append(predicted_q_initial)

        initial_td_target = td_target_per_epiosde[0]  # if applicable
        agent.episode_td_targets.append(initial_td_target)

        if len(q_values_per_episode) > 1:
            correlation = np.corrcoef(q_values_per_episode, cumulative_rewards)[0, 1]
            agent.episode_correlations.append(correlation)
        else:
            agent.episode_correlations.append(0)  # or np.nan

        time.sleep(2)

        # Save model every 10 episodes
        if episode % 10 == 0 or episode == num_episodes - 1:
            agent.save_checkpoint()        
            # Save numerical data
            save_data_to_csv(reward_per_episode, "reward_per_episode.csv", ["Total Reward"])
            save_data_to_csv(agent.mean_rewards_per_episode, "mean_reward_per_episode.csv", ["Mean Reward"])
            save_data_to_csv(zip(agent.actor_losses_train, agent.critic_losses_train), "losses.csv", ["Actor Loss", "Critic Loss"])
            save_data_to_csv(zip(agent.actual_q_values_train, agent.predicted_q_values_train), "q_values.csv", ["Actual Q", "Predicted Q"])
            print('Saved files')


    episodes = range(len(agent.episode_returns))
    # plt.figure(figsize=(12, 4))

    # Plot 1: Q-value vs. Return
    # fig17 = plt.figure(figsize=(10, 5))
    # plt.plot(episodes, agent.episode_initial_qs, label='Predicted Q-value')
    # plt.plot(episodes, agent.episode_returns, label='Monte Carlo Return')
    # plt.xlabel('Episode')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.title('Initial Q-value vs Return')
    # save_plot(fig17, "InitialQValueVsReturn.png")

    fig17, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Predicted Q-value", color="orange")
    ax1.plot(agent.episode_initial_qs, label="Predicted Q-Value", color="orange")
    ax1.tick_params(axis="y", labelcolor="orange")

    # Create secondary y-axis for Critic Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel("Monte Carlo Return", color="green")
    ax2.plot(agent.episode_returns, label="Monte Carlo Return", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    save_plot(fig17, "InitialQValueVsReturn.png")

    # # Plot 2: TD Error
    # plt.subplot(1, 3, 2)
    # plt.plot(range(len(agent.episode_td_errors)), agent.episode_td_errors)
    # plt.xlabel('Episode')
    # plt.ylabel('Mean TD Error')
    # plt.title('TD Error Over Episodes')

    # Plot 3: Correlation
    fig18 = plt.figure(figsize=(10, 5))
    plt.plot(episodes, agent.episode_correlations)
    plt.xlabel('Episode')
    plt.ylabel('Correlation')
    plt.title('Correlation Between Q-values and Returns')

    plt.tight_layout()
    plt.show()
    save_plot(fig18, "QValueAndReturnsCorrelation.png")

    # Plot results (Edited)
    fig1 = plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(reward_per_episode, label="Total Reward Per Episode", color="blue")
    plt.title("Total Reward Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    # Plot Actor Loss
    plt.subplot(2, 2, 2)
    plt.plot(agent.actor_losses_train, label="Training Actor Loss", color="orange")
    plt.title("Networks' Training loss for per step")
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Critic Loss
    plt.subplot(2, 2, 2)
    plt.plot(agent.critic_losses_train, label="Critic Loss")
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.legend()

    # Plot TD Error
    plt.subplot(2, 2, 3)
    plt.plot(agent.td_errors_train, label="TD Error")
    plt.title("Training  TD Error per step")
    plt.xlabel("Timestep")
    plt.ylabel("TD Error mean of minibacth")
    plt.legend()

    # Plot Action Values (Q-values)
    plt.subplot(2, 2, 4)
    plt.plot(agent.predicted_q_values_train, label="Predicted Q-Value")
    plt.title("Training Critic's Q-Value")
    plt.xlabel("Episodes")
    plt.ylabel("Q-Value mean of minibatch")
    plt.legend()

    plt.tight_layout()
    plt.show()

    save_plot(fig1, "various.png")

    # Plot Actor Loss on primary y-axis
    fig11, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Actor Loss", color="orange")
    ax1.plot(agent.actor_losses_train, label="Actor Loss", color="orange")
    ax1.tick_params(axis="y", labelcolor="orange")

    # Create secondary y-axis for Critic Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel("Critic Loss", color="green")
    ax2.plot(agent.critic_losses_train, label="Critic Loss", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    fig11.tight_layout()
    plt.title("Networks' Training loss per step")
    plt.show()
    save_plot(fig11, "ActorVsCriticLosses.png")

    # Plot Actual vs. Predicted Q-values
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(agent.actual_q_values_train, label="Actual Q-value (TD Target)", color="blue")
    plt.plot(agent.predicted_q_values_train, label="Predicted Q-value (Critic Output)", color="red", linestyle="dashed")
    plt.xlabel("Timestep")
    plt.ylabel("Q-Value")
    plt.title("Training Actual vs. Predicted Q-values means of Minibatch")
    plt.legend()
    plt.show()
    save_plot(fig2, "TrainingQ_values_ActualVsPredicted.png")

    action_log = np.array(action_log)  # Convert to NumPy array
    fig3 = plt.figure(figsize=(8, 5))
    plt.hist(action_log[:, 0], bins=20, alpha=0.6, label='Action 1')
    plt.hist(action_log[:, 1], bins=20, alpha=0.6, label='Action 2')
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.title(f"Action Distribution - Episode {episode}")
    plt.legend()
    plt.show()
    save_plot(fig3, "action_dist.png")

    # Actor Loss Plots (One for Each Tracked Episode)
    fig4 = plt.figure(figsize=(10, 5))
    for ep, losses in actor_loss_per_episode.items():
        plt.plot(losses, label=f"Episode {ep}")
    plt.title("Training Actor Loss Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.legend()

    # Critic Loss Plots (One for Each Tracked Episode)
    fig5 = plt.figure(figsize=(10, 5))
    for ep, losses in critic_loss_per_episode.items():
        plt.plot(losses, label=f"Episode {ep}")
    plt.title("Training Critic Loss Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
    save_plot(fig4, "TrainingActorLoss.png")
    save_plot(fig5, "TrainingCriticLoss.png")

    # Plot Actual vs. Predicted Q-values
    fig6 = plt.figure(figsize=(10, 5))
    plt.plot(agent.actor_losses_negative_train, label="Negative Actor Loss", color="blue")
    plt.plot(agent.predicted_q_values_train, label="Predicted Q-value (Critic Output)", color="red", linestyle="dashed")
    plt.xlabel("Steps")
    plt.ylabel("Q-Value")
    plt.title("Training Negative Actor loss Vs Predicted Q-value")
    plt.legend()
    plt.show()
    save_plot(fig6, "TrainingActorLossVsQ_Value.png")

    # Plot Predicted Q-values per episode
    fig7 = plt.figure(figsize=(12, 5))
    for ep, value in q_value_per_tracked_episode.items():
        plt.plot(value, label=f"Episode {ep}")
    plt.xlabel("Timestep")
    plt.ylabel("Q-Value")
    plt.title("Training Predicted Q-value")
    plt.legend()
    plt.show()
    save_plot(fig7, "TrainingQ_values.png")

    # Plot Predicted TD Error per episode
    fig8 = plt.figure(figsize=(12, 5))
    for ep, error in td_error_per_episode.items():
        plt.plot(error, label=f"Episode {ep}")
    plt.xlabel("Timestep")
    plt.ylabel("TD Error")
    plt.title("Training TD Error per episode")
    plt.legend()
    plt.show()
    save_plot(fig8, "TrainingtdError.png")

    # Plot trajectory 
    traj = all_trajectories[-1]
    trajectory = np.array(traj)
    fig9 = plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=2, label="Trajectory")
    # Mark the starting point
    plt.plot(trajectory[0, 0], trajectory[0, 1], marker='s', markersize=10, color='green', label='Start')
    # Mark the end point
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], marker='X', markersize=10, color='blue', label='End')
    # Plot the target position if available
    if target_position is not None:
        plt.plot(target_position[0], target_position[1], marker='*', markersize=15, color='red', label='Target')
    # Labels and formatting
    plt.title("Last Episode trajectory")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()
    save_plot(fig9, "LastEpisodeTrajectory.png")

    #Plot heading error
    fig10 = plt.figure(figsize=(10, 5))
    plt.plot(all_heading_errors[-1], color='orange')
    plt.title("Last Episode Heading Error")
    plt.xlabel("Step")
    plt.ylabel("Heading Error (radians)")
    plt.grid(True)
    plt.show()
    save_plot(fig10, "LastEpisodeHeadingError.png")


    # Plot each episode
    fig12 = plt.figure(figsize=(10, 6))
    for episode_idx, rewards in enumerate(all_cumulative_rewards):
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Episode {episode_idx + 1}")

    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Steps for Each Episode")
    plt.legend()
    plt.grid(True)
    # Move the legend outside the plot
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.show()
    save_plot(fig12, "CumulativeRewards.png")

    # Extract last episode's Q-values and TD-targets
    last_cumulative_q_values = all_cumulative_q_value_per_episode[-1]  # Last episode's Q-values
    last_cumulative_td_targets = all_cumulative_tdtarget_per_episode[-1]  # Last episode's TD-targets
    last_cumulative_rewards = all_cumulative_rewards[-1]

    # Create step indices (x-axis)
    steps = list(range(len(last_cumulative_q_values)))  # Steps in the last episode

    # Plot Q-values and TD-targets
    fig13 = plt.figure(figsize=(10, 5))
    plt.plot(steps, last_cumulative_q_values, label="Q-values", marker='o', linestyle='-')
    plt.plot(steps, last_cumulative_td_targets, label="TD-targets", marker='s', linestyle='--')

    # Graph Labels
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Cumulative Q-values and TD-targets Over Time (Last Episode)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    save_plot(fig13, "LastEpisodePredictedVsActual.png")

        # Create step indices (x-axis)
    steps = list(range(len(last_cumulative_q_values)))  # Steps in the last episode

    # Plot Q-values and TD-targets
    fig16 = plt.figure(figsize=(10, 5))
    plt.plot(steps, last_cumulative_q_values, label="Q-values", marker='o', linestyle='-')
    plt.plot(steps, last_cumulative_td_targets, label="TD-targets", marker='s', linestyle='--')
    plt.plot(steps, last_cumulative_rewards, label="Cumulative Reward", color='r', marker='D', linestyle='-')

    # Graph Labels
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Cumulative Q-values, TD-targets, Rewards Over Time (Last Episode)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    save_plot(fig16, "LastEpisodePredictedVsActualVsRewardSameAxes.png")

    # Assuming last_q_values, last_td_targets, and last_cumulative_rewards are already defined
    steps = list(range(len(last_cumulative_q_values)))

    fig14, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Q-values and TD-targets on the first y-axis
    ax1.plot(steps, last_cumulative_q_values, label="Cumulative Q-values", marker='o', linestyle='-')
    ax1.plot(steps, last_cumulative_td_targets, label="Cumulative TD-targets", marker='s', linestyle='--')
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative Q-values & TD-targets", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for cumulative rewards
    ax2 = ax1.twinx()
    ax2.plot(steps, last_cumulative_rewards, label="Cumulative Reward", color='r', marker='D', linestyle='-')
    ax2.set_ylabel("Cumulative Reward", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Cumulative Q-values, Cumulative TD-targets, and Cumulative Rewards Over Time (Last Episode)")
    plt.grid(True)
    plt.show()
    save_plot(fig14, "LastEpisodeCumulativePredictedVsActualVsRewards.png")

    last_q_values = np.array(all_q_value_per_episode[-1])
    last_td_targets = np.array(all__tdtarget_per_episode[-1])
    steps = np.arange(len(last_q_values))  # steps: 0,1,...,299 if there are 300 steps

    fig15 = plt.figure(figsize=(10, 5))
    plt.plot(steps, last_q_values, label="Q-values", marker='o', linestyle='-')
    plt.plot(steps, last_td_targets, label="TD-targets", marker='s', linestyle='--')

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Q-values and TD-targets Over Time (Last Episode)")
    plt.legend()
    plt.grid(True)
    plt.show()
    save_plot(fig15, "LastEpisodePredictedVsActual.png")

    fig16, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Q-values and TD-targets on the first y-axis
    ax1.plot(steps, last_q_values, label="Q-values", marker='o', linestyle='-')
    ax1.plot(steps, last_td_targets, label="TD-targets", marker='s', linestyle='--')
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Q-values & TD-targets", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for cumulative rewards
    ax2 = ax1.twinx()
    ax2.plot(steps, last_cumulative_rewards, label="Cumulative Reward", color='r', marker='D', linestyle='-')
    ax2.set_ylabel("Cumulative Reward", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Q-values, TD-targets, and Cumulative Rewards Over Time (Last Episode)")
    plt.grid(True)
    plt.show()
    save_plot(fig16, "LastEpisodePredictedVsActualVsRewards.png")


    
            
if __name__ == "__main__":
    main()
