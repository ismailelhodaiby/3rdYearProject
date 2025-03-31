import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
import time
import os
import glob

# ----------------------------------------------------------------------------------------------
# Saving and Loading Model
# ----------------------------------------------------------------------------------------------

# Define checkpoint saving frequency
SAVE_INTERVAL = 30000  # Save every 5000 steps (adjust as needed)
CHECKPOINT_PATH = "ddpg_checkpoints"

# Ensure the checkpoint directory exists
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


def save_checkpoint(step, actor, critic, actor_optimizer, critic_optimizer, replay_buffer=None):
    checkpoint = {
        'step': step,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
    }
    if replay_buffer: 
        checkpoint['replay_buffer'] = replay_buffer

    checkpoint_file = os.path.join(CHECKPOINT_PATH, f"checkpoint_step_{step}.pth")
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved at step {step} to {checkpoint_file}")


def load_latest_checkpoint(actor, critic, actor_optimizer, critic_optimizer):
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_PATH, "checkpoint_step_*.pth"))
    if not checkpoint_files:
        print("No checkpoint found, starting from scratch.")
        return 0  # Start from step 0

    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)  # Get latest file
    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, weights_only=False)

    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
    replay_buffer = checkpoint.get('replay_buffer', None)

    return checkpoint['step'], replay_buffer

# ----------------------------------------------------------------------------------------------
# COPPELIA SIM CUSTOM ENVIRONMENT FOR DIFFERENTIAL DRIVE ROBOT
# ----------------------------------------------------------------------------------------------

class DifferentialDriveEnv(gym.Env):
    """
    A custom Gym environment wrapping CoppeliaSim for a differential drive robot.
    Observation: [dx, dy, theta] where
      - dx, dy: difference between target and chassis positions
      - theta: chassis orientation (assumed around the z-axis)
    Action: [v_left, v_right] wheel velocities, normalized to [-1, 1].
    Reward: Negative Euclidean distance to target.
    """
    def __init__(self):
        super(DifferentialDriveEnv, self).__init__()
        # Import and initialize CoppeliaSim remote API client
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        
        self.sim.setStepping(True)
        
        # Initialize joints
        self.rightJointHandle = self.sim.getObject("/joint_right")
        self.leftJointHandle = self.sim.getObject("/joint_left")
        self.sim.setObjectInt32Param(self.rightJointHandle, 2000, 1)
        self.sim.setObjectInt32Param(self.leftJointHandle, 2000, 1)
        
        # Initialize chassis and target
        self.chassisHandle = self.sim.getObject("/Chassis")
        self.targetHandle = self.sim.getObject("/Target")
        
        # Define observation space and action space.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        # Actions: left and right wheel velocities, normalized to [-1, 1]
        self.action_space = spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)

        self.max_distance = 0

    def reset(self):
        # Reset simulation state
        # For now, we assume the simulation is reset externally.
        #reset_out = self.sim.reset() if hasattr(self.sim, "reset") else None
    # Stop the simulation first if it is still running
        self.sim.stopSimulation()

        self.max_distance = 0

        # Wait until the simulation is fully stopped before restarting
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)

        # Start the simulation again
        self.sim.startSimulation()
        
        # Get current positions and orientation
        chassis_pos = self.sim.getObjectPosition(self.chassisHandle, self.sim.handle_world)
        target_pos = self.sim.getObjectPosition(self.targetHandle, self.sim.handle_world)
        orientation = self.sim.getObjectOrientation(self.chassisHandle, self.sim.handle_world)
        
        dx = target_pos[0] - chassis_pos[0]
        dy = target_pos[1] - chassis_pos[1]
        theta = orientation[2]  # assuming z-axis rotation
        
        obs = np.array([dx, dy, theta], dtype=np.float32)
        # Return observation and an empty info dict (following Gymnasium API)
        return obs, {}

    def step(self, action):
        # Set wheel velocities based on action
        self.sim.setJointTargetVelocity(self.leftJointHandle, float(action[0]))
        self.sim.setJointTargetVelocity(self.rightJointHandle, float(action[1]))
        
        # Advance the simulation one step
        self.sim.step()  # Advance the simulation in CoppeliaSim environment.
        
        # Get new positions and orientation
        chassis_pos = self.sim.getObjectPosition(self.chassisHandle, self.sim.handle_world)
        target_pos = self.sim.getObjectPosition(self.targetHandle, self.sim.handle_world)
        orientation = self.sim.getObjectOrientation(self.chassisHandle, self.sim.handle_world)
        
        dx = target_pos[0] - chassis_pos[0]
        dy = target_pos[1] - chassis_pos[1]
        theta = orientation[2]
        obs = np.array([dx, dy, theta], dtype=np.float32)
        
        # Compute reward as negative distance to target
        distance = np.sqrt(dx**2 + dy**2)

        #if distance < self.max_distance:
        #    reward = (self.max_distance - distance)
        #else:
        #    self.max_distance = distance
        #    reward = 0
        
        #reward = (10/(distance**2))

        reward = -distance
        
        # Define done when within a threshold
        done = distance < 2
        
        # Return observation, reward, terminated, truncated, and info (Gymnasium style) including Chassis position
        info = {'chassis_pos': chassis_pos}
        return obs, reward, done, False, info

    def close(self):
        self.sim.stopSimulation()

    def render(self, mode='human'):
        
        pass


# ----------------------------------------------------------------------------------------------
# DDPG MODEL (with GPU support and TensorBoard/Matplotlib logging)
# ----------------------------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, act_dim))
        self.net = nn.Sequential(*layers)
        self.act_limit = act_limit

    def forward(self, obs):
        a = self.net(obs)
        normalized_action = torch.tanh(a)  # outputs in [-1, 1]
        return normalized_action


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev_size = obs_dim + act_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        # Ensure obs is extracted if returned as tuple.
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        obs = np.array(obs, dtype=np.float32).flatten()
        next_obs = np.array(next_obs, dtype=np.float32).flatten()
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = np.array(act, dtype=np.float32).flatten()
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        batch = dict(
            obs=self.obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs]
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def ddpg(env_fn,
         actor_hidden_sizes=(256, 256),
         critic_hidden_sizes=(256, 256),
         actor_lr=5e-4,
         critic_lr=5e-4,
         gamma=0.99,
         polyak=0.995,
         replay_size=int(1e6),
         batch_size=100,
         start_steps=1000,
         update_after=100,
         update_every=50,
         max_ep_len=1000,
         epochs=300):
    
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    
    # Set device for GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create networks and move them to device
    actor = Actor(obs_dim, act_dim, actor_hidden_sizes, act_limit).to(device)
    critic = Critic(obs_dim, act_dim, critic_hidden_sizes).to(device)
    actor_target = Actor(obs_dim, act_dim, actor_hidden_sizes, act_limit).to(device)
    critic_target = Critic(obs_dim, act_dim, critic_hidden_sizes).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)
    
    log_dir = "runs/ddpg_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    
    episode_returns = []
    episode_lengths = []
    actor_losses_list = []
    critic_losses_list = []
    update_steps = []
    success_count = 0
    
    # Reset environment and extract observation
    reset_out = env.reset()
    o = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    ep_ret, ep_len = 0, 0
    total_steps = epochs * max_ep_len

    # Additional lists for diagnostics
    aggregated_actor_losses = []
    aggregated_critic_losses = []
    aggregated_update_steps = []
    predicted_q_means = []
    backup_q_means = []
    reward_list = []  # For logging average reward per update

    last_trajectory = []   # This will store the chassis positions for the last episode
    current_trajectory = []  # Reset each episode   

    start_step, loaded_replay_buffer = load_latest_checkpoint(actor, critic, actor_optimizer, critic_optimizer)

    if loaded_replay_buffer is not None:
        replay_buffer = loaded_replay_buffer
    else:
        replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)
    #replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)

    #start_step  = 0

    for t in range(start_step, total_steps):
        # Action selection with exploration noise
        if t > start_steps:
            if isinstance(o, tuple):
                o = o[0]
            o_tensor = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            a = actor(o_tensor).detach().cpu().numpy()[0]
            a += np.random.normal(0, 1, size=act_dim)
            a = np.clip(a, -act_limit, act_limit)
        else:
            a = env.action_space.sample()
        
        # Step the environment
        step_out = env.step(a)
        if len(step_out) == 5:
            raw_next, r, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            raw_next, r, done, info = step_out

        # Extract the chassis position from info and store it
        chassis_pos = info.get('chassis_pos', None)
        if chassis_pos is not None:
            current_trajectory.append(chassis_pos)
                
        next_o = raw_next[0] if isinstance(raw_next, tuple) else raw_next
        ep_ret += r
        ep_len += 1
        
        replay_buffer.store(o, a, r, next_o, done)
        o = next_o
        
        if done:
            writer.add_scalar("Episode Return", ep_ret, t)
            writer.add_scalar("Episode Length", ep_len, t)
            success_count += 1
            print(f"Episode Return: {ep_ret}, Episode Length: {ep_len}, No. of success: {success_count}")
            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            # Save the current episode's trajectory as the last trajectory before resetting
            last_trajectory = current_trajectory.copy()
            current_trajectory = []  # Reset for the next episode
            reset_out = env.reset()
            o = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            ep_ret, ep_len = 0, 0
        
        if t >= update_after and t % update_every == 0:
            batch_pred_q = []
            batch_backup_q = []
            batch_rewards = []
            inner_actor_loss_sum = 0.0
            inner_critic_loss_sum = 0.0
            inner_count = 0
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                obs = batch["obs"].to(device)
                acts = batch["acts"].to(device)
                rews = batch["rews"].to(device)
                next_obs = batch["next_obs"].to(device)
                done_batch = batch["done"].to(device)
                
                with torch.no_grad():
                    a_next = actor_target(next_obs)
                    q_target = critic_target(next_obs, a_next)
                    backup = rews.unsqueeze(1) + gamma * (1 - done_batch.unsqueeze(1)) * q_target
                
                q_val = critic(obs, acts)
                critic_loss = F.mse_loss(q_val, backup)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                actor_loss = -critic(obs, actor(obs)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                writer.add_scalar("Loss/Critic", critic_loss.item(), t)
                writer.add_scalar("Loss/Actor", actor_loss.item(), t)
                
                inner_actor_loss_sum += actor_loss.item()
                inner_critic_loss_sum += critic_loss.item()
                inner_count += 1
                
                # Collect diagnostic statistics for each inner iteration
                batch_pred_q.append(q_val.mean().item())
                batch_backup_q.append(backup.mean().item())
                batch_rewards.append(rews.mean().item())

                
                # Update target networks
                with torch.no_grad():
                    for p, p_target in zip(actor.parameters(), actor_target.parameters()):
                        p_target.data.mul_(polyak)
                        p_target.data.add_((1 - polyak) * p.data)
                    for p, p_target in zip(critic.parameters(), critic_target.parameters()):
                        p_target.data.mul_(polyak)
                        p_target.data.add_((1 - polyak) * p.data)
                
            # Compute averaged losses over the update cycle
            avg_actor_loss = inner_actor_loss_sum / inner_count
            avg_critic_loss = inner_critic_loss_sum / inner_count
            
            # Also log averaged diagnostic values
            avg_pred_q = np.mean(batch_pred_q)
            avg_backup_q = np.mean(batch_backup_q)
            avg_reward = np.mean(batch_rewards)

            # Append aggregated values
            aggregated_update_steps.append(t)
            aggregated_actor_losses.append(avg_actor_loss)
            aggregated_critic_losses.append(avg_critic_loss)

            predicted_q_means.append(avg_pred_q)  # <-- Append avg_pred_q here
            backup_q_means.append(avg_backup_q)   # <-- Append avg_backup_q here
            reward_list.append(avg_reward)          # <-- Append avg_reward here
            
            writer.add_scalar("Diagnostics/Avg Predicted Q", avg_pred_q, t)
            writer.add_scalar("Diagnostics/Avg Backup Q", avg_backup_q, t)
            writer.add_scalar("Diagnostics/Avg Reward", avg_reward, t)
            
            if t % 500 == 0:
                for name, param in actor.named_parameters():
                    writer.add_histogram(f"Actor/{name}_weights", param, t)
                    if param.grad is not None:
                        writer.add_histogram(f"Actor/{name}_grads", param.grad, t)
                for name, param in critic.named_parameters():
                    writer.add_histogram(f"Critic/{name}_weights", param, t)
                    if param.grad is not None:
                        writer.add_histogram(f"Critic/{name}_grads", param.grad, t)

        if t >= update_after and t % update_every == 0 and t % SAVE_INTERVAL == 0 or t == total_steps - 1:  # Save every SAVE_INTERVAL steps
            save_checkpoint(t, actor, critic, actor_optimizer, critic_optimizer, replay_buffer)


    print(f'Training finished with {success_count} successful epsiodes')
    writer.close()
    
    # Matplotlib Plots
    plt.figure(figsize=(10, 4))
    plt.plot(episode_returns)
    plt.title("Episode Returns Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(episode_lengths)
    plt.title("Episode Lengths Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(update_steps, actor_losses_list, label="Actor Loss")
    plt.plot(update_steps, critic_losses_list, label="Critic Loss")
    plt.title("Losses Over Time")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(aggregated_update_steps, aggregated_actor_losses, label="Aggregated Actor Loss")
    plt.plot(aggregated_update_steps, aggregated_critic_losses, label="Aggregated Critic Loss")
    plt.title("Aggregated Losses Over Update Steps")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(update_steps, critic_losses_list, label="Critic Loss")
    plt.title("Critic Losses Over Time")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(update_steps, actor_losses_list, label="Actor Loss")
    plt.title("Actor Losses Over Time")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot Q-value diagnostics
    plt.figure(figsize=(10, 4))
    print("Length of aggregated_update_steps:", len(aggregated_update_steps))
    print("Length of predicted_q_means:", len(predicted_q_means))
    print("Length of update steps:", len(update_steps))

    plt.plot(aggregated_update_steps, predicted_q_means, label="Avg Predicted Q")
    plt.plot(aggregated_update_steps, backup_q_means, label="Avg Backup Q")
    plt.title("Average Q-values Over Update Steps")
    plt.xlabel("Update Step")
    plt.ylabel("Q-value")
    plt.legend()
    plt.show()

    # Plot average reward per update
    plt.figure(figsize=(10, 4))
    plt.plot(aggregated_update_steps, reward_list, label="Avg Reward")
    plt.title("Average Reward Over Update Steps")
    plt.xlabel("Update Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

    # Trajectory Plot for the Last Episode
    if last_trajectory and len(last_trajectory) > 0:
        traj = np.array(last_trajectory)  # Each element is [x, y, z]
        plt.figure(figsize=(8,6))
        plt.plot(traj[:,0], traj[:,1], '-o', label="Trajectory")
        plt.scatter(traj[0,0], traj[0,1], color='green', s=100, label="Start")
        plt.scatter(traj[-1,0], traj[-1,1], color='red', s=100, label="End")
        # Get target position from the simulation (assuming the target stays constant)
        target_pos = env.sim.getObjectPosition(env.targetHandle, env.sim.handle_world)
        plt.scatter(target_pos[0], target_pos[1], color='blue', s=100, label="Target")
        plt.title("Trajectory of the Last Episode")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.show()

# ----------------------------------------------------------------------------------------------
# Main: Run DDPG with Differential Drive Environment
# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Use the custom environment
    env_fn = lambda: DifferentialDriveEnv()
    try:
        ddpg(env_fn)  # Run training loop with DDPG
    finally:
        # Make sure to stop the simulation when done
        env = env_fn()
        env.close()
    #ddpg(env_fn)
