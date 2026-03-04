# vec_env_wrapper.py
"""
VHAS Vectorized Environment Wrapper for RSL-RL.

This module provides a concrete implementation of rsl_rl.env.VecEnv that wraps
one or more VHAS_GNN_Wrapper environments, managing parallel execution and
batching observations into TensorDict format required by RSL-RL.
"""

import torch
from tensordict import TensorDict
from typing import Callable
import sys
import os
from collections import deque
import numpy as np

# Add rsl_rl to path if needed
sys.path.insert(0, os.path.abspath('../../rsl_rl'))

from rsl_rl.env import VecEnv


class VHAS_VecEnv(VecEnv):
    """
    A custom VecEnv wrapper for VHAS clinical workflow environments.
    
    This class:
    1. Manages multiple VHAS_GNN_Wrapper instances in parallel (or serial)
    2. Batches their observations into TensorDict format
    3. Implements the RSL-RL VecEnv interface
    4. Handles automatic resets when episodes terminate
    
    Key Features:
    - Compatible with RSL-RL's OnPolicyRunner
    - Supports action masking via info dict
    - Batches graph-structured observations (padded HeteroData → TensorDict)
    - Tracks episode lengths and timeouts
    
    Args:
        env_fn: Callable that creates a VHAS_GNN_Wrapper instance
        num_envs: Number of parallel environments
        device: Device for tensor operations ('cuda' or 'cpu')
        max_episode_length: Maximum episode length (default: 20)
    """
    
    def __init__(
        self,
        env_fn: Callable,
        num_envs: int,
        device: str,
        max_episode_length: int = 20
    ):
        print(f"\n🏗️  Initializing VHAS_VecEnv...")
        print(f"   • Number of environments: {num_envs}")
        print(f"   • Device: {device}")
        
        # Core attributes required by RSL-RL
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_episode_length = max_episode_length
        
        # Create parallel environments
        print(f"   • Creating {num_envs} environment instances...")
        self.envs = [env_fn() for _ in range(num_envs)]
        
        # Extract metadata from first environment (all should be identical)
        env_sample = self.envs[0]
        self.num_actions = env_sample.action_space.n
        self.observation_space = env_sample.observation_space
        self.action_space = env_sample.action_space
        
        # Episode tracking buffer (required by RSL-RL)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Episode reward tracking (for logging)
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # Episode statistics tracking (for periodic logging)
        self.episode_rewards = deque(maxlen=1000)  # Keep last 1000 episodes
        self.episode_lengths = deque(maxlen=1000)
        self.episode_successes = deque(maxlen=1000)  # Track success (terminal reward > 0)
        self.total_episodes = 0
        
        # Configuration dict (can be populated with env-specific config)
        self.cfg = {
            'num_envs': num_envs,
            'max_episode_length': max_episode_length,
            'num_actions': self.num_actions
        }
        
        # Initialize observation buffer
        self.obs_buf = self._create_obs_buffer()
        
        # Action mask buffer (for masked PPO)
        self.action_mask_buf = torch.ones(
            (self.num_envs, self.num_actions),
            dtype=torch.uint8,
            device=self.device
        )
        
        # Reset all environments to populate initial observations
        print(f"   • Resetting all environments...")
        self.reset()
        
        print(f"   ✓ VHAS_VecEnv initialized successfully")
        print(f"   • Observation keys: {list(self.obs_buf.keys())}")
        print(f"   • Action space: Discrete({self.num_actions})")
    
    def _print_episode_stats(self):
        """Print episode statistics every 100 episodes."""
        if len(self.episode_rewards) < 100:
            return
        
        # Calculate statistics from last 100 episodes
        last_100_rewards = list(self.episode_rewards)[-100:]
        last_100_lengths = list(self.episode_lengths)[-100:]
        last_100_successes = list(self.episode_successes)[-100:]
        
        avg_reward = np.mean(last_100_rewards)
        avg_length = np.mean(last_100_lengths)
        success_rate = np.mean(last_100_successes) * 100  # Convert to percentage
        
        # Print formatted stats
        print(f"\n{'='*70}")
        print(f"Episode {self.total_episodes}")
        print(f"  Avg Reward (last 100): {avg_reward:.2f}")
        print(f"  Avg Length (last 100): {avg_length:.1f}")
        print(f"  Success Rate (last 100): {success_rate:.1f}%")
        print(f"{'='*70}\n")
    
    def _create_obs_buffer(self) -> TensorDict:
        """
        Create a batched observation buffer based on observation_space.
        
        The buffer will have shape [num_envs, ...original_shape] for each key.
        This is populated during reset() and step().
        
        Returns:
            TensorDict: Batched observation buffer
        """
        obs_dict = {}
        
        for key, space in self.observation_space.spaces.items():
            # Determine dtype based on space
            if 'mask' in key or key == 'action_mask':
                dtype = torch.uint8
            elif 'index' in key:
                dtype = torch.long
            else:
                dtype = torch.float32
            
            # Create batched tensor: [num_envs, ...space.shape]
            obs_dict[key] = torch.zeros(
                (self.num_envs, *space.shape),
                dtype=dtype,
                device=self.device
            )
        
        return TensorDict(obs_dict, batch_size=[self.num_envs])
    
    def get_observations(self) -> TensorDict:
        """
        Return the current observation buffer.
        
        This method is called by RSL-RL's OnPolicyRunner to get observations
        before calling act() on the policy.
        
        Returns:
            TensorDict: Current observations for all environments
        """
        return self.obs_buf
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """
        Execute one step in all environments.
        
        Args:
            actions: Tensor of shape [num_envs] or [num_envs, 1] containing action indices
        
        Returns:
            observations: TensorDict of batched observations
            rewards: Tensor of shape [num_envs] containing rewards
            dones: Tensor of shape [num_envs] containing done flags
            extras: Dict containing additional info (time_outs, logs, etc.)
        """
        # Ensure actions are in correct shape
        # RSL-RL stores actions with shape [num_envs, action_dim].
        # Với GNN policy rời rạc, ta trả về one-hot [num_envs, num_actions].
        # Ở đây ta convert về chỉ số action [num_envs] trước khi gọi env.step().
        if actions.dim() == 2 and actions.shape[1] == self.num_actions:
            # One-hot or logits -> indices
            actions_indices = actions.argmax(dim=-1)
        else:
            # Scalar actions (e.g., [num_envs] hoặc [num_envs, 1])
            actions_indices = actions.squeeze(-1)
        
        # Convert actions to CPU numpy for Gymnasium compatibility
        actions_cpu = actions_indices.cpu().numpy()
        
        # Storage for step results
        rewards_list = []
        dones_list = []
        infos_list = []
        
        # Step each environment
        for i in range(self.num_envs):
            obs, reward, done, truncated, info = self.envs[i].step(int(actions_cpu[i]))
            
            # Update observation buffer
            # obs is a TensorDict from VHAS_GNN_Wrapper
            for key in self.obs_buf.keys():
                if key in obs:
                    self.obs_buf[key][i] = obs[key].to(self.device)
            
            # Store action mask if available
            if 'action_mask' in info:
                self.action_mask_buf[i] = torch.from_numpy(info['action_mask']).to(self.device)
            
            rewards_list.append(reward)
            dones_list.append(done or truncated)
            infos_list.append(info)
            
            # Track episode reward
            self.episode_reward_buf[i] += reward
            
            # Increment episode length
            self.episode_length_buf[i] += 1
            
            # Auto-reset if episode finished
            if done or truncated:
                # Record episode statistics
                episode_reward = self.episode_reward_buf[i].item()
                episode_length = self.episode_length_buf[i].item()
                
                # Check if episode was successful (terminal reward > 0)
                # Success is indicated by positive terminal reward in info
                is_success = False
                if 'rewards' in info and 'terminal' in info['rewards']:
                    is_success = info['rewards']['terminal'] > 0
                elif 'terminal_reward' in info:
                    is_success = info['terminal_reward'] > 0
                
                # Store episode stats
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_successes.append(1 if is_success else 0)
                self.total_episodes += 1
                
                # Print episode stats every 100 episodes
                if self.total_episodes % 100 == 0:
                    self._print_episode_stats()
                
                # Store terminal observation info before reset (use saved values)
                terminal_info = {
                    'terminal_observation': obs,
                    'episode_length': episode_length,  # Use saved value, not from buffer
                    'episode_reward': episode_reward   # Use saved value, not from buffer
                }
                
                # Reset episode buffers AFTER storing info
                self.episode_reward_buf[i] = 0.0
                self.episode_length_buf[i] = 0
                
                # Reset environment
                new_obs, new_info = self.envs[i].reset()
                
                # Update buffer with reset observation
                for key in self.obs_buf.keys():
                    if key in new_obs:
                        self.obs_buf[key][i] = new_obs[key].to(self.device)
                
                # Update action mask from reset
                if 'action_mask' in new_info:
                    self.action_mask_buf[i] = torch.from_numpy(new_info['action_mask']).to(self.device)
                
                # Reset episode length
                self.episode_length_buf[i] = 0
                
                # Add terminal info to info dict
                infos_list[i]['terminal_info'] = terminal_info
        
        # Convert lists to tensors
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones_list, dtype=torch.bool, device=self.device)
        
        # Build extras dict for RSL-RL
        extras = self._build_extras_dict(dones_tensor, infos_list)
        
        return self.obs_buf, rewards_tensor, dones_tensor, extras
    
    def _build_extras_dict(self, dones: torch.Tensor, infos: list) -> dict:
        """
        Build the extras dictionary expected by RSL-RL.
        
        Key extras:
        - time_outs: Indicates which episodes ended due to time limits (not terminal states)
        - log: Additional logging information
        
        Args:
            dones: Boolean tensor of done flags
            infos: List of info dicts from each environment
        
        Returns:
            dict: Extras dictionary for RSL-RL
        """
        extras = {}
        
        # Time-outs: episodes that ended due to max_episode_length (not true terminal)
        # In our case, if episode_length >= max_episode_length, it's a timeout.
        # RSL-RL expects `time_outs` to be shape [num_envs], because PPO will
        # call `extras["time_outs"].unsqueeze(1)` internally.
        time_outs = self.episode_length_buf >= self.max_episode_length
        extras["time_outs"] = time_outs.float()
        
        # Logging information
        log_dict = {}
        
        # Aggregate episode statistics
        episode_lengths = []
        episode_rewards = []
        
        for info in infos:
            if 'terminal_info' in info:
                terminal_info = info['terminal_info']
                episode_lengths.append(terminal_info.get('episode_length', 0))
                episode_rewards.append(terminal_info.get('episode_reward', 0))
        
        if episode_lengths:
            log_dict['/episode/mean_length'] = sum(episode_lengths) / len(episode_lengths)
            log_dict['/episode/mean_reward'] = sum(episode_rewards) / len(episode_rewards)
        
        # Add reward breakdown if available
        for i, info in enumerate(infos):
            if 'rewards' in info:
                for reward_type, value in info['rewards'].items():
                    key = f'/rewards/{reward_type}'
                    if key not in log_dict:
                        log_dict[key] = []
                    log_dict[key].append(value)
        
        # Average reward components
        for key in list(log_dict.keys()):
            if key.startswith('/rewards/'):
                values = log_dict[key]
                log_dict[key] = sum(values) / len(values) if values else 0.0
        
        extras['log'] = log_dict
        
        return extras
    
    def reset(self) -> TensorDict:
        """
        Reset all environments.
        
        This is called at the start of training and can be called manually.
        
        Returns:
            TensorDict: Initial observations for all environments
        """
        for i in range(self.num_envs):
            obs, info = self.envs[i].reset()
            
            # Update observation buffer
            for key in self.obs_buf.keys():
                if key in obs:
                    self.obs_buf[key][i] = obs[key].to(self.device)
            
            # Update action mask
            if 'action_mask' in info:
                self.action_mask_buf[i] = torch.from_numpy(info['action_mask']).to(self.device)
            
            # Reset episode tracking buffers
            self.episode_length_buf[i] = 0
            self.episode_reward_buf[i] = 0.0
        
        return self.obs_buf
    
    def close(self):
        """Close all environments and cleanup resources."""
        for env in self.envs:
            env.close()
    
    def __del__(self):
        """Destructor to ensure environments are closed."""
        try:
            self.close()
        except:
            pass
