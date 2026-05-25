# train_mlp.py
"""
Training policy trainer with an MLP baseline using Stable-Baselines3.

STRATEGY:
- Use SB3's MlpPolicy (simple but effective)
- Integrate the routing mechanism via a callback
- Use action masking to restrict the policy to approved candidates
- Lightweight, fast to run, and a strong baseline
"""

import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch

# Add project paths
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'simulation')

from ai.gro.simulation.env import GuidelineCompliantEnv
from guidance import GuidanceMechanism


def _resolve_stage2_policies_root() -> Path:
    """Return the stage-2 policy artifact root for Modal or local execution."""
    modal_root = Path("/models")
    if modal_root.exists():
        return modal_root / "stage2_gro_policies"
    return Path(__file__).resolve().parents[1] / "models" / "stage2_gro_policies"


def _resolve_default_model_artifact_path(filename: str) -> Path:
    """Resolve the exact artifact path for a staged policy checkpoint."""
    return _resolve_stage2_policies_root() / filename


class GuidanceAndMaskingCallback(BaseCallback):
    """
    Callback that integrates the routing mechanism with action masking for MaskablePPO.

    Per-step workflow:
    1. Retrieve the current state text from the environment
    2. Call the routing mechanism to propose top-k candidates
    3. Store the action mask for the ActionMasker wrapper to consume
    """
    def __init__(self, guidance_mechanism: GuidanceMechanism, top_k: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.guidance = guidance_mechanism
        self.top_k = top_k
        
        # Performance tracking
        self.total_steps = 0
        self.guidance_calls = 0
        
        # Store the current mask for ActionMasker
        self.current_mask = None
        
    def _on_step(self) -> bool:
        """Called at each environment step."""
        try:
            # Read the current state text from the environment (unwrap from DummyVecEnv)
            current_state_text = self.training_env.get_attr('current_state_text')[0]
            
            if current_state_text is None:
                # Fallback: no masking
                if self.verbose > 0:
                    print("Warning: No state text available, skipping routing")
                return True
            
            # Get candidate action names from the routing mechanism
            candidate_names = self.guidance.propose_actions(
                current_state_text, 
                top_k=self.top_k
            )
            
            # Get the environment's action mapping (unwrap Monitor/TimeLimit wrappers)
            env = self.training_env.envs[0]
            # Unwrap Monitor to get the base environment
            from gymnasium.wrappers import TimeLimit
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            name_to_action = base_env.name_to_action
            action_space_n = base_env.action_space.n
            
            # Create boolean action mask for MaskablePPO
            mask = np.zeros(action_space_n, dtype=bool)
            valid_candidates = []
            for name in candidate_names:
                if name in name_to_action:
                    action_idx = name_to_action[name]
                    mask[action_idx] = True
                    valid_candidates.append(name)
            
            # This is a design decision — SummaryAgent is required for completion.
            if 'SummaryAgent' in name_to_action:
                summary_idx = name_to_action['SummaryAgent']
                if not mask[summary_idx]:
                    mask[summary_idx] = True
                    valid_candidates.append('SummaryAgent')
            
            # Ensure there is at least one valid action.
            if not mask.any():
                if self.verbose > 0:
                    print(f"⚠️  WARNING: No valid candidates from {candidate_names}, allowing all actions")
                mask[:] = True
            
            # Store the mask for the ActionMasker wrapper to retrieve
            self.current_mask = mask
            
            self.guidance_calls += 1
            self.total_steps += 1
            
            # Log periodically.
            if self.verbose > 0 and self.total_steps % 1000 == 0:
                print(f"Routing callback: {self.guidance_calls} calls, {self.total_steps} steps")
            
        except Exception as e:
            if self.verbose > 0:
                print(f"❌ Error in routing callback: {e}")
                import traceback
                traceback.print_exc()
            # Do not stop training on callback errors.
            return True
        
        return True


class MetricsCallback(BaseCallback):
    """Callback to log custom metrics for rewards and success rate."""
    def __init__(self, log_dir: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        self.log_dir = log_dir
        
        # Create a log file if log_dir is provided.
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.log_dir, 'training_log.txt')
            with open(self.log_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("TRAINING LOG\n")
                f.write("="*80 + "\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
        
    def _on_step(self) -> bool:
        # Check for episode termination.
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            # Episode ended.
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                
                    # Track episode reward and length.
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episode_count += 1
                    
                    # Determine success using terminal reward from the last step's info.
                    # Success is indicated by a positive terminal reward (R_term > 0).
                    if 'rewards' in info and info['rewards'].get('terminal', 0) > 0:
                        self.success_count += 1
                    
                    # Log every 100 episodes.
                    if self.episode_count % 100 == 0:
                        avg_reward = np.mean(self.episode_rewards[-100:])
                        avg_length = np.mean(self.episode_lengths[-100:])
                        success_rate = self.success_count / 100
                        
                        log_msg = f"\n{'='*70}\n"
                        log_msg += f"Episode {self.episode_count}\n"
                        log_msg += f"  Avg Reward (last 100): {avg_reward:.2f}\n"
                        log_msg += f"  Avg Length (last 100): {avg_length:.1f}\n"
                        log_msg += f"  Success Rate (last 100): {success_rate:.1%}\n"
                        log_msg += f"{'='*70}\n"
                        
                        print(log_msg)
                        
                        # Write to log file.
                        if self.log_dir:
                            with open(self.log_file, 'a') as f:
                                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ")
                                f.write(f"Episode {self.episode_count} | ")
                                f.write(f"Reward: {avg_reward:.2f} | ")
                                f.write(f"Length: {avg_length:.1f} | ")
                                f.write(f"Success: {success_rate:.1%}\n")
                        
                        # Log to TensorBoard.
                        self.logger.record('rollout/ep_reward_mean_100', avg_reward)
                        self.logger.record('rollout/ep_len_mean_100', avg_length)
                        self.logger.record('rollout/success_rate_100', success_rate)
                        
                        # Reset the success counter after logging.
                        self.success_count = 0
        
        return True


def train_orchestrator_baseline(
    encoder_path: str = 'all-mpnet-base-v2',
    embedding_space_path: str = None,
    action_encoder_path: str = None,  
    use_dual_encoder: bool = False,   
    traces_filepath: str = 'ai/data/workflow_traces/train_traces.json',
    kb_path: str = 'data/simulation_kb.json',
    output_dir: str = 'output/orchestrator_mlp_guided',
    final_model_path: str | Path | None = None,
    total_timesteps: int = 1000000,  # Increase to 1M for better convergence
    learning_rate: float = 1e-4,     # Lower LR for stability
    n_steps: int = 4096,             # Increase to collect more experience per rollout
    batch_size: int = 128,           # Increase for more stable updates
    n_epochs: int = 20,              # More epochs per update
    gamma: float = 0.95,             # Lower to focus more on immediate reward
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    top_k_guidance: int = 5,
    device: str = 'auto'
):
    """Train the policy with an MLP baseline and routing support.

    Args:
        encoder_path: Path to finetuned encoder model (StateEncoder if dual-encoder)
        embedding_space_path: Path to embedding space for routing
        action_encoder_path: Path to ActionEncoder (for dual-encoder only)
        use_dual_encoder: Whether to use dual-encoder architecture
        traces_filepath: Path to the unified trace file
        kb_path: Path to the reference knowledge store
        output_dir: Directory to save the trained model
        total_timesteps: Total training timesteps
        learning_rate: PPO learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        top_k_guidance: Number of candidates from routing
        device: 'cpu', 'cuda', or 'auto'
    """
    
    print("=" * 80)
    print(f"🚀 Training VHAS Orchestrator - MLP + {'Dual-' if use_dual_encoder else ''}Routing Baseline")
    print("=" * 80)
    
    # --- 1. Load components ---
    print("\n📥 Loading components...")
    
    # Load encoder (StateEncoder if dual-encoder)
    encoder_type = "StateEncoder" if use_dual_encoder else "Encoder"
    print(f"   Loading {encoder_type} from: {encoder_path}")
    encoder = SentenceTransformer(encoder_path, device=device if device != 'auto' else 'cpu')
    print(f"   ✓ {encoder_type} loaded: {encoder.get_sentence_embedding_dimension()} dims")
    
    # Initialize routing support (optional)
    guidance = None
    guidance_callback = None
    
    if embedding_space_path and os.path.exists(embedding_space_path):
        try:
            if use_dual_encoder:
                # Use dual-encoder routing.
                from guidance_dual import DualGuidanceMechanism
                print(f"   Loading dual-encoder routing model from: {embedding_space_path}")
                print(f"   StateEncoder: {encoder_path}")
                print(f"   ActionEncoder: {action_encoder_path}")
                guidance = DualGuidanceMechanism(
                    state_encoder_path=encoder_path,
                    action_encoder_path=action_encoder_path,
                    embedding_space_path=embedding_space_path
                )
                print("   ✓ Dual-encoder routing initialized")
            else:
                # Use single-encoder routing.
                print(f"   Loading single-encoder routing model from: {embedding_space_path}")
                guidance = GuidanceMechanism(
                    encoder_path=encoder_path,
                    embedding_space_path=embedding_space_path
                )
                print("   ✓ Single-encoder routing initialized")
            
            guidance_callback = GuidanceAndMaskingCallback(
                guidance_mechanism=guidance,
                top_k=top_k_guidance,
                verbose=1
            )
        except Exception as e:
            print(f"   ⚠️  Could not load routing support: {e}")
            print("   Continuing without routing support (all actions allowed)")
            import traceback
            traceback.print_exc()
    else:
            print("   ℹ️  No embedding space path provided, training without routing support")
    
    # --- 2. Create environment ---
    print("\n🏗️  Creating environment...")
    
    def mask_fn(env):
        """Mask function for ActionMasker — returns the mask from the callback."""
        if guidance_callback is not None and guidance_callback.current_mask is not None:
            return guidance_callback.current_mask
        # Fallback: allow all actions.
        return np.ones(env.action_space.n, dtype=bool)
    
    def make_env():
        """Factory function that creates and wraps the environment."""
        env = GuidelineCompliantEnv(
            encoder_model=encoder,
            traces_filepath=traces_filepath,
            kb_path=kb_path,
            use_guidance=False,  # Routing handled by the callback.
            use_dual_encoder=use_dual_encoder  # Keep the dual-encoder flag.
        )
        # Wrap with Monitor for episode stats.
        env = Monitor(env)
        # Always wrap with ActionMasker for MaskablePPO.
        env = ActionMasker(env, mask_fn)
        return env
    
    # Wrap the environment in DummyVecEnv (SB3 requirement).
    env = DummyVecEnv([make_env])
    
    # Normalize observations and rewards for more stable training.
    env = VecNormalize(
        env,
        norm_obs=True,          # Normalize observations (already normalized by encoder, but doesn't hurt)
        norm_reward=True,       # Normalize rewards (critical for stable value function)
        clip_obs=10.0,          # Clip normalized obs to [-10, 10]
        clip_reward=10.0,       # Clip normalized rewards to [-10, 10]
        gamma=gamma,            # Use the same gamma as PPO for return normalization
        epsilon=1e-8            # Small constant for numerical stability
    )
    
    print(f"   ✓ Environment created with VecNormalize")
    print(f"   Action space: {env.envs[0].action_space}")
    print(f"   Observation space: {env.envs[0].observation_space}")
    print(f"   Reward normalization: ENABLED (clip_reward={10.0})")
    
    # --- 3. Initialize MaskablePPO model ---
    print("\n🧠 Initializing MaskablePPO with MlpPolicy...")
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=0.01,           # Increase exploration.
        vf_coef=0.5,             # Weight value function loss.
        max_grad_norm=0.5,       # Clip gradients.
        verbose=1,
        tensorboard_log="./ppo_vhas_tensorboard/",
        device=device,
        policy_kwargs={
            'net_arch': [256, 256],  # Two-layer MLP with 256 units each.
            'activation_fn': torch.nn.ReLU
        }
    )
    
    print(f"   ✓ MaskablePPO initialized")
    print(f"   Policy architecture: {model.policy}")
    
    # --- 4. Set up callbacks ---
    print("\n⚙️  Setting up callbacks...")
    
    callbacks = []
    
    # Routing callback (if available).
    if guidance_callback is not None:
        callbacks.append(guidance_callback)
        print(f"   ✓ Routing callback added (top_k={top_k_guidance})")
    
    # Metrics callback with logging.
    metrics_callback = MetricsCallback(log_dir=output_dir, verbose=1)
    callbacks.append(metrics_callback)
    print(f"   ✓ Metrics callback added (logging to {output_dir})")
    
    # Checkpoint callback for periodic saves.
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=output_dir,
        name_prefix='orchestrator_checkpoint'
    )
    callbacks.append(checkpoint_callback)
    print(f"   ✓ Checkpoint callback added (save every 10k steps to {output_dir})")
    
    # --- 5. Training ---
    print("\n" + "=" * 80)
    print("🎮 Starting training")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {model.device}")
    print("=" * 80 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # --- 6. Save final model ---
    print("\n💾 Saving final model...")
    resolved_final_model_path = Path(final_model_path) if final_model_path is not None else _resolve_default_model_artifact_path("vhas_mlp_policy.zip")
    if resolved_final_model_path.suffix != ".zip":
        resolved_final_model_path = resolved_final_model_path.with_suffix(".zip")
    resolved_final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(resolved_final_model_path))
    print(f"   ✓ Model saved to: {resolved_final_model_path}")
    
    # Save VecNormalize statistics (important for inference).
    vec_normalize_path = Path(output_dir) / "vec_normalize.pkl"
    vec_normalize_path.parent.mkdir(parents=True, exist_ok=True)
    env.save(str(vec_normalize_path))
    print(f"   ✓ VecNormalize stats saved to: {vec_normalize_path}")
    
    # Save training stats.
    stats = {
        'total_timesteps': model.num_timesteps,
        'episode_rewards': metrics_callback.episode_rewards,
        'episode_lengths': metrics_callback.episode_lengths,
        'episode_count': metrics_callback.episode_count,
        'reward_mean': env.ret_rms.mean if hasattr(env, 'ret_rms') else None,
        'reward_var': env.ret_rms.var if hasattr(env, 'ret_rms') else None
    }
    
    import json
    stats_path = Path(output_dir) / "training_stats.json"
    with stats_path.open('w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON.
        stats_json = {
            'total_timesteps': int(stats['total_timesteps']),
            'episode_count': int(stats['episode_count']),
            'final_avg_reward': float(np.mean(stats['episode_rewards'][-100:])) if stats['episode_rewards'] else 0,
            'final_avg_length': float(np.mean(stats['episode_lengths'][-100:])) if stats['episode_lengths'] else 0
        }
        json.dump(stats_json, f, indent=2)
    print(f"   ✓ Training stats saved to: {stats_path}")
    
    # --- 7. Summary ---
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    print(f"Total episodes: {metrics_callback.episode_count}")
    print(f"Total timesteps: {model.num_timesteps:,}")
    if metrics_callback.episode_rewards:
        print(f"Final avg reward (last 100): {np.mean(metrics_callback.episode_rewards[-100:]):.2f}")
        print(f"Final avg length (last 100): {np.mean(metrics_callback.episode_lengths[-100:]):.1f}")
    print(f"\nModel saved to: {resolved_final_model_path}")
    print(f"TensorBoard logs: ./ppo_vhas_tensorboard/")
    print("=" * 80)
    
    return model, stats


if __name__ == "__main__":
    # Example usage.
    model, stats = train_orchestrator_baseline(
        encoder_path='all-mpnet-base-v2',  # Use base model for local testing.
        embedding_space_path=None,  # Set to an actual path if available.
        traces_filepath='ai/data/workflow_traces/train_traces.json',
        kb_path='data/simulation_kb.json',
        output_dir='output/orchestrator_mlp_guided',
        total_timesteps=50000,  # Reduced for testing.
        top_k_guidance=5,
        device='auto'
    )
    
    print("\n🎉 Training pipeline completed successfully!")
