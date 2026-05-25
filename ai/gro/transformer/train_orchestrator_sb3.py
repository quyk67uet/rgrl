# transformer/train_orchestrator_sb3.py
"""
Training orchestrator with a Transformer policy and guidance using Stable-Baselines3.

STRATEGY:
- Use custom TransformerFeaturesExtractor for sequence processing
- Integrate GuidanceMechanism via Callback
- Action masking to restrict the policy to guided candidates
- History-aware decision making
"""

import os
import sys
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch

# Add paths
# Note: When running on Modal, current directory is already /data/gro/transformer
# sys.path.insert(0, 'scripts')
# sys.path.insert(0, 'simulation')
# sys.path.insert(0, 'transformer')

from env import ClinicalWorkflowEnv
from guidance import GuidanceMechanism
from policy_model import TransformerFeaturesExtractor  # NEW: Custom Transformer


class GuidanceAndMaskingCallback(BaseCallback):
    """
    Callback that combines guidance with action masking for MaskablePPO.
    
    Per step:
    1. Read the current state text from the environment
    2. Ask guidance for top-k candidates
    3. Store the mask for ActionMasker
    """
    def __init__(self, guidance_mechanism: GuidanceMechanism, top_k: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.guidance = guidance_mechanism
        self.top_k = top_k
        
        # Stats tracking
        self.total_steps = 0
        self.guidance_calls = 0
        
        # Store current mask for ActionMasker
        self.current_mask = None
        
    def _on_step(self) -> bool:
        """Called at each environment step."""
        try:
            # Read the current state text from the environment.
            current_state_text = self.training_env.get_attr('current_state_text')[0]
            
            if current_state_text is None:
                # Fallback: no masking
                if self.verbose > 0:
                    print("Warning: No current_state_text available, skipping guidance")
                return True
            
            # Get candidates from guidance.
            candidate_names = self.guidance.propose_actions(
                current_state_text, 
                top_k=self.top_k
            )
            
            # Read the action mapping from the base environment.
            env = self.training_env.envs[0]
            from gymnasium.wrappers import TimeLimit
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            name_to_action = base_env.name_to_action
            action_space_n = base_env.action_space.n
            
            # Build the action mask.
            mask = np.zeros(action_space_n, dtype=bool)
            valid_candidates = []
            for name in candidate_names:
                if name in name_to_action:
                    action_idx = name_to_action[name]
                    mask[action_idx] = True
                    valid_candidates.append(name)
            
            # Always include SummaryAgent so episodes can terminate.
            if 'SummaryAgent' in name_to_action:
                summary_idx = name_to_action['SummaryAgent']
                if not mask[summary_idx]:
                    mask[summary_idx] = True
                    valid_candidates.append('SummaryAgent')
            
            # Ensure at least one action remains valid.
            if not mask.any():
                if self.verbose > 0:
                    print(f"Warning: no valid candidates from {candidate_names}; allowing all actions")
                mask[:] = True
            
            # Store the mask for ActionMasker.
            self.current_mask = mask
            
            self.guidance_calls += 1
            self.total_steps += 1
            
            # Log periodically.
            if self.verbose > 0 and self.total_steps % 1000 == 0:
                print(f"Guidance callback: {self.guidance_calls} calls, {self.total_steps} steps")
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error in GuidanceCallback: {e}")
                import traceback
                traceback.print_exc()
            # Keep training running if the callback fails.
            return True
        
        return True


class MetricsCallback(BaseCallback):
    """Callback that logs reward and success metrics."""
    def __init__(self, log_dir: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        self.log_dir = log_dir
        
        # Create the log file if a directory was provided.
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
        # Check for episode end.
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            # Episode ended.
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                
                # Track episode reward.
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episode_count += 1
                    
                    # Success means the terminal reward is positive.
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
                        
                        # Write to the log file.
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
                        
                        # Reset the success counter.
                        self.success_count = 0
        
        return True


def train_orchestrator_baseline(
    encoder_path: str = 'all-mpnet-base-v2',
    embedding_space_path: str = None,
    action_encoder_path: str = None,  # NEW: For dual-encoder
    use_dual_encoder: bool = False,   # NEW: Flag to use dual-encoder
    scenarios_data_dir: str = '../data/scenarios/data',
    kb_path: str = 'data/simulation_kb.json',
    output_dir: str = 'output/vhas_orchestrator_transformer_guided',
    total_timesteps: int = 1000000,  # Increased to 1M for better convergence
    learning_rate: float = 3e-5,     # Optimized for stability
    n_steps: int = 4096,             # Increased to collect more experience
    batch_size: int = 64,            # Optimized for better gradients
    n_epochs: int = 10,              # Reduced to limit overfitting
    gamma: float = 0.99,             # Optimized for long-term planning
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    top_k_guidance: int = 5,
    device: str = 'auto'
):
    """
    Train the orchestrator with Transformer + guidance (v2.0).
    
    Args:
        encoder_path: Path to finetuned encoder model (StateEncoder if dual-encoder)
        embedding_space_path: Path to embedding space for guidance
        action_encoder_path: Path to ActionEncoder (for dual-encoder only)
        use_dual_encoder: Whether to use dual-encoder architecture
        scenarios_data_dir: Directory containing clinical traces
        kb_path: Path to simulation knowledge base
        output_dir: Directory to save trained model
        total_timesteps: Total training timesteps
        learning_rate: PPO learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        top_k_guidance: Number of candidates from guidance
        device: 'cpu', 'cuda', or 'auto'
    """
    
    print("=" * 80)
    print(f"Training VHAS Orchestrator - Transformer + {'Dual-' if use_dual_encoder else ''}Guidance (v2.0)")
    print("=" * 80)
    
    # --- 1. Load components ---
    print("\n📥 Loading components...")
    
    # Load the encoder (StateEncoder for dual-encoder mode).
    encoder_type = "StateEncoder" if use_dual_encoder else "Encoder"
    print(f"   Loading {encoder_type} from: {encoder_path}")
    encoder = SentenceTransformer(encoder_path, device=device if device != 'auto' else 'cpu')
    print(f"   ✓ {encoder_type} loaded: {encoder.get_sentence_embedding_dimension()} dims")
    
    # Initialize guidance (optional).
    guidance = None
    guidance_callback = None
    
    if embedding_space_path and os.path.exists(embedding_space_path):
        try:
            if use_dual_encoder:
                # Use dual-encoder guidance.
                from guidance_dual import DualGuidanceMechanism
                print(f"   Loading Dual-Encoder Guidance Mechanism from: {embedding_space_path}")
                print(f"   StateEncoder: {encoder_path}")
                print(f"   ActionEncoder: {action_encoder_path}")
                guidance = DualGuidanceMechanism(
                    state_encoder_path=encoder_path,
                    action_encoder_path=action_encoder_path,
                    embedding_space_path=embedding_space_path
                )
                print("   ✓ Dual-Encoder Guidance Mechanism initialized")
            else:
                # Use single-encoder guidance (legacy).
                print(f"   Loading Single-Encoder Guidance Mechanism from: {embedding_space_path}")
                guidance = GuidanceMechanism(
                    encoder_path=encoder_path,
                    embedding_space_path=embedding_space_path
                )
                print("   ✓ Single-Encoder Guidance Mechanism initialized")
            
            guidance_callback = GuidanceAndMaskingCallback(
                guidance_mechanism=guidance,
                top_k=top_k_guidance,
                verbose=1
            )
        except Exception as e:
            print(f"Could not load guidance: {e}")
            print("Continuing without guidance; all actions remain allowed.")
            import traceback
            traceback.print_exc()
    else:
        print("No embedding space path provided; training without guidance.")
    
    # --- 2. Create the environment ---
    print("\n🏗️  Creating environment...")
    
    def mask_fn(env):
        """Return the current mask for ActionMasker."""
        if guidance_callback is not None and guidance_callback.current_mask is not None:
            return guidance_callback.current_mask
        # Fallback: allow all actions.
        return np.ones(env.action_space.n, dtype=bool)
    
    def make_env():
        """Create the environment instance."""
        env = ClinicalWorkflowEnv(
            encoder_model=encoder,
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            use_guidance=False,  # Guidance is handled by the callback.
            use_dual_encoder=use_dual_encoder
        )
        # Wrap with Monitor for episode stats.
        env = Monitor(env)
        # Wrap with ActionMasker for MaskablePPO.
        env = ActionMasker(env, mask_fn)
        return env
    
    # Wrap in DummyVecEnv (SB3 requirement).
    env = DummyVecEnv([make_env])
    
    # Normalize observations and rewards for stable training.
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        epsilon=1e-8
    )
    
    print("Environment created with VecNormalize.")
    print(f"   Action space: {env.envs[0].action_space}")
    print(f"   Observation space: {env.envs[0].observation_space}")
    print(f"   Reward normalization: enabled (clip_reward={10.0})")
    
    # --- 3. Initialize MaskablePPO with the Transformer policy ---
    print("\n🧠 Initializing MaskablePPO with Transformer Policy...")
    
    # Use the custom TransformerFeaturesExtractor.
    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            n_heads=4,
            n_layers=2,
            dim_feedforward=512,
            dropout=0.1
        ),
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU
    )
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-5,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_vhas_transformer_tensorboard/",
        device=device
    )
    
    print("MaskablePPO with Transformer initialized.")
    print(f"   Policy architecture: {model.policy}")
    print(f"   Feature extractor: TransformerFeaturesExtractor")
    
    # --- 4. Set up callbacks ---
    print("\n⚙️  Setting up callbacks...")
    
    callbacks = []
    
    # Guidance callback.
    if guidance_callback is not None:
        callbacks.append(guidance_callback)
        print(f"   Guidance callback added (top_k={top_k_guidance})")
    
    # Metrics callback with logging.
    metrics_callback = MetricsCallback(log_dir=output_dir, verbose=1)
    callbacks.append(metrics_callback)
    print(f"   Metrics callback added (logging to {output_dir})")
    
    # Checkpoint callback.
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=output_dir,
        name_prefix='orchestrator_checkpoint'
    )
    callbacks.append(checkpoint_callback)
    print(f"   Checkpoint callback added (save every 10k steps to {output_dir})")
    
    # --- 5. Train ---
    print("\n" + "=" * 80)
    print("Starting training")
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
    
    # --- 6. Save the final model ---
    print("\n💾 Saving final model...")
    final_model_path = os.path.join(output_dir, "vhas_orchestrator_transformer_final")
    model.save(final_model_path)
    print(f"   ✓ Model saved to: {final_model_path}")
    
    # Save VecNormalize statistics for inference.
    vec_normalize_path = os.path.join(output_dir, "vec_normalize.pkl")
    env.save(vec_normalize_path)
    print(f"   ✓ VecNormalize stats saved to: {vec_normalize_path}")
    
    # Save training stats.
    stats = {
        'total_timesteps': model.num_timesteps,
        'episode_rewards': metrics_callback.episode_rewards,
        'episode_lengths': metrics_callback.episode_lengths,
        'episode_count': metrics_callback.episode_count,
        'final_avg_reward': float(np.mean(metrics_callback.episode_rewards[-100:])) if metrics_callback.episode_rewards else 0,
        'final_avg_length': float(np.mean(metrics_callback.episode_lengths[-100:])) if metrics_callback.episode_lengths else 0,
        'reward_mean': env.ret_rms.mean if hasattr(env, 'ret_rms') else None,
        'reward_var': env.ret_rms.var if hasattr(env, 'ret_rms') else None
    }
    
    import json
    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        # Convert numpy values to JSON-safe types.
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
    print("Training complete!")
    print("=" * 80)
    print(f"Total episodes: {metrics_callback.episode_count}")
    print(f"Total timesteps: {model.num_timesteps:,}")
    if metrics_callback.episode_rewards:
        print(f"Final avg reward (last 100): {np.mean(metrics_callback.episode_rewards[-100:]):.2f}")
        print(f"Final avg length (last 100): {np.mean(metrics_callback.episode_lengths[-100:]):.1f}")
    print(f"\nModel saved to: {final_model_path}.zip")
    print("TensorBoard logs: ./ppo_vhas_tensorboard/")
    print("=" * 80)
    
    return model, stats


if __name__ == "__main__":
    # Example usage.
    model, stats = train_orchestrator_baseline(
        encoder_path='all-mpnet-base-v2',
        embedding_space_path=None,
        scenarios_data_dir='../data/scenarios/data',
        kb_path='data/simulation_kb.json',
        output_dir='output/vhas_orchestrator_transformer_guided',
        total_timesteps=50000,
        top_k_guidance=5,
        device='auto'
    )
    
    print("\nTraining pipeline completed successfully!")
