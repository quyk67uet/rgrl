# train.py
"""
Training script for GNN-based VHAS Orchestrator using RSL-RL.

This script trains the ActorCriticGNN policy on clinical workflow data,
using graph-structured observations (HeteroData) and action masking.

Architecture: AFAN (Anticipatory Flow Anticipation Network) with GATv2Conv
Framework: RSL-RL (supports TensorDict + complex observations)

USAGE:
    python train.py
    
    Or customize parameters:
    python -c "from train import train_vhas_gnn; train_vhas_gnn(max_iterations=1000)"
"""

import os
import sys
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Add project paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../rsl_rl'))

# RSL-RL imports
from rsl_rl.runners import OnPolicyRunner

# VHAS imports
from env import ClinicalWorkflowEnv
from wrappers import VHAS_GNN_Wrapper
from actor_critic_gnn import ActorCriticGNN
from vec_env_wrapper import VHAS_VecEnv


def make_env(
    encoder_path: str,
    scenarios_data_dir: str,
    kb_path: str,
    guidance_encoder_path: str = None,
    guidance_embedding_space_path: str = None,
    use_guidance: bool = True,
    top_k_guidance: int = 5,
    device: str = "cuda"
):
    """
    Build a wrapped ClinicalWorkflowEnv.

    Steps:
    1. Load the SentenceTransformer encoder
    2. Create ClinicalWorkflowEnv (HeteroData output)
    3. Wrap it with VHAS_GNN_Wrapper (HeteroData -> TensorDict)

    Returns:
        VHAS_GNN_Wrapper: RSL-RL compatible env
    """
    # Load encoder
    encoder = SentenceTransformer(encoder_path, device=device)
    
    # Create base env (returns HeteroData observations)
    raw_env = ClinicalWorkflowEnv(
        encoder_model=encoder,
        scenarios_data_dir=scenarios_data_dir,
        kb_path=kb_path,
        guidance_encoder_path=guidance_encoder_path,
        guidance_embedding_space_path=guidance_embedding_space_path,
        use_guidance=use_guidance,
        top_k_guidance=top_k_guidance,
        device=device
    )
    
    # Wrap to convert HeteroData -> TensorDict for RSL-RL
    wrapped_env = VHAS_GNN_Wrapper(
        env=raw_env,
        max_nodes=50,  # Max nodes per type (padding)
        max_edges=100,  # Max edges per type (padding)
        embedding_dim=encoder.get_sentence_embedding_dimension()
    )
    
    return wrapped_env


def train_vhas_gnn(
    # Data paths
    encoder_path: str = 'all-mpnet-base-v2',
    scenarios_data_dir: str = '../data/scenarios/data',
    kb_path: str = '../data/simulation_kb.json',
    guidance_encoder_path: str = None,
    guidance_embedding_space_path: str = None,
    
    # Training hyperparameters
    num_envs: int = 1,  # Start with 1 for debugging
    num_steps_per_env: int = 256,  # Steps per rollout
    max_iterations: int = 500,  # Total training iterations
    
    # PPO hyperparameters (tuned for stability and exploration)
    learning_rate: float = 1e-4,  # Lower than 3e-4 for stability
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 0.5,  # Lowered to avoid value loss spikes
    entropy_coef: float = 0.02,  # Raised for better exploration
    
    # GNN policy hyperparameters
    actor_hidden_dims: list = None,
    critic_hidden_dims: list = None,
    
    # Guidance
    use_guidance: bool = True,
    top_k_guidance: int = 5,
    
    # Logging
    log_dir: str = None,
    experiment_name: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Main training function for GNN-based VHAS Orchestrator.

    Args:
        encoder_path: Path to the SentenceTransformer model
        scenarios_data_dir: Directory with expert traces
        kb_path: Path to the simulation knowledge base
        guidance_encoder_path: Optional guidance encoder path
        guidance_embedding_space_path: Optional guidance embedding path
        num_envs: Number of parallel environments
        num_steps_per_env: Steps per env per iteration
        max_iterations: Total training iterations
        learning_rate: PPO learning rate
        num_learning_epochs: PPO epochs per update
        num_mini_batches: Mini-batches per epoch
        clip_param: PPO clip parameter
        gamma: Discount factor
        lam: GAE lambda
        value_loss_coef: Value loss weight
        entropy_coef: Entropy bonus weight
        actor_hidden_dims: Actor MLP hidden sizes
        critic_hidden_dims: Critic MLP hidden sizes
        use_guidance: Enable action masking guidance
        top_k_guidance: Top-k guidance actions
        log_dir: Log and checkpoint directory
        experiment_name: Experiment name
        device: Device to use ('cuda' or 'cpu')
    """
    
    print("=" * 80)
    print("🚀 VHAS GNN-BASED ORCHESTRATOR TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Encoder: {encoder_path}")
    print(f"Number of environments: {num_envs}")
    print(f"Steps per environment: {num_steps_per_env}")
    print(f"Max iterations: {max_iterations}")
    print(f"Guidance: {'ENABLED' if use_guidance else 'DISABLED'}")
    print("=" * 80)
    
    # Set default hidden dimensions if needed
    if actor_hidden_dims is None:
        actor_hidden_dims = [256, 256]
    if critic_hidden_dims is None:
        critic_hidden_dims = [256, 256]
    
    # Generate an experiment name if needed
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"vhas_gnn_{timestamp}"
    
    # Set log directory
    if log_dir is None:
        log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # --- 1. Create vectorized environment ---
    print("\n🏗️  Creating vectorized environment...")
    
    def env_factory():
        """Build one wrapped VHAS environment."""
        return make_env(
            encoder_path=encoder_path,
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            guidance_encoder_path=guidance_encoder_path,
            guidance_embedding_space_path=guidance_embedding_space_path,
            use_guidance=use_guidance,
            top_k_guidance=top_k_guidance,
            device=device
        )
    
    # Create VHAS_VecEnv wrapper (RSL-RL VecEnv interface)
    env = VHAS_VecEnv(
        env_fn=env_factory,
        num_envs=num_envs,
        device=device,
        max_episode_length=20  # Max steps per episode
    )
    
    print(f"   ✓ VecEnv created successfully")
    print(f"   • Number of parallel environments: {env.num_envs}")
    print(f"   • Number of actions: {env.num_actions}")
    print(f"   • Max episode length: {env.max_episode_length}")
    print(f"   • Device: {env.device}")

    # --- 2. Create training config ---
    print("\n⚙️  Creating training configuration...")
    
    # Get observation keys from the vectorized env
    sample_obs = env.get_observations()
    obs_keys = list(sample_obs.keys())
    
    train_cfg = {
        # Observation groups (RSL-RL requirement)
        # Choose keys for policy vs critic
        "obs_groups": {
            "policy": obs_keys,  # Use all keys for policy
            "critic": obs_keys   # Use all keys for critic
        },
        
        # Algorithm config (PPO)
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "learning_rate": learning_rate,
            "num_learning_epochs": num_learning_epochs,
            "num_mini_batches": num_mini_batches,
            "clip_param": clip_param,
            "gamma": gamma,
            "lam": lam,
            "value_loss_coef": value_loss_coef,
            "entropy_coef": entropy_coef,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,  # No Random Network Distillation
            "symmetry_cfg": None  # No symmetry augmentation
        },
        
        # Policy config (GNN)
        # OnPolicyRunner calls:
        #   ActorCriticGNN(obs, obs_groups, env.num_actions, **policy_cfg)
        # so do not pass num_actions / num_actor_obs / num_critic_obs here
        "policy": {
            "class_name": "actor_critic_gnn.ActorCriticGNN",  # Custom GNN policy
            "actor_hidden_dims": actor_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
            "activation": "elu",
        },
        
        # Runner config
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        "save_interval": 50,  # Save a checkpoint every 50 iters
        "experiment_name": experiment_name,
            "run_name": ""
        }
    
    print(f"   ✓ Configuration created")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Gamma: {gamma}")
    print(f"   • Clip param: {clip_param}")
    print(f"   • Entropy coefficient: {entropy_coef} (increased for better exploration)")
    print(f"   • Value loss coefficient: {value_loss_coef} (reduced for stability)")
    print(f"   • Actor hidden dims: {actor_hidden_dims}")
    print(f"   • Critic hidden dims: {critic_hidden_dims}")
    
    # --- 3. Create RSL-RL runner ---
    print("\n🤖 Initializing OnPolicyRunner...")
    
    try:
        runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device
        )
        print(f"   ✓ Runner initialized")
        print(f"   • Policy: ActorCriticGNN")
        print(f"   • Algorithm: PPO")
        print(f"   • Log directory: {log_dir}")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize runner")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # --- 4. Start training ---
    print("\n" + "=" * 80)
    print("🎮 STARTING TRAINING")
    print("=" * 80)
    print(f"Total training steps: {max_iterations * num_steps_per_env:,}")
    print(f"Expected time: ~{max_iterations * num_steps_per_env / 100 / 60:.1f} minutes (estimate)")
    print("=" * 80 + "\n")
    
    try:
        runner.learn(num_learning_iterations=max_iterations)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Total iterations: {max_iterations}")
        print(f"Model saved to: {log_dir}")
        print(f"Tensorboard logs: {log_dir}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print(f"   Partial model saved to: {log_dir}")
        
    except Exception as e:
        print(f"\n❌ ERROR during training:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return runner


if __name__ == "__main__":
    # Example usage - adjust paths as needed

    # Local testing
    train_vhas_gnn(
        encoder_path='all-mpnet-base-v2',  # Use base model
        scenarios_data_dir='../data/scenarios/data',
        kb_path='../data/simulation_kb.json',
        guidance_encoder_path=None,  # Set an actual path if available
        guidance_embedding_space_path=None,  # Set an actual path if available
        use_guidance=False,  # Disable for initial testing
        num_envs=1,  # Single env for debugging
        num_steps_per_env=1024,
        max_iterations=10,  # Only 10 iterations for testing
        learning_rate=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n🎉 Training pipeline completed successfully!")
