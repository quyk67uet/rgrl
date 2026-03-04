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
    Factory function to create a wrapped ClinicalWorkflowEnv.
    
    This function:
    1. Loads the SentenceTransformer encoder
    2. Creates ClinicalWorkflowEnv (returns HeteroData)
    3. Wraps with VHAS_GNN_Wrapper (converts HeteroData → TensorDict)
    
    Returns:
        VHAS_GNN_Wrapper: Environment compatible with RSL-RL
    """
    # Load encoder
    encoder = SentenceTransformer(encoder_path, device=device)
    
    # Create base environment (returns HeteroData observations)
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
    
    # Wrap to convert HeteroData → TensorDict (for RSL-RL compatibility)
    wrapped_env = VHAS_GNN_Wrapper(
        env=raw_env,
        max_nodes=50,  # Maximum nodes per type (padding size)
        max_edges=100,  # Maximum edges per type (padding size)
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
    num_envs: int = 1,  # Start with 1 for debugging, can scale up
    num_steps_per_env: int = 256,  # Steps per rollout (reduced for faster testing)
    max_iterations: int = 500,  # Total training iterations
    
    # PPO hyperparameters (optimized for stability and exploration)
    learning_rate: float = 1e-4,  # Reduced from 3e-4 for better stability
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 0.5,  # Reduced from 1.0 to prevent value loss explosion
    entropy_coef: float = 0.02,  # INCREASED from 0.01 to 0.02 for better exploration
    
    # GNN Policy hyperparameters
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
        encoder_path: Path to SentenceTransformer model
        scenarios_data_dir: Directory containing expert traces
        kb_path: Path to simulation knowledge base
        guidance_encoder_path: Path to finetuned encoder for guidance (optional)
        guidance_embedding_space_path: Path to embedding space for guidance (optional)
        num_envs: Number of parallel environments
        num_steps_per_env: Steps collected per environment per iteration
        max_iterations: Total training iterations
        learning_rate: Learning rate for PPO
        num_learning_epochs: Number of epochs per PPO update
        num_mini_batches: Number of mini-batches per epoch
        clip_param: PPO clipping parameter
        gamma: Discount factor
        lam: GAE lambda
        value_loss_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        actor_hidden_dims: Hidden layer sizes for actor MLP head
        critic_hidden_dims: Hidden layer sizes for critic MLP head
        use_guidance: Whether to use guidance for action masking
        top_k_guidance: Number of top actions from guidance
        log_dir: Directory for logs and checkpoints
        experiment_name: Name for this experiment
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
    
    # Set default hidden dimensions if not provided
    if actor_hidden_dims is None:
        actor_hidden_dims = [256, 256]
    if critic_hidden_dims is None:
        critic_hidden_dims = [256, 256]
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"vhas_gnn_{timestamp}"
    
    # Set log directory
    if log_dir is None:
        log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # --- 1. CREATE VECTORIZED ENVIRONMENT ---
    print("\n🏗️  Creating vectorized environment...")
    
    def env_factory():
        """Factory function that creates a single wrapped VHAS environment."""
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
    
    # Create VHAS_VecEnv wrapper (implements RSL-RL's VecEnv interface)
    env = VHAS_VecEnv(
        env_fn=env_factory,
        num_envs=num_envs,
        device=device,
        max_episode_length=20  # Maximum steps per episode
    )
    
    print(f"   ✓ VecEnv created successfully")
    print(f"   • Number of parallel environments: {env.num_envs}")
    print(f"   • Number of actions: {env.num_actions}")
    print(f"   • Max episode length: {env.max_episode_length}")
    print(f"   • Device: {env.device}")

    # --- 2. CREATE TRAINING CONFIGURATION ---
    print("\n⚙️  Creating training configuration...")
    
    # Get observation keys from the vectorized environment
    sample_obs = env.get_observations()
    obs_keys = list(sample_obs.keys())
    
    train_cfg = {
        # Observation groups (RSL-RL requirement)
        # Specify which observation keys are used for policy vs critic
        "obs_groups": {
            "policy": obs_keys,  # Use all observation keys for policy
            "critic": obs_keys   # Use all observation keys for critic
        },
        
        # Algorithm configuration (PPO)
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
        
        # Policy configuration (GNN)
        # Lưu ý: OnPolicyRunner sẽ gọi:
        #   ActorCriticGNN(obs, obs_groups, env.num_actions, **policy_cfg)
        # nên KHÔNG truyền lại num_actions / num_actor_obs / num_critic_obs ở đây
        "policy": {
            "class_name": "actor_critic_gnn.ActorCriticGNN",  # trỏ tới custom GNN policy
            "actor_hidden_dims": actor_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
            "activation": "elu",
        },
        
        # Runner configuration
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        "save_interval": 50,  # Save checkpoint every 50 iterations
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
    
    # --- 3. CREATE RSL-RL RUNNER ---
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
    
    # --- 4. START TRAINING ---
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
    
    # For local testing
    train_vhas_gnn(
        encoder_path='all-mpnet-base-v2',  # Use base model
        scenarios_data_dir='../data/scenarios/data',
        kb_path='../data/simulation_kb.json',
        guidance_encoder_path=None,  # Set to actual path if available
        guidance_embedding_space_path=None,  # Set to actual path if available
        use_guidance=False,  # Disable for initial testing
        num_envs=1,  # Single environment for debugging
        num_steps_per_env=1024,
        max_iterations=10,  # Just 10 iterations for testing
        learning_rate=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n🎉 Training pipeline completed successfully!")
