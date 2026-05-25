# train_transformer_deployed.py
"""
Training VHAS Orchestrator with Transformer Policy + Guidance.
DEPLOYED MODE - runs in the background and keeps going if the machine shuts down.

USAGE:
    # Test with 2048 steps
   modal run --detach train_transformer_deployed.py::start_training --total-timesteps 2048
   
    # Full training with 1M steps
   modal run --detach train_transformer_deployed.py::start_training --total-timesteps 1000000

    After running the command above, you can close the machine right away.
    Training runs on Modal cloud.

Check logs:
   modal app logs vhas-orchestrator-transformer --follow

Download results:
   modal volume ls vhas-training-results
   modal volume get vhas-training-results transformer_<TIMESTAMP>/ ./results_transformer/
"""

import modal
from datetime import datetime

app = modal.App("vhas-orchestrator-transformer")

# Image with SB3, PyTorch, and dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "stable-baselines3[extra]>=2.0.0",
        "sb3-contrib>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "tensorboard>=2.14.0",
        "scikit-learn>=1.3.0"
    )
)

# Volumes - mount data and models
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=False)
finetuned_output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=False)
training_results_vol = modal.Volume.from_name("vhas-training-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # Transformer benefits from GPU.
    memory=16384,  # 16GB RAM for Transformer.
    timeout=14400,  # 4 hours for 1M timesteps.
    volumes={
        "/data": training_data_vol,
        "/models": finetuned_output_vol,
        "/results": training_results_vol
    }
)
def train_transformer(
    total_timesteps: int = 2048,  # Default: test with 2048 steps.
    top_k_guidance: int = 5,
    run_id: str = None
):
    """
    Train the orchestrator with a Transformer policy.
    Runs in the background and keeps going if the client disconnects.
    """
    import sys
    sys.path.insert(0, '/data/data/gro/transformer')  # Volume root is /, so the mount lands at /data/data/gro/transformer.
    
    from train_orchestrator_sb3 import train_orchestrator_baseline
    
    print("=" * 80)
    print("🚀 VHAS ORCHESTRATOR TRAINING - TRANSFORMER POLICY")
    print("=" * 80)
    print(f"Policy: Transformer (history-aware)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Top-k guidance: {top_k_guidance}")
    print(f"Device: A10G GPU (Transformer optimized)")
    print(f"Memory: 16GB RAM")
    print(f"🌙 DEPLOYED MODE: Training will continue even if you disconnect")
    print("=" * 80)
    
    # Set paths.
    encoder_path = "/models/model_a"  # Use the finetuned encoder.
    embedding_space_path = "/models/embedding_space_base_no_states"
    scenarios_data_dir = "/data/scenarios/data"
    kb_path = "/data/simulation_kb.json"
    
    # Create a unique output folder when run_id is provided.
    if run_id:
        output_dir = f"/results/transformer_{run_id}"
    else:
        output_dir = "/results/transformer"
    
    # Train.
    try:
        model, stats = train_orchestrator_baseline(
            encoder_path=encoder_path,
            embedding_space_path=embedding_space_path,
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            total_timesteps=total_timesteps,
            output_dir=output_dir,
            top_k_guidance=top_k_guidance,
            device="cuda"  # Use GPU
        )
        
        # Commit results.
        training_results_vol.commit()
        
        print("\n" + "=" * 80)
        print("✅ Training Complete - TRANSFORMER POLICY")
        print("=" * 80)
        print(f"\nFinal Stats:")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
        print(f"  Episodes: {stats['episode_count']}")
        
        # Format with type checks.
        avg_reward = stats.get('final_avg_reward')
        if isinstance(avg_reward, (int, float)):
            print(f"  Final avg reward (last 100): {avg_reward:.2f}")
        else:
            print(f"  Final avg reward (last 100): {avg_reward}")
        
        avg_length = stats.get('final_avg_length')
        if isinstance(avg_length, (int, float)):
            print(f"  Final avg length (last 100): {avg_length:.2f}")
        else:
            print(f"  Final avg length (last 100): {avg_length}")
        
        return {
            "model": "transformer",
            "status": "success",
            "stats": stats,
            "output_dir": output_dir
        }
    
    except Exception as e:
        print(f"\n❌ Training failed for Transformer: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "model": "transformer",
            "status": "failed",
            "error": str(e),
            "output_dir": output_dir
        }


@app.function(
    image=image,  # CRITICAL FIX: Need same image for deserialization
    timeout=18000  # 5 hours
)
def start_training(
    total_timesteps: int = 2048,  # Default: test with 2048 steps.
    top_k_guidance: int = 5,
    add_timestamp: bool = True
):
    """
    Function to trigger Transformer policy training.
    You can shut down the machine after calling it.
    """
    # Generate a run ID if timestamping is enabled.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    print("=" * 80)
    print("🚀 Starting Transformer Policy Training (Background Mode)")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   • Policy: Transformer (history-aware, 20-step sequences)")
    print(f"   • Timesteps: {total_timesteps:,}")
    print(f"   • Top-k guidance: {top_k_guidance}")
    print(f"   • GPU: A10G (24GB VRAM)")
    print(f"   • Memory: 16GB RAM")
    print(f"   • Architecture: 4 layers, 8 heads, 1024 feedforward")
    print(f"   • Run ID: {run_id or 'None (will overwrite existing)'}")
    
    if total_timesteps <= 10000:
        print(f"\n🧪 TEST MODE: Running with {total_timesteps} steps to verify pipeline")
    else:
        print(f"\n🎯 FULL TRAINING: Running with {total_timesteps:,} steps")
    
    print(f"\n🌙 BACKGROUND MODE:")
    print(f"   • Training will continue even if you disconnect")
    print(f"   • You can safely close the terminal or shut down the computer")
    print(f"   • Check progress later with: modal app logs vhas-orchestrator-transformer")
    
    print(f"\n🚀 Starting training...\n")
    
    result = train_transformer.remote(
        total_timesteps=total_timesteps,
        top_k_guidance=top_k_guidance,
        run_id=run_id
    )
    
    print(f"\n" + "=" * 80)
    print(f"✅ TRANSFORMER TRAINING COMPLETED!")
    print(f"=" * 80)
    
    if result['status'] == 'success':
        stats = result['stats']
        print(f"\n📊 Results:")
        print(f"   Status: {result['status']}")
        print(f"   Total timesteps: {stats['total_timesteps']:,}")
        print(f"   Episodes: {stats['episode_count']}")
        
        # Format with type checks.
        avg_reward = stats.get('final_avg_reward', 0)
        if isinstance(avg_reward, (int, float)):
            print(f"   Final avg reward: {avg_reward:.2f}")
        else:
            print(f"   Final avg reward: {avg_reward}")
        
        avg_length = stats.get('final_avg_length', 0)
        if isinstance(avg_length, (int, float)):
            print(f"   Final avg length: {avg_length:.2f}")
        else:
            print(f"   Final avg length: {avg_length}")
        
        print(f"   Output: {result['output_dir']}")
        
        if total_timesteps <= 10000:
            print(f"\n✅ Pipeline verification successful!")
            print(f"   You can now run full training with --total-timesteps 1000000")
    else:
        print(f"\n❌ Training failed: {result.get('error', 'Unknown')}")
    
    print(f"\n💾 Download the trained model:")
    print(f"   modal volume get vhas-training-results {result.get('output_dir', '').replace('/results/', '')} ./results_transformer/")
    
    return result


# Entry point for CLI
if __name__ == "__main__":
    with app.run():
        start_training.remote(total_timesteps=2048, top_k_guidance=5)
