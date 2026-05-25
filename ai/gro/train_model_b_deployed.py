# train_model_b_deployed.py
"""
Train the VHAS Orchestrator for Model B (Base → Pre-train → Fine-tune) with Guidance.
DEPLOYED MODE: Runs in background and continues if the client disconnects.

Usage:

    modal run --detach train_model_b_deployed.py::start_training

After launching, you may close your machine — training runs on Modal cloud.

Check logs:

    modal app logs vhas-orchestrator-model-b --follow

Download results:

    modal volume ls vhas-training-results
    modal volume get vhas-training-results orchestrator_model_b_<TIMESTAMP>/ ./results_model_b/
"""

import modal
from datetime import datetime

app = modal.App("vhas-orchestrator-model-b")

# Docker image with SB3 and required dependencies
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

# Volumes mounted into the container
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=False)
finetuned_output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=False)
training_results_vol = modal.Volume.from_name("vhas-training-results", create_if_missing=True)


@app.function(
    image=image,
    cpu=4.0,  # MlpPolicy runs faster on CPU
    memory=8192,  # 8GB RAM
    timeout=14400,  # 4 hours for 1M timesteps
    volumes={
        "/data": training_data_vol,
        "/models": finetuned_output_vol,
        "/results": training_results_vol
    }
)
def train_model_b(
    total_timesteps: int = 1000000,
    top_k_guidance: int = 5,
    run_id: str = None
):
    """
    Train the orchestrator for Model B (Base → Pre-train → Fine-tune).
    Runs in background and continues if the client disconnects.
    """
    import sys
    sys.path.insert(0, '/data/data')
    
    from train_orchestrator_sb3 import train_orchestrator_baseline
    
    print("=" * 80)
    print("🚀 VHAS Orchestrator Training - MODEL B")
    print("=" * 80)
    print(f"Model: Model B (Base → Pre-train → Fine-tune)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Top-k guidance: {top_k_guidance}")
    print(f"Device: cpu (MlpPolicy optimized)")
    print(f"🌙 DEPLOYED MODE: Training will continue even if you disconnect")
    print("=" * 80)
    
    # Set paths
    encoder_path = "/models/model_b"
    embedding_space_path = "/models/embedding_space_pretrained_no_states"
    scenarios_data_dir = "/data/scenarios/data"
    kb_path = "/data/simulation_kb.json"
    
    # Create a unique output folder; include run_id when provided
    if run_id:
        output_dir = f"/results/orchestrator_model_b_{run_id}"
    else:
        output_dir = "/results/orchestrator_model_b"
    
    # Execute training
    try:
        model, stats = train_orchestrator_baseline(
            encoder_path=encoder_path,
            embedding_space_path=embedding_space_path,
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            total_timesteps=total_timesteps,
            output_dir=output_dir,
            top_k_guidance=top_k_guidance,
            device="cpu"
        )
        
        # Commit results
        training_results_vol.commit()
        
        print("\n" + "=" * 80)
        print("✅ Training Complete - MODEL B")
        print("=" * 80)
        print(f"\nFinal Stats:")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
        print(f"  Episodes: {stats['episode_count']}")
        print(f"  Final avg reward (last 100): {stats.get('final_avg_reward', 'N/A'):.2f}")
        print(f"  Final avg length (last 100): {stats.get('final_avg_length', 'N/A'):.2f}")
        
        return {
            "model": "model_b",
            "status": "success",
            "stats": stats,
            "output_dir": output_dir
        }
    
    except Exception as e:
        print(f"\n❌ Training failed for Model B: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "model": "model_b",
            "status": "failed",
            "error": str(e),
            "output_dir": output_dir
        }


@app.function(
    timeout=18000  # 5 hours
)
def start_training(
    total_timesteps: int = 1000000,
    top_k_guidance: int = 5,
    add_timestamp: bool = True
):
    """
    Trigger function to start training for Model B.
    Training will continue even if the client disconnects.
    """
    # Generate run ID if timestamp requested
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    print("=" * 80)
    print("🚀 Starting Model B Training (Background Mode)")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   • Model: Model B (Base → Pre-train → Fine-tune)")
    print(f"   • Timesteps: {total_timesteps:,}")
    print(f"   • Top-k guidance: {top_k_guidance}")
    print(f"   • CPU: 4-core (MlpPolicy optimized)")
    print(f"   • Run ID: {run_id or 'None (will overwrite existing)'}")
    print(f"\n🌙 BACKGROUND MODE:")
    print(f"   • Training will continue even if you disconnect")
    print(f"   • You can safely close terminal or shutdown computer")
    print(f"   • Check progress later with: modal app logs vhas-orchestrator-model-b")
    
    print(f"\n🚀 Starting training...\n")
    
    result = train_model_b.remote(
        total_timesteps=total_timesteps,
        top_k_guidance=top_k_guidance,
        run_id=run_id
    )
    
    print(f"\n" + "=" * 80)
    print(f"✅ MODEL B TRAINING COMPLETED!")
    print(f"=" * 80)
    
    if result['status'] == 'success':
        stats = result['stats']
        print(f"\n📊 Results:")
        print(f"   Status: {result['status']}")
        print(f"   Total timesteps: {stats['total_timesteps']:,}")
        print(f"   Episodes: {stats['episode_count']}")
        print(f"   Final avg reward: {stats.get('final_avg_reward', 0):.2f}")
        print(f"   Final avg length: {stats.get('final_avg_length', 0):.2f}")
        print(f"   Output: {result['output_dir']}")
    else:
        print(f"\n❌ Training failed: {result.get('error', 'Unknown')}")
    
    print(f"\n💾 Download trained model:")
    print(f"   modal volume get vhas-training-results {result.get('output_dir', '').replace('/results/', '')} ./results_model_b/")
    
    return result


# Entry point for CLI
if __name__ == "__main__":
    with app.run():
        start_training.remote(total_timesteps=1000000, top_k_guidance=5)

