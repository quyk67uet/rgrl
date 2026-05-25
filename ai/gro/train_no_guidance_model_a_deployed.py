# train_no_guidance_model_a_deployed.py
"""
Experiment: Train Model A WITHOUT Guidance (Ablation Study).
DEPLOYED MODE: Runs in background and continues if the client disconnects.

Usage:

    modal run --detach train_no_guidance_model_a_deployed.py::start_training

After launching, you may close your machine — training runs on Modal cloud.

Check logs:

    modal app logs vhas-no-guidance-model-a --follow

Download results:

    modal volume ls vhas-training-results
    modal volume get vhas-training-results no_guidance_model_a_<TIMESTAMP>/ ./results_no_guidance_model_a/
"""

import modal
from datetime import datetime
from pathlib import Path

app = modal.App("vhas-no-guidance-model-a")

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


def _resolve_stage2_policy_artifact_path(filename: str) -> Path:
    """Return the stage-2 policy artifact path for Modal or local execution."""
    modal_models_root = Path("/models")
    if modal_models_root.exists():
        return modal_models_root / "stage2_gro_policies" / filename
    return Path(__file__).resolve().parents[1] / "models" / "stage2_gro_policies" / filename


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=14400,  # 4 hours
    volumes={
        "/data": training_data_vol,
        "/models": finetuned_output_vol,
        "/results": training_results_vol
    }
)
def train_model_a_no_guidance(
    total_timesteps: int = 1000000,
    run_id: str = None
):
    """
    Train Model A without the guidance mechanism (ablation baseline).
    """
    import sys
    sys.path.insert(0, '/data/data')
    
    from train_orchestrator_sb3 import train_orchestrator_baseline
    
    print("=" * 80)
    print("🧪 EXPERIMENT: Model A WITHOUT Guidance")
    print("=" * 80)
    print(f"Model: Model A (Base → Fine-tune)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Guidance: DISABLED (Ablation Study)")
    print(f"Device: cpu (MlpPolicy optimized)")
    print(f"🌙 DEPLOYED MODE: Training will continue even if you disconnect")
    print("=" * 80)
    
    # Set paths
    encoder_path = "/models/model_a"
    scenarios_data_dir = "/data/scenarios/data"
    kb_path = "/data/simulation_kb.json"
    final_model_path = _resolve_stage2_policy_artifact_path("vanilla_rl_baseline.zip")
    
    # Output directory with clear naming
    if run_id:
        output_dir = f"/results/no_guidance_model_a_{run_id}"
    else:
        output_dir = "/results/no_guidance_model_a"
    
    # Train WITHOUT guidance (embedding_space_path=None)
    try:
        model, stats = train_orchestrator_baseline(
            encoder_path=encoder_path,
            embedding_space_path=None,  # NO GUIDANCE
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            total_timesteps=total_timesteps,
            output_dir=output_dir,
            final_model_path=final_model_path,
            top_k_guidance=5,  # Ignored when no guidance
            device="cpu"
        )
        
        # Commit results
        training_results_vol.commit()
        
        print("\n" + "=" * 80)
        print("✅ Training Complete - Model A (NO GUIDANCE)")
        print("=" * 80)
        print(f"\nFinal Stats:")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
        print(f"  Episodes: {stats['episode_count']}")
        print(f"  Final avg reward: {stats.get('final_avg_reward', 'N/A'):.2f}")
        print(f"  Final avg length: {stats.get('final_avg_length', 'N/A'):.2f}")
        
        return {
            "experiment": "no_guidance",
            "model": "model_a",
            "status": "success",
            "stats": stats,
            "output_dir": output_dir,
            "final_model_path": str(final_model_path)
        }
    
    except Exception as e:
        print(f"\n❌ Training failed for Model A: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "experiment": "no_guidance",
            "model": "model_a",
            "status": "failed",
            "error": str(e),
            "output_dir": output_dir,
            "final_model_path": str(final_model_path)
        }


@app.function(
    timeout=18000  # 5 hours
)
def start_training(
    total_timesteps: int = 1000000,
    add_timestamp: bool = True
):
    """
    Trigger function to start training for Model A without guidance.
    Training continues even if the client disconnects.
    """
    # Generate run ID if timestamp requested
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    print("=" * 80)
    print("🧪 ABLATION STUDY: Model A Without Guidance")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   • Experiment: No Guidance Baseline")
    print(f"   • Model: Model A (Base → Fine-tune)")
    print(f"   • Timesteps: {total_timesteps:,}")
    print(f"   • Guidance: DISABLED")
    print(f"   • CPU: 4-core (MlpPolicy optimized)")
    print(f"   • Run ID: {run_id or 'None (will overwrite existing)'}")
    print(f"\n🌙 BACKGROUND MODE:")
    print(f"   • Training will continue even if you disconnect")
    print(f"   • You can safely close terminal or shutdown computer")
    print(f"   • Check progress later with: modal app logs vhas-no-guidance-model-a")
    
    print(f"\n🚀 Starting training...\n")
    
    result = train_model_a_no_guidance.remote(
        total_timesteps=total_timesteps,
        run_id=run_id
    )
    
    print(f"\n" + "=" * 80)
    print(f"✅ MODEL A (NO GUIDANCE) TRAINING COMPLETED!")
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
    print(f"   modal volume get vhas-training-results {result.get('output_dir', '').replace('/results/', '')} ./results_no_guidance_model_a/")
    print(f"   Final model artifact: {result.get('final_model_path')}")
    
    return result


# Entry point for CLI
if __name__ == "__main__":
    with app.run():
        start_training.remote(total_timesteps=1000000)


