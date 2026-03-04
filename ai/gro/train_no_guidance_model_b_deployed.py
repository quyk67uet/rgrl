# train_no_guidance_model_b_deployed.py
"""
Experiment: Training Model B WITHOUT Guidance (Ablation Study)
DEPLOYED MODE - chạy background không bị cancel khi tắt máy.

USAGE:
   modal run --detach train_no_guidance_model_b_deployed.py::start_training

   Sau khi chạy lệnh trên, BẠN CÓ THỂ TẮT MÁY NGAY.
   Training sẽ chạy trên Modal cloud.

Kiểm tra logs:
   modal app logs vhas-no-guidance-model-b --follow

Download kết quả:
   modal volume ls vhas-training-results
   modal volume get vhas-training-results no_guidance_model_b_<TIMESTAMP>/ ./results_no_guidance_model_b/
"""

import modal
from datetime import datetime

app = modal.App("vhas-no-guidance-model-b")

# Image với SB3 và dependencies
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

# Volumes
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=False)
finetuned_output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=False)
training_results_vol = modal.Volume.from_name("vhas-training-results", create_if_missing=True)


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
def train_model_b_no_guidance(
    total_timesteps: int = 1000000,
    run_id: str = None
):
    """
    Train Model B without guidance mechanism.
    """
    import sys
    sys.path.insert(0, '/data/data')
    
    from train_orchestrator_sb3 import train_orchestrator_baseline
    
    print("=" * 80)
    print("🧪 EXPERIMENT: Model B WITHOUT Guidance")
    print("=" * 80)
    print(f"Model: Model B (Base → Pre-train → Fine-tune)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Guidance: DISABLED (Ablation Study)")
    print(f"Device: cpu (MlpPolicy optimized)")
    print(f"🌙 DEPLOYED MODE: Training will continue even if you disconnect")
    print("=" * 80)
    
    # Set paths
    encoder_path = "/models/model_b"
    scenarios_data_dir = "/data/scenarios/data"
    kb_path = "/data/simulation_kb.json"
    
    # Output with clear naming
    if run_id:
        output_dir = f"/results/no_guidance_model_b_{run_id}"
    else:
        output_dir = "/results/no_guidance_model_b"
    
    # Train WITHOUT guidance (embedding_space_path=None)
    try:
        model, stats = train_orchestrator_baseline(
            encoder_path=encoder_path,
            embedding_space_path=None,  # NO GUIDANCE
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            total_timesteps=total_timesteps,
            output_dir=output_dir,
            top_k_guidance=5,  # Ignored when no guidance
            device="cpu"
        )
        
        # Commit results
        training_results_vol.commit()
        
        print("\n" + "=" * 80)
        print("✅ Training Complete - Model B (NO GUIDANCE)")
        print("=" * 80)
        print(f"\nFinal Stats:")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
        print(f"  Episodes: {stats['episode_count']}")
        print(f"  Final avg reward: {stats.get('final_avg_reward', 'N/A'):.2f}")
        print(f"  Final avg length: {stats.get('final_avg_length', 'N/A'):.2f}")
        
        return {
            "experiment": "no_guidance",
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
            "experiment": "no_guidance",
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
    add_timestamp: bool = True
):
    """
    Function để trigger training cho Model B without guidance.
    Có thể tắt máy sau khi gọi function này.
    """
    # Generate run ID if timestamp requested
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    print("=" * 80)
    print("🧪 ABLATION STUDY: Model B Without Guidance")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   • Experiment: No Guidance Baseline")
    print(f"   • Model: Model B (Base → Pre-train → Fine-tune)")
    print(f"   • Timesteps: {total_timesteps:,}")
    print(f"   • Guidance: DISABLED")
    print(f"   • CPU: 4-core (MlpPolicy optimized)")
    print(f"   • Run ID: {run_id or 'None (will overwrite existing)'}")
    print(f"\n🌙 BACKGROUND MODE:")
    print(f"   • Training will continue even if you disconnect")
    print(f"   • You can safely close terminal or shutdown computer")
    print(f"   • Check progress later with: modal app logs vhas-no-guidance-model-b")
    
    print(f"\n🚀 Starting training...\n")
    
    result = train_model_b_no_guidance.remote(
        total_timesteps=total_timesteps,
        run_id=run_id
    )
    
    print(f"\n" + "=" * 80)
    print(f"✅ MODEL B (NO GUIDANCE) TRAINING COMPLETED!")
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
    print(f"   modal volume get vhas-training-results {result.get('output_dir', '').replace('/results/', '')} ./results_no_guidance_model_b/")
    
    return result


# Entry point for CLI
if __name__ == "__main__":
    with app.run():
        start_training.remote(total_timesteps=1000000)


