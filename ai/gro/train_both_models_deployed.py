# train_both_models_deployed.py
"""
Training VHAS Orchestrator với MLP + Guidance cho CẢ 2 MODELS (A và B) 
DEPLOYED MODE - chạy background không bị cancel khi tắt máy.

USAGE - QUAN TRỌNG: PHẢI DÙNG --detach:
   
   modal run --detach train_both_models_deployed.py::start_training

   Sau khi chạy lệnh trên, BẠN CÓ THỂ TẮT MÁY NGAY.
   
   Training sẽ chạy trên Modal cloud ~2 giờ.

Kiểm tra logs:
   modal app logs vhas-orchestrator-dual-training --follow

Download kết quả:
   modal volume ls vhas-training-results
   modal volume get vhas-training-results /orchestrator_mlp_guided_model_a_<TIMESTAMP>/ ./results_a/
"""

import modal
import time

app = modal.App("vhas-orchestrator-dual-training")

# Image với SB3 và dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "stable-baselines3[extra]>=2.0.0",
        "sb3-contrib>=2.0.0",  # Added for MaskablePPO
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
    cpu=4.0,  # MlpPolicy runs faster on CPU
    memory=8192,  # 8GB RAM
    timeout=14400,  # 4 hours for 1M timesteps
    volumes={
        "/data": training_data_vol,
        "/models": finetuned_output_vol,
        "/results": training_results_vol
    }
)
def train_single_model(
    model_choice: str = "model_b",
    total_timesteps: int = 200000,
    top_k_guidance: int = 5,
    run_id: str = None
):
    """
    Train orchestrator với một model cụ thể.
    Chạy trong background, không bị cancel khi client disconnect.
    """
    from datetime import datetime
    import sys
    sys.path.insert(0, '/data/data')
    
    from train_orchestrator_sb3 import train_orchestrator_baseline
    
    print("=" * 80)
    print(f"🚀 VHAS Orchestrator Training - {model_choice.upper()}")
    print("=" * 80)
    print(f"Model: {model_choice}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Top-k guidance: {top_k_guidance}")
    print(f"Device: cpu (MlpPolicy optimized)")
    print(f"🌙 DEPLOYED MODE: Training will continue even if you disconnect")
    print("=" * 80)
    
    # Set paths
    encoder_path = f"/models/{model_choice}"
    embedding_suffix = "base_no_states" if model_choice == "model_a" else "pretrained_no_states"
    embedding_space_path = f"/models/embedding_space_{embedding_suffix}"
    scenarios_data_dir = "/data/scenarios/data"
    kb_path = "/data/simulation_kb.json"
    
    # Create unique output folder with timestamp if run_id provided
    if run_id:
        output_dir = f"/results/orchestrator_mlp_guided_{model_choice}_{run_id}"
    else:
        output_dir = f"/results/orchestrator_mlp_guided_{model_choice}"
    
    # Train
    try:
        model, stats = train_orchestrator_baseline(
            encoder_path=encoder_path,
            embedding_space_path=embedding_space_path,
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            total_timesteps=total_timesteps,
            output_dir=output_dir,
            top_k_guidance=top_k_guidance,
            device="cpu"  # MlpPolicy faster on CPU
        )
        
        # Commit results
        training_results_vol.commit()
        
        print("\n" + "=" * 80)
        print(f"✅ Training Complete - {model_choice.upper()}")
        print("=" * 80)
        print(f"\nFinal Stats:")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
        print(f"  Training time: {stats.get('training_time', 0):.2f}s ({stats.get('training_time', 0)/3600:.2f}h)")
        print(f"  Mean reward (last 100): {stats.get('mean_reward_100', 'N/A')}")
        print(f"  Success rate (last 100): {stats.get('success_rate_100', 'N/A')}")
        
        return {
            "model": model_choice,
            "status": "success",
            "stats": stats,
            "output_dir": output_dir
        }
    
    except Exception as e:
        print(f"\n❌ Training failed for {model_choice}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "model": model_choice,
            "status": "failed",
            "error": str(e),
            "output_dir": output_dir
        }


@app.function(
    timeout=18000  # 5 hours - enough for parallel training + buffer
)
def start_training(
    total_timesteps: int = 1000000,  # Tăng lên 1M để converge tốt hơn
    top_k_guidance: int = 5,
    add_timestamp: bool = True
):
    """
    Function để trigger training cho cả 2 models.
    Có thể tắt máy sau khi gọi function này.
    """
    from datetime import datetime
    
    # Generate run ID if timestamp requested
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    print("=" * 80)
    print("🚀 Starting Dual Model Training (Background Mode)")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   • Training: Model A + Model B (parallel)")
    print(f"   • Timesteps: {total_timesteps:,} per model")
    print(f"   • Top-k guidance: {top_k_guidance}")
    print(f"   • CPU: 2x 4-core (MlpPolicy optimized)")
    print(f"   • Run ID: {run_id or 'None (will overwrite existing)'}")
    print(f"\n🌙 BACKGROUND MODE:")
    print(f"   • Training will continue even if you disconnect")
    print(f"   • You can safely close terminal or shutdown computer")
    print(f"   • Check progress later with: modal app logs vhas-orchestrator-dual-training")
    
    print(f"\n🌐 Launching 2 parallel training jobs...\n")
    
    # Launch both in parallel
    results = list(
        train_single_model.starmap(
            [
                ("model_a", total_timesteps, top_k_guidance, run_id),
                ("model_b", total_timesteps, top_k_guidance, run_id)
            ]
        )
    )
    
    print(f"\n" + "=" * 80)
    print(f"✅ BOTH MODELS TRAINING COMPLETED!")
    print(f"=" * 80)
    
    for result in results:
        model = result['model']
        status = result['status']
        
        print(f"\n📊 {model.upper()} Results:")
        print(f"   Status: {status}")
        
        if status == 'success':
            stats = result['stats']
            print(f"   Total timesteps: {stats['total_timesteps']:,}")
            print(f"   Training time: {stats.get('training_time', 0):.2f}s ({stats.get('training_time', 0)/3600:.2f}h)")
            print(f"   Mean reward (last 100): {stats.get('mean_reward_100', 'N/A')}")
            print(f"   Success rate (last 100): {stats.get('success_rate_100', 'N/A')}")
            print(f"   Output: {result['output_dir']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    print(f"\n💾 Download trained models:")
    print(f"   modal volume get vhas-training-results orchestrator_mlp_guided_model_a ./results_a")
    print(f"   modal volume get vhas-training-results orchestrator_mlp_guided_model_b ./results_b")
    
    return results


# Entry point for CLI
if __name__ == "__main__":
    # Sau khi deploy, gọi function này để bắt đầu training
    with app.run():
        start_training.remote(total_timesteps=200000, top_k_guidance=5)
