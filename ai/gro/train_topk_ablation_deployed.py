# train_topk_ablation_deployed.py
"""
Experiment: Top-K Ablation Study
Train Model A or B with different top-k values for guidance to find optimal setting.

Tests: k=1, k=3, k=5 (baseline)

USAGE:
   modal run --detach train_topk_ablation_deployed.py::start_training
   modal run --detach train_topk_ablation_deployed.py::start_training --model-choice model_a

   Có thể tắt máy sau khi chạy lệnh trên.
"""

import modal
from datetime import datetime

app = modal.App("vhas-topk-ablation-experiment")

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
def train_with_topk(
    top_k: int = 5,
    total_timesteps: int = 1000000,
    run_id: str = None,
    model_choice: str = "model_b"  # NEW: model_a or model_b (default: model_b)
):
    """
    Train Model A or B with specific top-k value.
    
    Args:
        model_choice: "model_a" or "model_b" (default: model_b)
    """
    import sys
    sys.path.insert(0, '/data/data')
    
    from train_orchestrator_sb3 import train_orchestrator_baseline
    
    model_name = "Model A (Base → Fine-tune)" if model_choice == "model_a" else "Model B (Base → Pre-train → Fine-tune)"
    
    print("=" * 80)
    print(f"🧪 EXPERIMENT: Top-K={top_k} Ablation")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Top-K Guidance: {top_k}")
    print(f"Device: cpu (MlpPolicy optimized)")
    print("=" * 80)
    
    # Set paths based on model choice
    encoder_path = f"/models/{model_choice}"
    embedding_space_path = f"/models/embedding_space_{'base' if model_choice == 'model_a' else 'pretrained'}_no_states"
    scenarios_data_dir = "/data/scenarios/data"
    kb_path = "/data/simulation_kb.json"
    
    # Output with clear naming
    if run_id:
        output_dir = f"/results/topk_{top_k}_{model_choice}_{run_id}"
    else:
        output_dir = f"/results/topk_{top_k}_{model_choice}"
    
    # Train with specified top-k
    try:
        model, stats = train_orchestrator_baseline(
            encoder_path=encoder_path,
            embedding_space_path=embedding_space_path,
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            total_timesteps=total_timesteps,
            output_dir=output_dir,
            top_k_guidance=top_k,
            device="cpu"
        )
        
        # Commit results
        training_results_vol.commit()
        
        print("\n" + "=" * 80)
        print(f"✅ Training Complete - Top-K={top_k}")
        print("=" * 80)
        print(f"\nFinal Stats:")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
        print(f"  Episodes: {stats['episode_count']}")
        print(f"  Final avg reward: {stats.get('final_avg_reward', 'N/A'):.2f}")
        print(f"  Final avg length: {stats.get('final_avg_length', 'N/A'):.2f}")
        
        return {
            "experiment": f"topk_{top_k}",
            "model": model_choice,
            "top_k": top_k,
            "status": "success",
            "stats": stats,
            "output_dir": output_dir
        }
    
    except Exception as e:
        print(f"\n❌ Training failed for top-k={top_k}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "experiment": f"topk_{top_k}",
            "model": model_choice,
            "top_k": top_k,
            "status": "failed",
            "error": str(e)
        }


@app.function(
    timeout=25000  # ~7 hours for 4 parallel experiments
)
def start_training(
    total_timesteps: int = 1000000,
    add_timestamp: bool = True,
    model_choice: str = "model_b"  # NEW: model_a or model_b (default: model_b)
):
    """
    Start top-k ablation study with multiple k values in parallel.
    Tests k=1,3,5 by default.
    
    Args:
        model_choice: "model_a" or "model_b" (default: model_b)
    """
    topk_values = [1, 3, 5]  # Fixed: test k=1,3,5
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    model_name = "Model A (Base → Fine-tune)" if model_choice == "model_a" else "Model B (Base → Pre-train → Fine-tune)"
    
    print("=" * 80)
    print("🧪 ABLATION STUDY: Top-K Guidance")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   • Experiment: Top-K Ablation")
    print(f"   • Model: {model_name}")
    print(f"   • Timesteps: {total_timesteps:,} per experiment")
    print(f"   • Top-K values: {topk_values}")
    print(f"   • CPU: 4-core per experiment")
    print(f"   • Run ID: {run_id or 'default'}")
    print(f"   • Parallel: {len(topk_values)} experiments")
    print(f"\n🌙 BACKGROUND MODE: Can disconnect after launch")
    
    print(f"\n🚀 Launching {len(topk_values)} parallel experiments...\n")
    
    # Run all top-k experiments in parallel
    results = list(
        train_with_topk.starmap(
            [(k, total_timesteps, run_id, model_choice) for k in topk_values]
        )
    )
    
    print(f"\n" + "=" * 80)
    print(f"✅ ALL EXPERIMENTS COMPLETE!")
    print(f"=" * 80)
    
    # Print summary
    for result in results:
        k = result['top_k']
        status = result['status']
        
        print(f"\n📊 Top-K={k} Results:")
        print(f"   Status: {status}")
        
        if status == 'success':
            stats = result['stats']
            print(f"   Total timesteps: {stats['total_timesteps']:,}")
            print(f"   Episodes: {stats['episode_count']}")
            print(f"   Final avg reward: {stats.get('final_avg_reward', 0):.2f}")
            print(f"   Final avg length: {stats.get('final_avg_length', 0):.2f}")
            print(f"   Output: {result['output_dir']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    print(f"\n💾 Download results:")
    for result in results:
        if result['status'] == 'success':
            folder = result['output_dir'].replace('/results/', '')
            print(f"   modal volume get vhas-training-results {folder} ./results_{folder}/")
    
    print(f"\n📈 Compare results:")
    print(f"   python compare_topk_results.py")
    
    return results


# Convenience function for single top-k test
@app.function(
    timeout=18000
)
def start_single_topk(
    top_k: int = 3,
    total_timesteps: int = 1000000,
    add_timestamp: bool = True,
    model_choice: str = "model_b"  # NEW: model_a or model_b (default: model_b)
):
    """
    Start training with a single top-k value (for quick tests).
    
    Args:
        model_choice: "model_a" or "model_b" (default: model_b)
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    model_name = "Model A" if model_choice == "model_a" else "Model B"
    print(f"🧪 Testing single Top-K={top_k} with {model_name}")
    
    result = train_with_topk.remote(
        top_k=top_k,
        total_timesteps=total_timesteps,
        run_id=run_id,
        model_choice=model_choice
    )
    
    print(f"\n✅ Experiment complete: Top-K={top_k} ({model_name})")
    return result
