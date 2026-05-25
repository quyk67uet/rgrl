# train_topk_ablation_dual_deployed.py
"""Operational workflow compliance sensitivity study using a paired-encoder policy.

Evaluate the selected policy across multiple guidance thresholds to identify a
stable operating setting.

Evaluation points: k=1, k=3, k=5 (reference), k=8

USAGE:
   modal run --detach train_topk_ablation_dual_deployed.py::start_training

    After launching, you may close your machine — execution continues on Modal.
"""

import modal
from datetime import datetime

app = modal.App("vhas-topk-ablation-dual-experiment")

# Container image with required dependencies
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

# Persistent volumes
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
def train_with_topk_dual(
    top_k: int = 5,
    total_timesteps: int = 1000000,
    run_id: str = None,
    model_choice: str = "model_b"  # Selectable policy variant.
):
    """Run a single thresholded compliance evaluation for the selected policy.

    Args:
        model_choice: selectable policy variant (default: model_b)
    """
    import sys
    sys.path.insert(0, '/data/data')  # Resolve shared inputs from the mounted data root.
    
    from train_orchestrator_sb3 import train_orchestrator_baseline
    
    print("=" * 80)
    print(f"🧪 RUN: Threshold={top_k} compliance evaluation")
    print("=" * 80)
    print(f"Policy variant: {model_choice.upper()}")
    print(f"Controller structure: paired encoder")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Guidance threshold: {top_k}")
    print(f"Device: cpu (policy execution mode)")
    print("=" * 80)
    
    # Resolve artifact paths from the selected policy variant
    state_encoder_path = f"/models/{model_choice}_state_encoder"
    action_encoder_path = f"/models/{model_choice}_action_encoder"
    embedding_space_path = f"/models/embedding_space_{model_choice}_dual"
    scenarios_data_dir = "/data/scenarios/data"
    kb_path = "/data/simulation_kb.json"
    
    # Construct a stable output directory name
    if run_id:
        output_dir = f"/results/topk_{top_k}_{model_choice}_dual_{run_id}"
    else:
        output_dir = f"/results/topk_{top_k}_{model_choice}_dual"
    
    # Execute the selected thresholded run
    try:
        model, stats = train_orchestrator_baseline(
            encoder_path=state_encoder_path,
            embedding_space_path=embedding_space_path,
            action_encoder_path=action_encoder_path,
            use_dual_encoder=True,
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            total_timesteps=total_timesteps,
            output_dir=output_dir,
            top_k_guidance=top_k,
            device="cpu"
        )
        
        # Persist results to the volume
        training_results_vol.commit()
        
        print("\n" + "=" * 80)
        print(f"✅ Run complete - Threshold={top_k}")
        print("=" * 80)
        print(f"\nFinal Stats:")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
        print(f"  Episodes: {stats['episode_count']}")
        print(f"  Final avg reward: {stats.get('final_avg_reward', 'N/A'):.2f}")
        print(f"  Final avg length: {stats.get('final_avg_length', 'N/A'):.2f}")
        
        return {
            "experiment": f"topk_{top_k}_{model_choice}_dual",
            "top_k": top_k,
            "model": model_choice,
            "architecture": "paired-encoder",
            "status": "success",
            "stats": stats,
            "output_dir": output_dir
        }
    
    except Exception as e:
        print(f"\n❌ Run failed for threshold={top_k} ({model_choice}): {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "experiment": f"topk_{top_k}_{model_choice}_dual",
            "top_k": top_k,
            "model": model_choice,
            "architecture": "paired-encoder",
            "status": "failed",
            "error": str(e)
        }


@app.function(
    timeout=30000  # Allow headroom for four parallel runs.
)
def start_training(
    total_timesteps: int = 1000000,
    add_timestamp: bool = True,
    model_choice: str = "model_b"  # Selectable policy variant.
):
    """Launch a parallel threshold sweep for the selected policy.

    Evaluation points are k=1, k=3, k=5, and k=8 by default.

    Args:
        model_choice: selectable policy variant (default: model_b)
    """
    topk_values = [1, 3, 5, 8]  # Evaluation thresholds for the parallel sweep.
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    print("=" * 80)
    print("🧪 STUDY: Guidance threshold sweep")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   • Study: threshold sweep")
    print(f"   • Policy variant: {model_choice.upper()}")
    print(f"   • Controller structure: paired encoder")
    print(f"   • Timesteps: {total_timesteps:,} per experiment")
    print(f"   • Threshold values: {topk_values}")
    print(f"   • CPU: 4 cores per run")
    print(f"   • Run ID: {run_id or 'default'}")
    print(f"   • Parallel: {len(topk_values)} runs")
    print(f"   • Scope: 4 concurrent runs for k=1,3,5,8")
    print(f"\n🌙 DETACHED MODE: You can disconnect after launch")
    
    print(f"\n🚀 Launching {len(topk_values)} parallel runs (k=1,3,5,8)...\n")
    
    # Run all threshold evaluations in parallel
    results = list(
        train_with_topk_dual.starmap(
            [(k, total_timesteps, run_id, model_choice) for k in topk_values]
        )
    )
    
    print(f"\n" + "=" * 80)
    print(f"✅ ALL RUNS COMPLETE!")
    print(f"=" * 80)
    
    # Print summary
    for result in results:
        k = result['top_k']
        model = result.get('model', model_choice)
        status = result['status']
        
        print(f"\n📊 Threshold={k} ({model.upper()}) Results:")
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
    
    print(f"\n📈 Compare runs:")
    print(f"   python compare_topk_dual_results.py")
    
    return results


# Helper for a single threshold run
@app.function(
    timeout=18000
)
def start_single_topk(
    top_k: int = 3,
    total_timesteps: int = 1000000,
    add_timestamp: bool = True,
    model_choice: str = "model_b"  # Selectable policy variant.
):
    """Launch a single threshold run for quick verification.

    Args:
        model_choice: selectable policy variant (default: model_b)
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    print(f"🧪 Testing single threshold={top_k} ({model_choice.upper()})")
    
    result = train_with_topk_dual.remote(
        top_k=top_k,
        total_timesteps=total_timesteps,
        run_id=run_id,
        model_choice=model_choice
    )
    
    print(f"\n✅ Run complete: Threshold={top_k} ({model_choice.upper()})")
    return result

