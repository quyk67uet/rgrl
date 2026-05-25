# train_model_b_dual_deployed.py
"""
Training script for Model B with Dual-Encoder Guidance on Modal.

Model B: Base → Pre-train → Fine-tune with Dual-Encoder (comprehensive).
"""
import modal
import sys
from datetime import datetime

# --- Modal Setup ---
app = modal.App("vhas-train-model-b-dual")

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

# Mount volumes
volume_mounts = {
    "/data": training_data_vol,
    "/models": finetuned_output_vol,
    "/results": training_results_vol
}

@app.function(
    image=image,
    gpu="T4",
    volumes=volume_mounts,
    timeout=18000  # 5 hours
)
def train_model_b_dual(
    total_timesteps: int = 1000000,
    run_id: str = None
):
    """
    Training function for Model B using Dual-Encoder Guidance.
    """
    import os
    import sys
    
    # Add repository code path (files are mounted at /data/data)
    sys.path.insert(0, '/data/data')
    
    from train_orchestrator_sb3 import train_orchestrator_baseline
    
    print("\n" + "="*80)
    print("🚀 TRAINING MODEL B WITH DUAL-ENCODER GUIDANCE")
    print("="*80)
    print(f"   • Model: Model B (Base → Pre-train → Fine-tune)")
    print(f"   • Architecture: Dual-Encoder (Two-Tower)")
    print(f"   • StateEncoder: model_b_state_encoder")
    print(f"   • ActionEncoder: model_b_action_encoder")
    print(f"   • Total timesteps: {total_timesteps:,}")
    print(f"   • Run ID: {run_id}")
    print("="*80 + "\n")
    
    # Set paths for Dual-Encoder
    state_encoder_path = "/models/model_b_state_encoder"
    action_encoder_path = "/models/model_b_action_encoder"
    embedding_space_path = "/models/embedding_space_model_b_dual"
    scenarios_data_dir = "/data/scenarios/data"
    kb_path = "/data/simulation_kb.json"
    
    # Output directory
    if run_id:
        output_dir = f"/results/orchestrator_model_b_dual_{run_id}"
    else:
        output_dir = f"/results/orchestrator_model_b_dual"
    
    # Train
    model, stats = train_orchestrator_baseline(
        encoder_path=state_encoder_path,
        embedding_space_path=embedding_space_path,
        action_encoder_path=action_encoder_path,  # NEW: Pass action encoder
        use_dual_encoder=True,  # NEW: Enable dual-encoder mode
        scenarios_data_dir=scenarios_data_dir,
        kb_path=kb_path,
        output_dir=output_dir,
        total_timesteps=total_timesteps,
        top_k_guidance=5,
        device='cuda'
    )
    
    # Commit results
    training_results_vol.commit()
    
    print("\n" + "="*80)
    print("✅ MODEL B DUAL-ENCODER TRAINING COMPLETE!")
    print("="*80)
    print(f"   Output: {output_dir}")
    print("="*80 + "\n")
    
    return {
        "status": "success",
        "model": "B",
        "architecture": "Dual-Encoder",
        "output_dir": output_dir,
        "stats": stats
    }


@app.local_entrypoint()
def start_training(
    total_timesteps: int = 1000000,
    add_timestamp: bool = True
):
    """
    Local entrypoint to start the training job.
    """
    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    
    print("\n" + "="*80)
    print("🚀 STARTING MODEL B DUAL-ENCODER TRAINING ON MODAL")
    print("="*80)
    print(f"   • Model: B (Toàn diện)")
    print(f"   • Architecture: Dual-Encoder")
    print(f"   • Total timesteps: {total_timesteps:,}")
    print(f"   • Run ID: {run_id}")
    print(f"   • GPU: T4")
    print(f"   • Timeout: 5 hours")
    print("="*80 + "\n")
    
    # Submit job
    result = train_model_b_dual.remote(
        total_timesteps=total_timesteps,
        run_id=run_id
    )
    
    print("\n" + "="*80)
    print("✅ TRAINING JOB COMPLETE!")
    print("="*80)
    print(f"   Status: {result['status']}")
    print(f"   Output: {result['output_dir']}")
    print("="*80 + "\n")

