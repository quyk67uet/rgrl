# train_gnn_deployed.py
"""
Train VHAS Orchestrator with a GNN policy + RSL-RL on Modal.

DEPLOYED MODE - runs in the background and keeps going after disconnect.

USAGE:
    # Test with ~200k timesteps (default: 1024 steps/env * 4 envs * 50 iters)
    modal run --detach train_gnn_deployed.py::start_training --total-timesteps 200000

    # Full 1M-timestep run
    modal run --detach train_gnn_deployed.py::start_training --total-timesteps 1000000

After this, you can safely shut down your machine.
Training runs on Modal Cloud.

Check logs:
    modal app logs vhas-orchestrator-gnn --follow

Download results:
    modal volume ls vhas-training-results
    modal volume get vhas-training-results gnn_<TIMESTAMP>/ ./results_gnn/
"""

import modal
from datetime import datetime
from pathlib import Path

app = modal.App("vhas-orchestrator-gnn")

# Image with PyTorch, PyG, and RSL-RL deps
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Core ML/AI libs
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "torchvision>=0.5.0",      # RSL-RL dep
        "torch-geometric",          # GNN backbone
        "tensordict>=0.7.0",        # RSL-RL needs >=0.7.0
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        
        # RSL-RL deps
        "gitpython>=3.1.0",
        "tensorboard>=2.14.0",      # Required by RSL-RL logger
        "onnx>=1.0.0",              # RSL-RL dep (optional)
        "onnxscript>=0.5.4",        # RSL-RL dep (optional)
    )
    # Disable GitPython git check for minimal containers
    .env({"GIT_PYTHON_REFRESH": "quiet"})
)

# Volumes - mount data, models, and results
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=False)
finetuned_output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=False)
training_results_vol = modal.Volume.from_name("vhas-training-results", create_if_missing=True)


def _resolve_stage2_policies_root() -> Path:
    """Return the stage-2 policy artifact root for Modal or local execution."""
    modal_models_root = Path("/models")
    if modal_models_root.exists():
        return modal_models_root / "stage2_gro_policies"
    return Path(__file__).resolve().parents[2] / "models" / "stage2_gro_policies"


CANONICAL_GNN_CHECKPOINT = _resolve_stage2_policies_root() / "vhas_gnn_afan_best.pt"


@app.function(
    image=image,
    gpu="A10G",        # GNN benefits from GPU
    memory=16384,      # 16GB RAM
    timeout=14400,     # 4 hours
    volumes={
        "/data": training_data_vol,
        "/models": finetuned_output_vol,
        "/results": training_results_vol,
    },
)
def train_gnn(
    total_timesteps: int = 200_000,   # Total env steps, converted to iters
    num_envs: int = 4,
    num_steps_per_env: int = 256,  # Reduced from 1024 for faster tests
    use_guidance: bool = False,
    top_k_guidance: int = 5,
    run_id: str | None = None,
):
    """
    Train VHAS Orchestrator with a GNN policy + RSL-RL.

    Runs in the background and survives disconnects.
    """
    import sys
    import math
    import os
    import importlib.util

    # Add paths for GNN code and RSL-RL from the mounted volume
    # Note: The volume is mounted at /data, with data/gro/gnn/ and data/rsl_rl/rsl_rl/
    # So:
    #   - GNN code:   /data/data/gro/gnn
    #   - rsl_rl pkg: /data/data/rsl_rl  (contains rsl_rl/)
    gnn_path = "/data/data/gro/gnn"
    rsl_path = "/data/data/rsl_rl"
    
    sys.path.insert(0, gnn_path)
    sys.path.insert(0, rsl_path)
    
    # Debug: check whether the file exists
    train_file = os.path.join(gnn_path, "train.py")
    print(f"\n🔍 Checking paths...")
    print(f"   GNN path: {gnn_path}")
    print(f"   RSL-RL path: {rsl_path}")
    print(f"   Train file: {train_file}")
    print(f"   Train file exists: {os.path.exists(train_file)}")
    
    if not os.path.exists(train_file):
        # Try another RSL-RL path if needed
        rsl_path_alt = "/data/data/rsl_rl"
        if os.path.exists(rsl_path_alt):
            sys.path.insert(0, rsl_path_alt)
            print(f"   Found RSL-RL at: {rsl_path_alt}")
        
        # List files for debugging
        if os.path.exists(gnn_path):
            files = os.listdir(gnn_path)
            print(f"   Files in {gnn_path}: {files}")
        else:
            # Try fallback path
            gnn_path_alt = "/data/gro/gnn"
            if os.path.exists(gnn_path_alt):
                print(f"   Trying alternative path: {gnn_path_alt}")
                files = os.listdir(gnn_path_alt)
                print(f"   Files in {gnn_path_alt}: {files}")
                gnn_path = gnn_path_alt
                train_file = os.path.join(gnn_path, "train.py")
                sys.path.insert(0, gnn_path)
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(
                f"train.py not found at {train_file}. "
                f"Please ensure files are uploaded to Modal volume using upload_gnn_to_modal.ps1"
            )
    
    # Import the training function with importlib
    spec = importlib.util.spec_from_file_location("train", train_file)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    train_vhas_gnn = train_module.train_vhas_gnn

    print("=" * 80)
    print("🚀 VHAS ORCHESTRATOR TRAINING - GNN POLICY (RSL-RL)")
    print("=" * 80)
    print(f"Policy: GNN (AFAN, graph-structured)")
    print(f"Total timesteps (target): {total_timesteps:,}")
    print(f"num_envs: {num_envs}")
    print(f"num_steps_per_env: {num_steps_per_env}")
    print(f"Guidance: {'ENABLED' if use_guidance else 'DISABLED'} (top-k={top_k_guidance})")
    print(f"Device: A10G GPU")
    print(f"Memory: 16GB RAM")
    print(f"🌙 DEPLOYED MODE: Training will continue even if you disconnect")
    print("=" * 80)

    # Paths in the Modal volume
    encoder_path = "/models/model_a"          # Finetuned encoder
    traces_filepath = "/data/workflow_traces/train_traces.json"
    kb_path = "/data/simulation_kb.json"

    # Convert total_timesteps -> max_iterations for RSL-RL
    # Each iteration collects num_envs * num_steps_per_env steps
    steps_per_iteration = num_envs * num_steps_per_env
    max_iterations = max(1, math.floor(total_timesteps / steps_per_iteration))

    effective_timesteps = max_iterations * steps_per_iteration

    print(f"\n📊 Derived training schedule:")
    print(f"   • steps_per_iteration = num_envs * num_steps_per_env = {steps_per_iteration:,}")
    print(f"   • max_iterations = floor(total_timesteps / steps_per_iteration) = {max_iterations}")
    print(f"   • effective_timesteps = {effective_timesteps:,}")

    # Create a run-specific output dir
    output_stub = run_id or "gnn"
    log_dir = f"/results/gnn_{output_stub}"

    print(f"\n📁 Log & checkpoint directory: {log_dir}")
    print(f"📦 Canonical checkpoint path: {CANONICAL_GNN_CHECKPOINT}")

    try:
        runner = train_vhas_gnn(
            # Data
            encoder_path=encoder_path,
            traces_filepath=traces_filepath,
            kb_path=kb_path,
            guidance_encoder_path=None,               # Can be re-enabled later
            guidance_embedding_space_path=None,
            # Env / rollout
            num_envs=num_envs,
            num_steps_per_env=num_steps_per_env,
            max_iterations=max_iterations,
            # PPO hyperparams (use train_vhas_gnn defaults)
            use_guidance=use_guidance,
            top_k_guidance=top_k_guidance,
            # Logging
            log_dir=log_dir,
            experiment_name=f"vhas_gnn_{output_stub}",
            device="cuda",
        )

        # Commit results
        training_results_vol.commit()

        print("\n" + "=" * 80)
        print("✅ Training Complete - GNN POLICY (RSL-RL)")
        print("=" * 80)
        print(f"Runner final iteration: {runner.current_learning_iteration if hasattr(runner, 'current_learning_iteration') else 'N/A'}")

        return {
            "model": "gnn",
            "status": "success",
            "effective_timesteps": effective_timesteps,
            "max_iterations": max_iterations,
            "num_envs": num_envs,
            "num_steps_per_env": num_steps_per_env,
            "log_dir": log_dir,
            "final_checkpoint_path": str(CANONICAL_GNN_CHECKPOINT),
        }

    except Exception as e:
        import traceback

        print(f"\n❌ Training failed for GNN: {e}")
        traceback.print_exc()

        return {
            "model": "gnn",
            "status": "failed",
            "error": str(e),
            "log_dir": log_dir,
            "final_checkpoint_path": str(CANONICAL_GNN_CHECKPOINT),
        }


@app.function(
    image=image,  # Use the same image to avoid deserialization mismatches
    timeout=18000,
)
def start_training(
    total_timesteps: int = 200_000,
    num_envs: int = 4,
    num_steps_per_env: int = 1024,
    use_guidance: bool = False,
    top_k_guidance: int = 5,
    add_timestamp: bool = True,
):
    """
    Convenience entry point to start GNN training.
    You can shut down the machine after calling this.
    """
    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None

    print("=" * 80)
    print("🚀 Starting GNN Policy Training (Background Mode)")
    print("=" * 80)
    print("\n📊 Configuration:")
    print(f"   • Policy: GNN (graph-based AFAN)")
    print(f"   • Timesteps (target): {total_timesteps:,}")
    print(f"   • num_envs: {num_envs}")
    print(f"   • num_steps_per_env: {num_steps_per_env}")
    print(f"   • Guidance: {'ENABLED' if use_guidance else 'DISABLED'} (top-k={top_k_guidance})")
    print(f"   • GPU: A10G")
    print(f"   • Memory: 16GB RAM")
    print(f"   • Run ID: {run_id or 'None (will overwrite existing)'}")

    if total_timesteps <= 100_000:
        print(f"\n🧪 TEST MODE: Running with {total_timesteps:,} timesteps to verify pipeline")
    else:
        print(f"\n🎯 FULL TRAINING: Running with {total_timesteps:,} timesteps")

    print("\n🌙 BACKGROUND MODE:")
    print("   • Training will continue even if you disconnect")
    print("   • You can safely close terminal or shutdown computer")
    print("   • Check progress later with: modal app logs vhas-orchestrator-gnn")

    print("\n🚀 Starting training...\n")

    result = train_gnn.remote(
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        num_steps_per_env=num_steps_per_env,
        use_guidance=use_guidance,
        top_k_guidance=top_k_guidance,
        run_id=run_id,
    )

    print("\n" + "=" * 80)
    print("✅ GNN TRAINING COMPLETED (REMOTE CALL RETURNED)!")
    print("=" * 80)

    if result["status"] == "success":
        print("\n📊 Results:")
        print(f"   Status: {result['status']}")
        print(f"   Effective timesteps: {result['effective_timesteps']:,}")
        print(f"   max_iterations: {result['max_iterations']}")
        print(f"   num_envs: {result['num_envs']}")
        print(f"   num_steps_per_env: {result['num_steps_per_env']}")
        print(f"   Log dir: {result['log_dir']}")
        if total_timesteps <= 100_000:
            print("\n✅ Pipeline verification successful!")
            print("   You can now run full training with --total-timesteps 1000000")
    else:
        print(f"\n❌ Training failed: {result.get('error', 'Unknown')}")

    print("\n💾 Download trained model & logs:")
    print(f"   modal volume get vhas-training-results {result['log_dir'].replace('/results/', '')} ./results_gnn/")

    return result


# Entry point for CLI
if __name__ == "__main__":
    with app.run():
        start_training.remote(
            total_timesteps=200_000,
            num_envs=2,
            num_steps_per_env=256,  # Reduced for faster tests
            use_guidance=False,
        )

