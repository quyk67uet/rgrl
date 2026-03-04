# train_gnn_deployed.py
"""
Training VHAS Orchestrator với GNN Policy + RSL-RL trên Modal.

DEPLOYED MODE - chạy background không bị cancel khi tắt máy.

USAGE:
   # Test với ~200k timesteps (mặc định: 1024 steps/env * 4 envs * 50 iterations)
   modal run --detach train_gnn_deployed.py::start_training --total-timesteps 200000
   
   # Ví dụ full training 1M timesteps
   modal run --detach train_gnn_deployed.py::start_training --total-timesteps 1000000

Sau khi chạy lệnh trên, BẠN CÓ THỂ TẮT MÁY NGAY.
Training sẽ chạy trên Modal cloud.

Kiểm tra logs:
   modal app logs vhas-orchestrator-gnn --follow

Download kết quả:
   modal volume ls vhas-training-results
   modal volume get vhas-training-results gnn_<TIMESTAMP>/ ./results_gnn/
"""

import modal
from datetime import datetime

app = modal.App("vhas-orchestrator-gnn")

# Image với PyTorch, PyG, RSL-RL dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Core ML/AI libraries
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "torchvision>=0.5.0",      # RSL-RL dependency
        "torch-geometric",          # GNN backbone
        "tensordict>=0.7.0",        # RSL-RL requires >=0.7.0
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        
        # RSL-RL dependencies
        "gitpython>=3.1.0",
        "tensorboard>=2.14.0",      # Required by RSL-RL Logger
        "onnx>=1.0.0",              # RSL-RL dependency (optional but recommended)
        "onnxscript>=0.5.4",        # RSL-RL dependency (optional but recommended)
    )
    # Tắt check git executable của GitPython để tránh lỗi trong container tối giản
    .env({"GIT_PYTHON_REFRESH": "quiet"})
)

# Volumes - mount data, models và results
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=False)
finetuned_output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=False)
training_results_vol = modal.Volume.from_name("vhas-training-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",        # GNN cũng hưởng lợi từ GPU
    memory=16384,      # 16GB RAM
    timeout=14400,     # 4 giờ
    volumes={
        "/data": training_data_vol,
        "/models": finetuned_output_vol,
        "/results": training_results_vol,
    },
)
def train_gnn(
    total_timesteps: int = 200_000,   # số bước env tổng, sẽ convert sang iterations
    num_envs: int = 4,
    num_steps_per_env: int = 256,  # Reduced from 1024 for faster iteration testing
    use_guidance: bool = False,
    top_k_guidance: int = 5,
    run_id: str | None = None,
):
    """
    Train VHAS Orchestrator với GNN Policy + RSL-RL.

    Chạy trong background, không bị cancel khi client disconnect.
    """
    import sys
    import math
    import os
    import importlib.util

    # Thêm path tới code GNN và RSL-RL được mount từ volume
    # Note: Volume mount tại /data, nhưng bên trong volume có cấu trúc data/gro/gnn/ và data/rsl_rl/rsl_rl/
    # Vậy nên:
    #   - GNN code:   /data/data/gro/gnn
    #   - rsl_rl pkg: /data/data/rsl_rl  (bên trong có thư mục con rsl_rl/)
    gnn_path = "/data/data/gro/gnn"
    rsl_path = "/data/data/rsl_rl"
    
    sys.path.insert(0, gnn_path)
    sys.path.insert(0, rsl_path)
    
    # Debug: Kiểm tra file có tồn tại không
    train_file = os.path.join(gnn_path, "train.py")
    print(f"\n🔍 Checking paths...")
    print(f"   GNN path: {gnn_path}")
    print(f"   RSL-RL path: {rsl_path}")
    print(f"   Train file: {train_file}")
    print(f"   Train file exists: {os.path.exists(train_file)}")
    
    if not os.path.exists(train_file):
        # Thử path khác cho RSL-RL nếu cần
        rsl_path_alt = "/data/data/rsl_rl"
        if os.path.exists(rsl_path_alt):
            sys.path.insert(0, rsl_path_alt)
            print(f"   Found RSL-RL at: {rsl_path_alt}")
        
        # List files in directory for debugging
        if os.path.exists(gnn_path):
            files = os.listdir(gnn_path)
            print(f"   Files in {gnn_path}: {files}")
        else:
            # Thử path cũ (fallback)
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
    
    # Import training function - sử dụng importlib để đảm bảo import đúng
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

    # Path trong Modal volume
    encoder_path = "/models/model_a"          # Finetuned encoder (giống transformer)
    scenarios_data_dir = "/data/scenarios/data"
    kb_path = "/data/simulation_kb.json"

    # Convert total_timesteps -> max_iterations cho RSL-RL
    # Mỗi iteration thu thập: num_envs * num_steps_per_env steps
    steps_per_iteration = num_envs * num_steps_per_env
    max_iterations = max(1, math.floor(total_timesteps / steps_per_iteration))

    effective_timesteps = max_iterations * steps_per_iteration

    print(f"\n📊 Derived training schedule:")
    print(f"   • steps_per_iteration = num_envs * num_steps_per_env = {steps_per_iteration:,}")
    print(f"   • max_iterations = floor(total_timesteps / steps_per_iteration) = {max_iterations}")
    print(f"   • effective_timesteps = {effective_timesteps:,}")

    # Tạo output dir riêng theo run_id
    output_stub = run_id or "gnn"
    log_dir = f"/results/gnn_{output_stub}"

    print(f"\n📁 Log & checkpoint directory: {log_dir}")

    try:
        runner = train_vhas_gnn(
            # Data
            encoder_path=encoder_path,
            scenarios_data_dir=scenarios_data_dir,
            kb_path=kb_path,
            guidance_encoder_path=None,               # Có thể bật lại sau
            guidance_embedding_space_path=None,
            # Env / rollout
            num_envs=num_envs,
            num_steps_per_env=num_steps_per_env,
            max_iterations=max_iterations,
            # PPO hyperparams (sử dụng default trong train_vhas_gnn)
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
        }


@app.function(
    image=image,  # Cùng image để tránh lỗi deserialization (numpy, torch, ... giống nhau)
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
    Entry-point tiện dụng để trigger training GNN Policy.
    Có thể tắt máy sau khi gọi function này.
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
            num_steps_per_env=256,  # Reduced for faster testing
            use_guidance=False,
        )

