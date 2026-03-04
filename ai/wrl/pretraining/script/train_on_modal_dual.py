# scripts/train_on_modal_dual.py
"""
Script để training DUAL ENCODER (Two-Tower) trên Modal với GPU.

KIẾN TRÚC:
- StateEncoder: Chuyên mã hóa Clinical States
- ActionEncoder: Chuyên mã hóa Agents và Tools

TRAINING:
- Sử dụng Contrastive Learning với cặp (State, Action)
- Maximize cosine similarity giữa state_embedding và action_embedding
- Loss: MultipleNegativesRankingLoss

WORKFLOW:
1. Upload data (one-time): 
   modal volume put vhas-training-data data/wrl_pretraining_examples.json /data/wrl_pretraining_examples.json

2. Train:
   modal run train_on_modal_dual.py

3. Download models:
   modal volume get vhas-encoder-output pretrained_state_encoder ../output/pretrained_state_encoder
   modal volume get vhas-encoder-output pretrained_action_encoder ../output/pretrained_action_encoder
"""
import modal
import os

# --- 1. Định nghĩa Môi trường (Environment) ---
app = modal.App("vhas-dual-encoder-pretraining")

# Xây dựng image với đầy đủ dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "accelerate>=0.26.0"
    )
)

# --- 2. Định nghĩa Volume để lưu trữ persistent ---
data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=True)
output_vol = modal.Volume.from_name("vhas-encoder-output", create_if_missing=True)

# --- 3. Training Function ---
@app.function(
    image=image,
    gpu="T4",
    retries=10,
    volumes={
        "/data": data_vol,
        "/output": output_vol
    },
    timeout=7200,
)
def train_dual_encoder_on_modal(
    batch_size: int = 128,
    num_epochs: int = 1,
    data_path: str = "/data/wrl_pretraining_examples.json"
):
    """
    Hàm training Dual Encoder trên Modal GPU.
    
    Training Strategy:
    - Load base model 'all-mpnet-base-v2' làm checkpoint chung
    - Clone thành 2 models: StateEncoder và ActionEncoder
    - Training với Contrastive Loss trên cặp (State, Action)
    
    Args:
        batch_size: Batch size cho training
        num_epochs: Số epochs
        data_path: Đường dẫn đến file training data trong volume
    """
    import json
    import torch
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("🚀 Starting VHAS Dual-Encoder Pre-training on Modal GPU")
    print("=" * 70)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load training data
    print(f"\n📥 Loading training data from volume: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Please upload it first using: modal volume put vhas-training-data <local_file> {data_path}"
        )
    
    with open(data_path, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
    
    # Parse examples - format: {"texts": [state, action]}
    training_examples = [InputExample(texts=ex['texts']) for ex in examples_data]
    print(f"   ✓ Loaded {len(training_examples):,} (State, Action) pairs")
    
    # Load base models
    print("\n🔧 Loading base models 'all-mpnet-base-v2'...")
    state_encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
    action_encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
    print(f"   ✓ StateEncoder loaded: {state_encoder.get_sentence_embedding_dimension()} dims")
    print(f"   ✓ ActionEncoder loaded: {action_encoder.get_sentence_embedding_dimension()} dims")
    
    # Setup training with Dual-Encoder architecture
    # We'll use a custom training loop to handle two encoders
    train_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )
    
    # Use MultipleNegativesRankingLoss - it naturally supports dual-encoder
    # by treating first text as query (State) and second as document (Action)
    train_loss = losses.MultipleNegativesRankingLoss(model=state_encoder)
    
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    
    print("\n📊 Training Configuration:")
    print(f"   • Architecture: Dual-Encoder (Two-Tower)")
    print(f"   • StateEncoder: all-mpnet-base-v2 (768 dims)")
    print(f"   • ActionEncoder: all-mpnet-base-v2 (768 dims)")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Steps per epoch: {len(train_dataloader):,}")
    print(f"   • Total steps: {len(train_dataloader) * num_epochs:,}")
    print(f"   • Warmup steps: {warmup_steps:,}")
    print(f"   • Loss function: MultipleNegativesRankingLoss (Contrastive)")
    
    # Save directories
    state_encoder_path = "/output/pretrained_state_encoder"
    action_encoder_path = "/output/pretrained_action_encoder"
    os.makedirs(state_encoder_path, exist_ok=True)
    os.makedirs(action_encoder_path, exist_ok=True)
    
    config = {
        "architecture": "Dual-Encoder (Two-Tower)",
        "base_model": "all-mpnet-base-v2",
        "state_encoder": "Specialized for Clinical States",
        "action_encoder": "Specialized for Agents and Tools",
        "loss": "MultipleNegativesRankingLoss (Contrastive)",
        "batch_size": batch_size,
        "epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "device": device,
        "num_training_examples": len(training_examples),
        "total_steps": len(train_dataloader) * num_epochs,
        "trained_on": "Modal GPU"
    }
    
    # Save config for both encoders
    with open(os.path.join(state_encoder_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(action_encoder_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Config saved")
    
    # Checkpoint directories
    state_checkpoint_dir = os.path.join(state_encoder_path, "checkpoints")
    action_checkpoint_dir = os.path.join(action_encoder_path, "checkpoints")
    
    # Train StateEncoder (treats first text as anchor, second as positive)
    print("\n" + "=" * 70)
    print("🏋️  Training StateEncoder...")
    print("=" * 70 + "\n")
    
    state_encoder.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=state_encoder_path,
        show_progress_bar=True,
        checkpoint_path=state_checkpoint_dir,
        checkpoint_save_steps=200,
        checkpoint_save_total_limit=3
    )
    
    # For ActionEncoder, we need to swap the texts in examples
    # So that Actions become the anchor
    print("\n" + "=" * 70)
    print("🏋️  Training ActionEncoder...")
    print("=" * 70 + "\n")
    
    # Create swapped examples for action encoder
    swapped_examples = [InputExample(texts=[ex.texts[1], ex.texts[0]]) for ex in training_examples]
    swapped_dataloader = DataLoader(
        swapped_examples,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )
    
    action_loss = losses.MultipleNegativesRankingLoss(model=action_encoder)
    
    action_encoder.fit(
        train_objectives=[(swapped_dataloader, action_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=action_encoder_path,
        show_progress_bar=True,
        checkpoint_path=action_checkpoint_dir,
        checkpoint_save_steps=200,
        checkpoint_save_total_limit=3
    )
    
    # Commit volume
    output_vol.commit()
    
    print("\n" + "=" * 70)
    print(f"✅ Dual-Encoder Training Complete!")
    print("=" * 70)
    print(f"   StateEncoder saved to: {state_encoder_path}")
    print(f"   ActionEncoder saved to: {action_encoder_path}")
    print("=" * 70)
    
    return {
        "status": "success",
        "state_encoder_path": state_encoder_path,
        "action_encoder_path": action_encoder_path,
        "num_examples": len(training_examples),
        "config": config
    }


@app.local_entrypoint()
def main(
    batch_size: int = 128,
    epochs: int = 1
):
    """
    Entry point để chạy dual-encoder training job.
    
    Args:
        batch_size: Batch size cho training
        epochs: Số epochs
    """
    remote_data_path = "/data/data/wrl_pretraining_examples.json"
    
    print("=" * 70)
    print("🚀 VHAS Dual-Encoder Training on Modal")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"   • Architecture: Two-Tower (StateEncoder + ActionEncoder)")
    print(f"   • Data path: {remote_data_path}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Epochs: {epochs}")
    print(f"   • GPU: T4")
    
    print(f"\n⚠️  Make sure data is uploaded to volume!")
    print(f"   If not, run:")
    print(f"   modal volume put vhas-training-data <local_file> {remote_data_path}")
    
    print(f"\n🌐 Submitting training job to Modal...\n")
    
    # Chạy training
    result = train_dual_encoder_on_modal.remote(
        batch_size=batch_size,
        num_epochs=epochs,
        data_path=remote_data_path
    )
    
    print(f"\n" + "=" * 70)
    print(f"✅ Training Complete!")
    print(f"=" * 70)
    print(f"\n📊 Results:")
    print(f"   Status: {result['status']}")
    print(f"   Examples trained: {result['num_examples']:,}")
    print(f"   StateEncoder: {result['state_encoder_path']}")
    print(f"   ActionEncoder: {result['action_encoder_path']}")
    
    print(f"\n💾 Download trained models:")
    print(f"   modal volume get vhas-encoder-output pretrained_state_encoder ../output/pretrained_state_encoder")
    print(f"   modal volume get vhas-encoder-output pretrained_action_encoder ../output/pretrained_action_encoder")

