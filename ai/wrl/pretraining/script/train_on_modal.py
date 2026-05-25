# scripts/train_on_modal.py
"""Script to train the encoder on Modal with GPU.

Workflow:
1. Upload data (one-time):
    modal volume put vhas-training-data data/wrl_pretraining_examples.json /data/wrl_pretraining_examples.json

2. Train:
    modal run train_on_modal.py

3. Download model:
    modal volume get vhas-encoder-output pretrained_encoder ../output/pretrained_encoder
"""
import modal
import os

# --- 1. Define environment ---
app = modal.App("vhas-encoder-pretraining")

# Build image with required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",  # Required by sentence-transformers
        "accelerate>=0.26.0"  # Required by transformers Trainer
    )
)

# --- 2. Define persistent Volumes ---
data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=True)
output_vol = modal.Volume.from_name("vhas-encoder-output", create_if_missing=True)

# --- 3. Training Function ---
@app.function(
    image=image,
    gpu="T4",
    retries=10,  # Increase retries to improve robustness against preemptions
    volumes={
        "/data": data_vol,
        "/output": output_vol
    },
    timeout=7200,
)
def train_encoder_on_modal(
    batch_size: int = 128,
    num_epochs: int = 1,
    data_path: str = "/data/wrl_pretraining_examples.json"
):
    """Training function that runs on Modal GPU.

    Reads data directly from the Volume (do not pass the data via arguments).

    Args:
        batch_size: training batch size
        num_epochs: number of epochs
        data_path: path to training data inside the volume
    """
    import json
    import torch
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("🚀 Starting VHAS Encoder Pre-training on Modal GPU")
    print("=" * 70)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Debug: list files in /data to see what's available
    print(f"\n🔍 Checking volume contents at /data:")
    import os
    if os.path.exists("/data"):
        files = os.listdir("/data")
        print(f"   Files found: {files}")
    else:
        print(f"   ⚠️  /data directory does not exist!")
    
    # Load training data from Volume
    print(f"\n📥 Loading training data from volume: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Please upload it first using: modal volume put vhas-training-data <local_file> {data_path}"
        )
    
    with open(data_path, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
    
    training_examples = [InputExample(texts=ex['texts']) for ex in examples_data]
    print(f"   ✓ Loaded {len(training_examples):,} training pairs")
    
    # Load base model
    print("\n🔧 Loading base model 'all-mpnet-base-v2'...")
    encoder_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    print(f"   ✓ Model loaded with {encoder_model.get_sentence_embedding_dimension()} dimensions")
    
    # Setup training
    train_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )
    
    train_loss = losses.MultipleNegativesRankingLoss(model=encoder_model)
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    
    print("\n📊 Training Configuration:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Steps per epoch: {len(train_dataloader):,}")
    print(f"   • Total steps: {len(train_dataloader) * num_epochs:,}")
    print(f"   • Warmup steps: {warmup_steps:,}")
    print(f"   • Loss function: MultipleNegativesRankingLoss")
    
    # Save config
    output_path = "/output/pretrained_encoder"
    os.makedirs(output_path, exist_ok=True)
    
    config = {
        "base_model": "all-mpnet-base-v2",
        "loss": "MultipleNegativesRankingLoss",
        "batch_size": batch_size,
        "epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "device": device,
        "num_training_examples": len(training_examples),
        "total_steps": len(train_dataloader) * num_epochs,
        "trained_on": "Modal GPU"
    }
    
    config_path = os.path.join(output_path, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Config saved to {config_path}")
    
    # Define checkpoint directory - SentenceTransformer.fit() will auto-handle resume
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    
    # Train
    print("\n" + "=" * 70)
    print("🏋️  Starting Training...")
    print("=" * 70 + "\n")
    
    encoder_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_path=checkpoint_dir,
        checkpoint_save_steps=200,  # Save checkpoint every 200 steps
        checkpoint_save_total_limit=3  # Keep only the 3 most recent checkpoints
    )
    
    # Commit volume to persist outputs
    output_vol.commit()
    
    print("\n" + "=" * 70)
    print(f"✅ Training Complete! Model saved to {output_path}")
    print("=" * 70)
    
    return {
        "status": "success",
        "output_path": output_path,
        "num_examples": len(training_examples),
        "config": config
    }


@app.local_entrypoint()
def main(
    batch_size: int = 128,
    epochs: int = 1
):
    """Entry point to run the training job.

    ⚠️  DATA MUST BE UPLOADED FIRST via CLI:
    modal volume put vhas-training-data data/wrl_pretraining_examples.json /data/wrl_pretraining_examples.json

    Args:
        batch_size: training batch size
        epochs: number of epochs
    """
    # Uploaded file path inside the volume: /data/wrl_pretraining_examples.json
    # When mounted, access it as /data/data/wrl_pretraining_examples.json inside the container
    remote_data_path = "/data/data/wrl_pretraining_examples.json"
    
    print("=" * 70)
    print("🚀 VHAS Encoder Training on Modal")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"   • Data path: {remote_data_path}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Epochs: {epochs}")
    print(f"   • GPU: T4")
    
    print(f"\n⚠️  Make sure data is uploaded to the volume!")
    print(f"   If not, run:")
    print(f"   modal volume put vhas-training-data <local_file> {remote_data_path}")
    
    print(f"\n🌐 Submitting training job to Modal...\n")
    
    # Launch training - do NOT pass data directly
    result = train_encoder_on_modal.remote(
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
    print(f"   Output: {result['output_path']}")
    
    print(f"\n💾 Download trained model:")
    print(f"   modal volume get vhas-encoder-output pretrained_encoder ../output/pretrained_encoder")
    print(f"\n💡 View all files in volume:")
    print(f"   modal volume ls vhas-encoder-output")
