# scripts/run_finetuning_jobs.py
"""Run encoder fine-tuning on Modal for two ablation-study models:
- Model A: Base → Fine-tune (Domain-specific)
- Model B: Base → Pre-train → Fine-tune (Comprehensive)

Workflow:
1. Upload finetuning data:
    modal volume put vhas-training-data data/wrl_finetuning_examples.json /finetuning/wrl_finetuning_examples.json

2. Run both jobs:
    modal run run_finetuning_jobs.py

3. Download models:
    modal volume get vhas-finetuned-output model_a ../output/model_a
    modal volume get vhas-finetuned-output model_b ../output/model_b
"""
import modal
import os

# --- 1. Define environment ---
app = modal.App("vhas-encoder-finetuning-ablation")

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

# --- 2. Define Volumes ---
# Share training-data volume to save storage
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=True)
pretrained_model_vol = modal.Volume.from_name("vhas-encoder-output", create_if_missing=True)
output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=True)

# --- 3. Model A: Fine-tune from Base Model (Domain-specific) ---
@app.function(
    image=image,
    gpu="T4",
    retries=5,
    volumes={
        "/data": training_data_vol,
        "/output": output_vol
    },
    timeout=3600,
)
def finetune_model_a(
    batch_size: int = 32,
    num_epochs: int = 3,
    data_path: str = "/data/finetuning/wrl_finetuning_examples.json"
):
    """Model A (Domain-specific): Base Model → Fine-tune.

    Train directly on clinical traces without general pretraining.
    """
    import json
    import torch
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("🔬 Model A: Fine-tuning from BASE MODEL (Domain-specific)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load finetuning data
    print(f"\n📥 Loading finetuning data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
    
    training_examples = [InputExample(texts=ex['texts']) for ex in examples_data]
    print(f"   ✓ Loaded {len(training_examples):,} training pairs")
    
    # Load BASE model
    print("\n🔧 Loading BASE model 'all-mpnet-base-v2'...")
    encoder_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    print(f"   ✓ Model loaded")
    
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
    print(f"   • Warmup steps: {warmup_steps:,}")
    
    output_path = "/output/model_a"
    os.makedirs(output_path, exist_ok=True)
    
    config = {
        "model_type": "Model A - Domain-specific",
        "base_model": "all-mpnet-base-v2",
        "pretraining": "None",
        "finetuning_data": "clinical_traces",
        "batch_size": batch_size,
        "epochs": num_epochs,
        "num_training_examples": len(training_examples)
    }
    
    with open(os.path.join(output_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    
    print("\n" + "=" * 70)
    print("🏋️  Starting Fine-tuning...")
    print("=" * 70 + "\n")
    
    encoder_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_path=checkpoint_dir,
        checkpoint_save_steps=100,
        checkpoint_save_total_limit=2
    )
    
    output_vol.commit()
    
    print("\n" + "=" * 70)
    print(f"✅ Model A Complete! Saved to {output_path}")
    print("=" * 70)
    
    return {"status": "success", "model": "A", "path": output_path}


# --- 4. Model B: Fine-tune from Pre-trained Model (Comprehensive) ---
@app.function(
    image=image,
    gpu="T4",
    retries=5,
    volumes={
        "/data": training_data_vol,
        "/pretrained": pretrained_model_vol,  # Volume chứa pretrained model
        "/output": output_vol
    },
    timeout=3600,
)
def finetune_model_b(
    batch_size: int = 32,
    num_epochs: int = 3,
    data_path: str = "/data/finetuning/wrl_finetuning_examples.json",
    pretrained_path: str = "/pretrained/pretrained_encoder"
):
    """Model B (Comprehensive): Base Model → Pre-train → Fine-tune.

    Uses general pretraining (ADP/T1) before clinical fine-tuning.
    """
    import json
    import torch
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("🌟 Model B: Fine-tuning from PRE-TRAINED MODEL (Comprehensive)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load finetuning data
    print(f"\n📥 Loading finetuning data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
    
    training_examples = [InputExample(texts=ex['texts']) for ex in examples_data]
    print(f"   ✓ Loaded {len(training_examples):,} training pairs")
    
    # Load PRE-TRAINED model
    print(f"\n🔧 Loading PRE-TRAINED model from: {pretrained_path}")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")
    
    encoder_model = SentenceTransformer(pretrained_path, device=device)
    print(f"   ✓ Pre-trained model loaded")
    
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
    print(f"   • Warmup steps: {warmup_steps:,}")
    
    output_path = "/output/model_b"
    os.makedirs(output_path, exist_ok=True)
    
    config = {
        "model_type": "Model B - Comprehensive",
        "base_model": "all-mpnet-base-v2",
        "pretraining": "ADP + T1 (343K pairs)",
        "finetuning_data": "clinical_traces (7K pairs)",
        "batch_size": batch_size,
        "epochs": num_epochs,
        "num_training_examples": len(training_examples)
    }
    
    with open(os.path.join(output_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    
    print("\n" + "=" * 70)
    print("🏋️  Starting Fine-tuning...")
    print("=" * 70 + "\n")
    
    encoder_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_path=checkpoint_dir,
        checkpoint_save_steps=100,
        checkpoint_save_total_limit=2
    )
    
    output_vol.commit()
    
    print("\n" + "=" * 70)
    print(f"✅ Model B Complete! Saved to {output_path}")
    print("=" * 70)
    
    return {"status": "success", "model": "B", "path": output_path}


@app.local_entrypoint()
def main():
    """Run both fine-tuning jobs in parallel."""
    print("=" * 70)
    print("🚀 VHAS Encoder Fine-tuning: Ablation Study")
    print("=" * 70)
    print("\nTwo Models:")
    print("   • Model A (Domain-specific): Base → Fine-tune")
    print("   • Model B (Comprehensive): Base → Pre-train → Fine-tune")

    print("\nEnsure finetuning data is uploaded:")
    print("   modal volume put vhas-training-data data/wrl_finetuning_examples.json /finetuning/wrl_finetuning_examples.json")

    print("\nSubmitting both jobs to Modal in parallel...\n")

    # Launch both jobs in parallel
    job_a = finetune_model_a.spawn()
    job_b = finetune_model_b.spawn()

    print("Jobs submitted:")
    print("   • Model A is running...")
    print("   • Model B is running...")

    # Wait for results
    result_a = job_a.get()
    result_b = job_b.get()

    print("\n" + "=" * 70)
    print("✅ All Fine-tuning Complete!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"   Model A: {result_a['status']} → {result_a['path']}")
    print(f"   Model B: {result_b['status']} → {result_b['path']}")

    print(f"\nDownload models:")
    print(f"   modal volume get vhas-finetuned-output model_a ../output/model_a")
    print(f"   modal volume get vhas-finetuned-output model_b ../output/model_b")