# scripts/run_finetuning_jobs_dual_corrected.py
"""
CORRECTED: Dual-Encoder Fine-tuning với logic đúng.

KIẾN TRÚC:
- ActionEncoder: Base → Pre-train (ADP+T1) → Fine-tune (Clinical)
- StateEncoder: Base → Fine-tune (Clinical) - SKIP pre-training

LÝ DO:
- ADP+T1 chỉ có (Action, Action) pairs → chỉ phù hợp cho ActionEncoder
- StateEncoder cần học từ Clinical States → chỉ có trong fine-tuning data

WORKFLOW:
1. Upload finetuning data:
   modal volume put vhas-training-data data/wrl_finetuning_examples.json /finetuning/wrl_finetuning_examples.json
   (Clinical states already included in wrl_finetuning_examples.json - no need to upload separately)

2. Run both jobs:
   modal run run_finetuning_jobs_dual_corrected.py

3. Download models:
   modal volume get vhas-finetuned-output model_a_state_encoder ../output/model_a_state_encoder
   modal volume get vhas-finetuned-output model_a_action_encoder ../output/model_a_action_encoder
   modal volume get vhas-finetuned-output model_b_state_encoder ../output/model_b_state_encoder
   modal volume get vhas-finetuned-output model_b_action_encoder ../output/model_b_action_encoder
"""
import modal
import os

# --- 1. Định nghĩa Môi trường ---
app = modal.App("vhas-dual-encoder-finetuning-corrected")

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

# --- 2. Định nghĩa Volumes ---
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=True)
pretrained_model_vol = modal.Volume.from_name("vhas-encoder-output", create_if_missing=True)
output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=True)

# --- 3. Model A: Fine-tune từ Base Model (Chuyên khoa) ---
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
def finetune_model_a_dual_corrected(
    batch_size: int = 32,
    num_epochs: int = 3,
    finetuning_data_path: str = "/data/finetuning/wrl_finetuning_examples.json"
):
    """
    Model A (Chuyên khoa): 
    - StateEncoder: Base → Fine-tune (State-Action pairs only)
    - ActionEncoder: Base → Fine-tune (State-Action pairs)
    
    NOTE: Clinical states already included in wrl_finetuning_examples.json
    """
    import json
    import torch
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("🔬 Model A: Fine-tuning DUAL-ENCODER from BASE MODEL (Domain-Specific)")
    print("=" * 70)
    print("CORRECTED LOGIC:")
    print("  - StateEncoder: Base → Fine-tune (Workflow)")
    print("  - ActionEncoder: Base → Fine-tune (Workflow)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load finetuning data (State, Action) pairs
    print(f"\n📥 Loading finetuning data from: {finetuning_data_path}")
    with open(finetuning_data_path, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
    
    training_examples = [InputExample(texts=ex['texts']) for ex in examples_data]
    print(f"   ✓ Loaded {len(training_examples):,} (State, Action) pairs")
    print(f"   ℹ️  Workflow states already included in State-Action pairs")
    
    # Load BASE models for both encoders
    print("\n🔧 Loading BASE models 'all-mpnet-base-v2'...")
    state_encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
    action_encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
    print(f"   ✓ StateEncoder loaded")
    print(f"   ✓ ActionEncoder loaded")
    
    # === TRAIN STATE ENCODER ===
    print("\n" + "=" * 70)
    print("🏋️  Fine-tuning StateEncoder...")
    print("=" * 70)
    
    # Use only state-action pairs (states already included as first element)
    print(f"   Total StateEncoder training examples: {len(training_examples)}")
    
    state_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )
    
    state_loss = losses.MultipleNegativesRankingLoss(model=state_encoder)
    warmup_steps = int(len(state_dataloader) * num_epochs * 0.1)
    
    print(f"   • Batch size: {batch_size}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Steps per epoch: {len(state_dataloader):,}")
    print(f"   • Warmup steps: {warmup_steps:,}")
    
    state_output_path = "/output/model_a_state_encoder"
    os.makedirs(state_output_path, exist_ok=True)
    
    state_encoder.fit(
        train_objectives=[(state_dataloader, state_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=state_output_path,
        show_progress_bar=True,
        checkpoint_path=os.path.join(state_output_path, "checkpoints"),
        checkpoint_save_steps=100,
        checkpoint_save_total_limit=2
    )
    
    # === TRAIN ACTION ENCODER ===
    print("\n" + "=" * 70)
    print("🏋️  Fine-tuning ActionEncoder...")
    print("=" * 70)
    
    # Use only state-action pairs, but swap them so Action is anchor
    swapped_examples = [InputExample(texts=[ex.texts[1], ex.texts[0]]) for ex in training_examples]
    
    action_dataloader = DataLoader(
        swapped_examples,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )
    
    action_loss = losses.MultipleNegativesRankingLoss(model=action_encoder)
    warmup_steps = int(len(action_dataloader) * num_epochs * 0.1)
    
    print(f"   • Batch size: {batch_size}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Steps per epoch: {len(action_dataloader):,}")
    print(f"   • Warmup steps: {warmup_steps:,}")
    
    action_output_path = "/output/model_a_action_encoder"
    os.makedirs(action_output_path, exist_ok=True)
    
    action_encoder.fit(
        train_objectives=[(action_dataloader, action_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=action_output_path,
        show_progress_bar=True,
        checkpoint_path=os.path.join(action_output_path, "checkpoints"),
        checkpoint_save_steps=100,
        checkpoint_save_total_limit=2
    )
    
    # Save config
    config = {
        "model_type": "Model A - Domain-Specific",
        "architecture": "Dual-Encoder (Two-Tower)",
        "state_encoder": {
            "base_model": "all-mpnet-base-v2",
            "pretraining": "None",
            "finetuning_data": f"workflow_traces ({len(training_examples)} State-Action pairs)"
        },
        "action_encoder": {
            "base_model": "all-mpnet-base-v2",
            "pretraining": "None",
            "finetuning_data": f"workflow_traces ({len(training_examples)} pairs)"
        },
        "batch_size": batch_size,
        "epochs": num_epochs
    }
    
    with open(os.path.join(state_output_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(action_output_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    output_vol.commit()
    
    print("\n" + "=" * 70)
    print(f"✅ Model A Complete!")
    print("=" * 70)
    print(f"   StateEncoder: {state_output_path}")
    print(f"   ActionEncoder: {action_output_path}")
    print("=" * 70)
    
    return {"status": "success", "model": "A", "state_path": state_output_path, "action_path": action_output_path}


# --- 4. Model B: Fine-tune từ Pre-trained Model (Toàn diện) ---
@app.function(
    image=image,
    gpu="T4",
    retries=5,
    volumes={
        "/data": training_data_vol,
        "/pretrained": pretrained_model_vol,
        "/output": output_vol
    },
    timeout=3600,
)
def finetune_model_b_dual_corrected(
    batch_size: int = 32,
    num_epochs: int = 3,
    finetuning_data_path: str = "/data/finetuning/wrl_finetuning_examples.json",
    pretrained_action_path: str = "/pretrained/pretrained_encoder"  # Only ActionEncoder has pre-training
):
    """
    Model B (Toàn diện):
    - StateEncoder: Base → Fine-tune (State-Action pairs only)
    - ActionEncoder: Base → Pre-train (ADP+T1) → Fine-tune (Clinical)
    
    NOTE: Clinical states already included in wrl_finetuning_examples.json
    """
    import json
    import torch
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("🌟 Model B: Fine-tuning DUAL-ENCODER (Comprehensive)")
    print("=" * 70)
    print("CORRECTED LOGIC:")
    print("  - StateEncoder: Base → Fine-tune (Workflow)")
    print("  - ActionEncoder: Base → Pre-train (ADP+T1) → Fine-tune (Workflow)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load finetuning data
    print(f"\n📥 Loading finetuning data from: {finetuning_data_path}")
    with open(finetuning_data_path, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
    
    training_examples = [InputExample(texts=ex['texts']) for ex in examples_data]
    print(f"   ✓ Loaded {len(training_examples):,} (State, Action) pairs")
    print(f"   ℹ️  Workflow states already included in State-Action pairs")
    
    # Load models
    print("\n🔧 Loading models...")
    
    # StateEncoder: Load from BASE (no pre-training for states)
    print("   Loading StateEncoder from BASE model...")
    state_encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
    print(f"   ✓ StateEncoder loaded from BASE")
    
    # ActionEncoder: Load from PRE-TRAINED model
    print(f"   Loading ActionEncoder from PRE-TRAINED model: {pretrained_action_path}")
    if not os.path.exists(pretrained_action_path):
        raise FileNotFoundError(f"Pretrained ActionEncoder not found at {pretrained_action_path}")
    
    action_encoder = SentenceTransformer(pretrained_action_path, device=device)
    print(f"   ✓ ActionEncoder loaded from PRE-TRAINED")
    
    # === TRAIN STATE ENCODER ===
    print("\n" + "=" * 70)
    print("🏋️  Fine-tuning StateEncoder (from BASE)...")
    print("=" * 70)
    
    print(f"   Total StateEncoder training examples: {len(training_examples)}")
    
    state_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )
    
    state_loss = losses.MultipleNegativesRankingLoss(model=state_encoder)
    warmup_steps = int(len(state_dataloader) * num_epochs * 0.1)
    
    print(f"   • Batch size: {batch_size}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Steps per epoch: {len(state_dataloader):,}")
    print(f"   • Warmup steps: {warmup_steps:,}")
    
    state_output_path = "/output/model_b_state_encoder"
    os.makedirs(state_output_path, exist_ok=True)
    
    state_encoder.fit(
        train_objectives=[(state_dataloader, state_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=state_output_path,
        show_progress_bar=True,
        checkpoint_path=os.path.join(state_output_path, "checkpoints"),
        checkpoint_save_steps=100,
        checkpoint_save_total_limit=2
    )
    
    # === TRAIN ACTION ENCODER ===
    print("\n" + "=" * 70)
    print("🏋️  Fine-tuning ActionEncoder (from PRE-TRAINED)...")
    print("=" * 70)
    
    swapped_examples = [InputExample(texts=[ex.texts[1], ex.texts[0]]) for ex in training_examples]
    
    action_dataloader = DataLoader(
        swapped_examples,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )
    
    action_loss = losses.MultipleNegativesRankingLoss(model=action_encoder)
    warmup_steps = int(len(action_dataloader) * num_epochs * 0.1)
    
    print(f"   • Batch size: {batch_size}")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Steps per epoch: {len(action_dataloader):,}")
    print(f"   • Warmup steps: {warmup_steps:,}")
    
    action_output_path = "/output/model_b_action_encoder"
    os.makedirs(action_output_path, exist_ok=True)
    
    action_encoder.fit(
        train_objectives=[(action_dataloader, action_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=action_output_path,
        show_progress_bar=True,
        checkpoint_path=os.path.join(action_output_path, "checkpoints"),
        checkpoint_save_steps=100,
        checkpoint_save_total_limit=2
    )
    
    # Save config
    config = {
        "model_type": "Model B - Comprehensive",
        "architecture": "Dual-Encoder (Two-Tower)",
        "state_encoder": {
            "base_model": "all-mpnet-base-v2",
            "pretraining": "None (state data not in ADP+T1)",
            "finetuning_data": f"workflow_traces ({len(training_examples)} State-Action pairs)"
        },
        "action_encoder": {
            "base_model": "all-mpnet-base-v2",
            "pretraining": "ADP + T1 (343K Action-Action pairs)",
            "finetuning_data": f"workflow_traces ({len(training_examples)} pairs)"
        },
        "batch_size": batch_size,
        "epochs": num_epochs
    }
    
    with open(os.path.join(state_output_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(action_output_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    output_vol.commit()
    
    print("\n" + "=" * 70)
    print(f"✅ Model B Complete!")
    print("=" * 70)
    print(f"   StateEncoder: {state_output_path}")
    print(f"   ActionEncoder: {action_output_path}")
    print("=" * 70)
    
    return {"status": "success", "model": "B", "state_path": state_output_path, "action_path": action_output_path}


@app.local_entrypoint()
def main():
    """
    Chạy cả 2 fine-tuning jobs song song.
    """
    print("=" * 70)
    print("🚀 VHAS Dual-Encoder Fine-tuning: CORRECTED VERSION")
    print("=" * 70)
    print("\n📊 Two Models:")
    print("   • Model A (Chuyên khoa):")
    print("     - StateEncoder: Base → Fine-tune (Clinical)")
    print("     - ActionEncoder: Base → Fine-tune (Clinical)")
    print("\n   • Model B (Toàn diện):")
    print("     - StateEncoder: Base → Fine-tune (Clinical)")
    print("     - ActionEncoder: Base → Pre-train (ADP+T1) → Fine-tune (Clinical)")
    
    print("\n⚠️  Make sure data is uploaded:")
    print("   modal volume put vhas-training-data data/wrl_finetuning_examples.json /finetuning/wrl_finetuning_examples.json")
    print("   (Clinical states already included in wrl_finetuning_examples.json)")
    
    print("\n🌐 Submitting both jobs to Modal in parallel...\n")
    
    # Launch both jobs in parallel
    job_a = finetune_model_a_dual_corrected.spawn()
    job_b = finetune_model_b_dual_corrected.spawn()
    
    print("✅ Both jobs submitted!")
    print("   • Model A is running...")
    print("   • Model B is running...")
    
    # Wait for results
    result_a = job_a.get()
    result_b = job_b.get()
    
    print("\n" + "=" * 70)
    print("✅ All Fine-tuning Complete!")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Model A:")
    print(f"     - StateEncoder: {result_a['state_path']}")
    print(f"     - ActionEncoder: {result_a['action_path']}")
    print(f"   Model B:")
    print(f"     - StateEncoder: {result_b['state_path']}")
    print(f"     - ActionEncoder: {result_b['action_path']}")
    
    print(f"\n💾 Download models:")
    print(f"   modal volume get vhas-finetuned-output model_a_state_encoder ../output/model_a_state_encoder")
    print(f"   modal volume get vhas-finetuned-output model_a_action_encoder ../output/model_a_action_encoder")
    print(f"   modal volume get vhas-finetuned-output model_b_state_encoder ../output/model_b_state_encoder")
    print(f"   modal volume get vhas-finetuned-output model_b_action_encoder ../output/model_b_action_encoder")

