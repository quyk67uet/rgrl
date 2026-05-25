# scripts/train_encoder.py

import json
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import os
import torch

def train_pretrained_encoder(training_examples_file: str, output_path: str):
    """Pre-train the encoder model using contrastive learning on general data."""
    print("--- Starting Stage 1B: Pre-training the Encoder Model ---")

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load base model and training data
    print("Loading base model 'all-mpnet-base-v2'...")
    # We can specify device when loading the model
    encoder_model = SentenceTransformer('all-mpnet-base-v2', device=device)

    print(f"Loading training examples from {training_examples_file}...")
    try:
        with open(training_examples_file, 'r', encoding='utf-8') as f:
            examples_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Training file not found at {training_examples_file}. Please run prepare_pretraining_data.py first.")
        return
        
    # Convert dicts back into InputExample objects
    training_examples = [InputExample(texts=ex['texts']) for ex in examples_data]

    # 2. Setup training environment
    batch_size = 128  # Larger batch to provide more negative examples per batch
    train_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )
    
    train_loss = losses.MultipleNegativesRankingLoss(model=encoder_model)
    
    # 3. Train the model
    num_epochs = 1
    # Use 10% of total steps as warmup, a common stabilization technique
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    print("\n--- Training Details ---")
    print(f"Number of training examples: {len(training_examples)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Total steps per epoch: {len(train_dataloader)}")
    print(f"Warmup steps: {warmup_steps}")
    print("------------------------\n")

    # Save config for reproducibility
    os.makedirs(output_path, exist_ok=True)
    config = {
        "base_model": "all-mpnet-base-v2",
        "loss": "MultipleNegativesRankingLoss",
        "batch_size": batch_size,
        "epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "device": device,
        "num_training_examples": len(training_examples),
        "total_steps": len(train_dataloader) * num_epochs
    }
    
    config_path = os.path.join(output_path, "training_config.json")
    print(f"Saving training config to '{config_path}'...")
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    encoder_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_path=os.path.join(output_path, 'checkpoints'),
        checkpoint_save_steps=1000  # Save checkpoint every 1000 steps
    )
    
    print(f"\n--- Pre-training Complete! Final model saved to '{output_path}' ---")

if __name__ == "__main__":
    # Build relative paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pretraining_dir = os.path.dirname(script_dir)  # d:/VHAS/data/pretraining

    TRAINING_EXAMPLES_FILE = os.path.join(pretraining_dir, 'data', 'wrl_pretraining_examples.json')
    OUTPUT_PATH = os.path.join(pretraining_dir, 'output', 'pretrained_encoder')

    train_pretrained_encoder(
        training_examples_file=TRAINING_EXAMPLES_FILE,
        output_path=OUTPUT_PATH
    )