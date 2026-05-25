# scripts/prepare_pretraining_data.py

import json
import os
from sentence_transformers.readers import InputExample
from tqdm import tqdm

def create_pretraining_pairs(sequences_file: str, output_dir: str):
    """Create positive pairs for contrastive pretraining from action sequences."""
    print("--- Starting WRL Pretraining Data Preparation ---")

    # 1. Load action sequences
    print(f"Loading action sequences from {sequences_file}...")
    try:
        with open(sequences_file, 'r', encoding='utf-8') as f:
            all_sequences = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {sequences_file}. Please run previous steps.")
        return

    # 2. Create InputExample objects (positive pairs)
    print(f"Generating positive pairs from {len(all_sequences)} sequences...")
    training_examples = []
    for sequence in tqdm(all_sequences, desc="Creating pairs"):
        if len(sequence) < 2:
            continue
            
        for i in range(len(sequence) - 1):
            anchor = sequence[i]
            positive = sequence[i+1]
            
            # Skip self-loop actions
            if anchor != positive:
                # Store positive pairs as simple dicts
                # The model learns representations directly from these action names.
                training_examples.append({"texts": [anchor, positive]})

    # 4. Save output files
    os.makedirs(output_dir, exist_ok=True)

    examples_output_file = os.path.join(output_dir, 'wrl_pretraining_examples.json')

    print(f"Saving {len(training_examples)} training examples to '{examples_output_file}'...")
    with open(examples_output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f)

    print("--- WRL Pretraining Data Preparation Complete! ---")

if __name__ == "__main__":
    # Build relative paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pretraining_dir = os.path.dirname(script_dir) 

    SEQUENCES_FILE = os.path.join(pretraining_dir, 'data', 'pretraining_sequences_combined.json')
    OUTPUT_DIR = os.path.join(pretraining_dir, 'data')

    create_pretraining_pairs(
        sequences_file=SEQUENCES_FILE,
        output_dir=OUTPUT_DIR
    )