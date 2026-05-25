"""Build full embedding spaces for the VHAS orchestrator.

This script uses tuned encoder models to produce unified embedding spaces
for agents, tools, and semantic states. It is intended to run on Modal GPUs
for faster batch encoding.

Workflow (Modal):
1. Upload definition files (vhas_universe.json, clinical_states.json):
    $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json
    $ modal volume put vhas-training-data data/clinical_states/clinical_states.json /definitions/clinical_states.json

2. Run on Modal:
    $ modal run build_embedding_space.py

3. Download outputs to local:
    $ modal volume get vhas-finetuned-output embedding_space_base ../output/embedding_space_base
    $ modal volume get vhas-finetuned-output embedding_space_pretrained ../output/embedding_space_pretrained

Output:
embedding_space_*/
  - embeddings.npy       # Numpy array (N, D) containing vectors for agents+tools+states
  - id_to_name.json      # ID -> entity name
  - owner_map.json       # tool name -> owner agent
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import modal

# Modal app and image definition
app = modal.App("vhas-build-embedding-space")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install([
        "sentence-transformers>=3.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ])
)

# Connect Modal volumes
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=False)
finetuned_output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=False)

@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU for faster inference
    timeout=1800,  # 30 minutes
    volumes={
        "/definitions": training_data_vol,
        "/data": finetuned_output_vol,  # Mount 1 lần, chứa cả models và output
    },
)
def build_space_for_model(model_path: str, universe_file: str, states_file: str, output_dir: str):
    """Load a tuned encoder model and build a unified embedding space.

    The encoder encodes agents, tools, and semantic states into a single vector
    space for downstream guidance and retrieval.
    """
    print(f"\n--- Building Embedding Space for model at: {model_path} ---")

    # 1. Load the tuned encoder model
    try:
        print("Loading fine-tuned encoder model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder_model = SentenceTransformer(model_path, device=device)
    except Exception as e:
        print(f"ERROR: Could not load model from {model_path}. Error: {e}")
        return

    # 2. Build unified corpus (agents + tools + states)
    print("Building unified corpus...")
    corpus_texts = []
    id_to_name = {}
    owner_map = {}
    
    current_id = 0
    
    # Load agents and tools from the universe definition
    try:
        with open(universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        
        for agent in universe.get('agents', []):
            corpus_texts.append(agent['description'])
            id_to_name[str(current_id)] = agent['name']
            current_id += 1
            
        for tool in universe.get('tools', []):
            corpus_texts.append(tool['description'])
            id_to_name[str(current_id)] = tool['name']
            owner_map[tool['name']] = tool['owner']
            current_id += 1
    except FileNotFoundError:
        print(f"ERROR: Universe file not found at {universe_file}.")
        return

    # Load clinical/semantic states
    try:
        with open(states_file, 'r', encoding='utf-8') as f:
            states = json.load(f)
        for state in states:
            corpus_texts.append(state)
            id_to_name[str(current_id)] = state
            current_id += 1
    except FileNotFoundError:
        print(f"ERROR: States file not found at {states_file}.")
        return
        
    print(f"Corpus built with {len(corpus_texts)} total entities.")

    # 3. Encode the entire corpus
    print("Encoding the entire corpus... (this may take a moment)")
    corpus_embeddings = encoder_model.encode(
        corpus_texts, 
        convert_to_numpy=True, # Chuyển thẳng sang numpy array
        show_progress_bar=True,
        batch_size=128 # Dùng batch size lớn cho inference
    )

    # 4. Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'embeddings.npy'), corpus_embeddings)
    with open(os.path.join(output_dir, 'id_to_name.json'), 'w') as f:
        json.dump(id_to_name, f, indent=2)
    with open(os.path.join(output_dir, 'owner_map.json'), 'w') as f:
        json.dump(owner_map, f, indent=2)
            
    print(f"--- Embedding space saved to '{output_dir}' ---")
    
    # Commit changes to volume
    from modal import Volume
    vol = Volume.from_name("vhas-finetuned-output")
    vol.commit()

@app.local_entrypoint()
def main():
    """Local entry to trigger the Modal functions from a local machine."""
    # Definition files (mounted from training_data_vol)
    UNIVERSE_FILE = '/definitions/definitions/vhas_universe.json'
    STATES_FILE = '/definitions/definitions/clinical_states.json'

    # Paths to two tuned models (mounted under /data)
    MODEL_A_PATH = '/data/model_a'
    MODEL_B_PATH = '/data/model_b'

    # Output directories (written into /data in the same volume)
    OUTPUT_A_DIR = '/data/embedding_space_base'
    OUTPUT_B_DIR = '/data/embedding_space_pretrained'

    print("\n" + "="*70)
    print("STAGE 1D: BUILD EMBEDDING SPACE")
    print("="*70)
    print("This step builds two unified embedding maps from two tuned models:")
    print("  - Model A (Base → Fine-tune): embedding_space_base/")
    print("  - Model B (Base → Pre-train → Fine-tune): embedding_space_pretrained/")
    print("Each output contains:")
    print("  - embeddings.npy: matrix with agents + tools + states")
    print("  - id_to_name.json: ID → entity name")
    print("  - owner_map.json: Tool → owner agent")
    print("="*70 + "\n")

    # Check that required definition files are uploaded
    import subprocess
    import sys

    print("Checking for required definition files in the volume...")
    result = subprocess.run(
        ["modal", "volume", "ls", "vhas-training-data", "/definitions"],
        capture_output=True,
        text=True,
    )

    if "vhas_universe.json" not in result.stdout or "clinical_states.json" not in result.stdout:
        print("ERROR: Required definition files not found in the volume.")
        print("Please upload them first:")
        print("  $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json")
        print("  $ modal volume put vhas-training-data data/clinical_states/clinical_states.json /definitions/clinical_states.json")
        sys.exit(1)

    print("Definition files present. Proceeding...\n")

    # Launch build for Model A
    print("Building embedding space for Model A (Base → Fine-tune)...")
    build_space_for_model.remote(
        model_path=MODEL_A_PATH,
        universe_file=UNIVERSE_FILE,
        states_file=STATES_FILE,
        output_dir=OUTPUT_A_DIR,
    )

    # Launch build for Model B
    print("Building embedding space for Model B (Base → Pre-train → Fine-tune)...")
    build_space_for_model.remote(
        model_path=MODEL_B_PATH,
        universe_file=UNIVERSE_FILE,
        states_file=STATES_FILE,
        output_dir=OUTPUT_B_DIR,
    )

    print("\n" + "="*70)
    print("ALL EMBEDDING SPACES HAVE BEEN QUEUED")
    print("="*70)
    print("To download results locally, run:")
    print("  $ modal volume get vhas-finetuned-output embedding_space_base ../output/embedding_space_base")
    print("  $ modal volume get vhas-finetuned-output embedding_space_pretrained ../output/embedding_space_pretrained")
    print("" )