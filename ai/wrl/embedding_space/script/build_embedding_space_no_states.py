"""Build embedding spaces containing only actionable entities (no states).

This revised script creates embedding spaces that include agents and tools
only. States are excluded because they are used as queries for guidance
rather than as actionable outputs; including states can dilute search
results and reduce retrieval quality.

Workflow (Modal):
1. Upload `vhas_universe.json` to the training volume:
    $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json

2. Run on Modal:
    $ modal run build_embedding_space_no_states.py

3. Download outputs:
    $ modal volume get vhas-finetuned-output embedding_space_base_no_states ./output/embedding_space_base_no_states
    $ modal volume get vhas-finetuned-output embedding_space_pretrained_no_states ./output/embedding_space_pretrained_no_states

Output:
embedding_space_*_no_states/
  - embeddings.npy       # Numpy array (N, D) - Agents + Tools only
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
app = modal.App("vhas-build-embedding-space-no-states")

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
def build_space_for_model(model_path: str, universe_file: str, output_dir: str):
    """Load a tuned encoder and build an embedding space for agents+tools.

    Args:
        model_path: Path to the encoder model
        universe_file: Path to vhas_universe.json (agents + tools)
        output_dir: Directory to save the embedding space
    """
    print(f"\n--- Building Embedding Space (NO STATES) for model at: {model_path} ---")

    # 1. Load the tuned encoder model
    try:
        print("Loading fine-tuned encoder model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder_model = SentenceTransformer(model_path, device=device)
    except Exception as e:
        print(f"ERROR: Could not load model from {model_path}. Error: {e}")
        return

    # 2. Build corpus with agents + tools only (no states)
    print("Building corpus (Agents + Tools only, NO states)...")
    corpus_texts = []
    id_to_name = {}
    owner_map = {}
    
    current_id = 0
    
    # Load agents and tools from the universe file
    try:
        with open(universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        
        # Add Agents
        for agent in universe.get('agents', []):
            corpus_texts.append(agent['description'])
            id_to_name[str(current_id)] = agent['name']
            current_id += 1
        
        num_agents = current_id
        
        # Add Tools
        for tool in universe.get('tools', []):
            corpus_texts.append(tool['description'])
            id_to_name[str(current_id)] = tool['name']
            owner_map[tool['name']] = tool['owner']
            current_id += 1
            
        num_tools = current_id - num_agents
        
    except FileNotFoundError:
        print(f"ERROR: Universe file not found at {universe_file}.")
        return

    # Clinical states are intentionally excluded (they're query inputs)
        
    print(f"✓ Corpus built with {len(corpus_texts)} total entities:")
    print(f"  - Agents: {num_agents}")
    print(f"  - Tools: {num_tools}")
    print(f"  - States: 0 (excluded by design)")

    # 3. Encode the corpus
    print("Encoding the entire corpus... (this may take a moment)")
    corpus_embeddings = encoder_model.encode(
        corpus_texts, 
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128
    )

    # 4. Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'embeddings.npy'), corpus_embeddings)
    with open(os.path.join(output_dir, 'id_to_name.json'), 'w') as f:
        json.dump(id_to_name, f, indent=2)
    with open(os.path.join(output_dir, 'owner_map.json'), 'w') as f:
        json.dump(owner_map, f, indent=2)
            
    print(f"--- Embedding space saved to '{output_dir}' ---")
    print(f"    Total embeddings: {corpus_embeddings.shape[0]}")
    print(f"    Embedding dimension: {corpus_embeddings.shape[1]}")
    
    # Commit changes to volume
    from modal import Volume
    vol = Volume.from_name("vhas-finetuned-output")
    vol.commit()

@app.local_entrypoint()
def main():
    """Local entry to trigger the Modal functions from a local machine."""
    # Definition file (mounted from training_data_vol)
    UNIVERSE_FILE = '/definitions/definitions/vhas_universe.json'

    # Paths to two tuned models (mounted under /data)
    MODEL_A_PATH = '/data/model_a'
    MODEL_B_PATH = '/data/model_b'

    # Output directories (suffix _no_states)
    OUTPUT_A_DIR = '/data/embedding_space_base_no_states'
    OUTPUT_B_DIR = '/data/embedding_space_pretrained_no_states'

    print("\n" + "="*70)
    print("STAGE 1D (REVISED): BUILD EMBEDDING SPACE - NO STATES")
    print("="*70)
    print("Goal: Build embedding spaces that include actionable entities only (agents + tools)")
    print("This step will create two maps from two tuned models:")
    print("  - Model A: embedding_space_base_no_states/")
    print("  - Model B: embedding_space_pretrained_no_states/")
    print("Each output will contain:")
    print("  - embeddings.npy: matrix with Agents + Tools")
    print("  - id_to_name.json: ID → entity name")
    print("  - owner_map.json: Tool → owner agent")
    print("" )

    # Check that the universe definition has been uploaded
    import subprocess
    import sys

    print("Checking for the universe definition in volume...")
    result = subprocess.run(
        ["modal", "volume", "ls", "vhas-training-data", "/definitions"],
        capture_output=True,
        text=True,
    )

    if "vhas_universe.json" not in result.stdout:
        print("ERROR: vhas_universe.json not found in the volume.")
        print("Please upload the universe definition:")
        print("  $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json")
        sys.exit(1)

    print("Definition file found. Proceeding...\n")

    # Launch build for Model A
    print("Building embedding space for Model A (Base → Fine-tune)...")
    build_space_for_model.remote(
        model_path=MODEL_A_PATH,
        universe_file=UNIVERSE_FILE,
        output_dir=OUTPUT_A_DIR,
    )

    # Launch build for Model B
    print("Building embedding space for Model B (Base → Pre-train → Fine-tune)...")
    build_space_for_model.remote(
        model_path=MODEL_B_PATH,
        universe_file=UNIVERSE_FILE,
        output_dir=OUTPUT_B_DIR,
    )

    print("\n" + "="*70)
    print("ALL NO-STATES EMBEDDING SPACES HAVE BEEN QUEUED")
    print("="*70)
    print("To download results locally, run:")
    print("  $ modal volume get vhas-finetuned-output embedding_space_base_no_states/ ./output/embedding_space_base_no_states")
    print("  $ modal volume get vhas-finetuned-output embedding_space_pretrained_no_states/ ./output/embedding_space_pretrained_no_states")
    print("" )

