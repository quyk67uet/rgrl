"""Build dual embedding spaces (Two-Tower architecture).

This script uses a tuned Dual-Encoder (StateEncoder + ActionEncoder) to
produce two separate embedding spaces:

- State Embedding Space: embeddings for semantic state prototypes
- Action Embedding Space: embeddings for agents and tools

Architecture:
- `StateEncoder` encodes states
- `ActionEncoder` encodes actions (agents + tools)
- Guidance queries the Action space using a state-based semantic query

Workflow (Modal):
1. Upload definition files to the volume:
    $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json
    $ modal volume put vhas-training-data data/clinical_states/clinical_states.json /definitions/clinical_states.json

2. Run on Modal:
    $ modal run build_embedding_space_dual.py

3. Download outputs to local:
    $ modal volume get vhas-finetuned-output embedding_space_model_a_dual/ ../output/embedding_space_model_a_dual
    $ modal volume get vhas-finetuned-output embedding_space_model_b_dual/ ../output/embedding_space_model_b_dual

Output (per model):
embedding_space_model_X_dual/
  - state_embeddings.npy       # Semantic state prototype embeddings (N_states, D)
  - action_embeddings.npy      # Action embeddings (N_actions, D)
  - state_id_to_name.json      # ID -> state prototype text
  - action_id_to_name.json     # ID -> action name
  - owner_map.json             # tool name -> owner agent
"""

import json
import os
from pathlib import Path

import modal
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Modal app and image definition
app = modal.App("vhas-build-dual-embedding-space")

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

# Main entrypoint path configuration
UNIVERSE_FILE = Path("/definitions/definitions/vhas_universe.json")
STATES_FILE = Path("/definitions/definitions/clinical_states.json")
MODEL_A_STATE_PATH = Path("/data/model_a_state_encoder")
MODEL_A_ACTION_PATH = Path("/data/model_a_action_encoder")
MODEL_B_STATE_PATH = Path("/data/model_b_state_encoder")
MODEL_B_ACTION_PATH = Path("/data/model_b_action_encoder")
OUTPUT_A_DIR = Path("/data/embedding_space_model_a_dual")
OUTPUT_B_DIR = Path("/data/embedding_space_model_b_dual")

@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={
        "/definitions": training_data_vol,
        "/data": finetuned_output_vol,
    },
)
def build_dual_space_for_model(
    state_encoder_path: str,
    action_encoder_path: str,
    universe_file: str,
    states_file: str,
    output_dir: str
):
    """Load Dual-Encoder models and build two embedding spaces.

    Args:
        state_encoder_path: Path to the state encoder model
        action_encoder_path: Path to the action encoder model
        universe_file: Path to vhas_universe.json (agents + tools)
        states_file: Path to semantic state prototypes file
        output_dir: Output directory to save embeddings and metadata
    """
    print(f"\n--- Building Dual Embedding Space ---")
    print(f"StateEncoder: {state_encoder_path}")
    print(f"ActionEncoder: {action_encoder_path}")

    # 1. Load Dual-Encoder models
    try:
        print("\nLoading Dual-Encoder models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_encoder = SentenceTransformer(state_encoder_path, device=device)
        if device == "cuda":
            torch.cuda.empty_cache()
        action_encoder = SentenceTransformer(action_encoder_path, device=device)
        print(f"   ✓ StateEncoder loaded ({state_encoder.get_sentence_embedding_dimension()} dims)")
        print(f"   ✓ ActionEncoder loaded ({action_encoder.get_sentence_embedding_dimension()} dims)")
    except Exception as e:
        print(f"ERROR: Could not load models. Error: {e}")
        return

    # 2. Build action corpus (agents + tools)
    print("\nBuilding Action corpus (Agents + Tools)...")
    action_texts = []
    action_id_to_name = {}
    owner_map = {}
    
    current_id = 0
    
    try:
        with open(universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        
        # Add Agents
        for agent in universe.get('agents', []):
            action_texts.append(agent['description'])
            action_id_to_name[str(current_id)] = agent['name']
            current_id += 1
        
        # Add Tools
        for tool in universe.get('tools', []):
            action_texts.append(tool['description'])
            action_id_to_name[str(current_id)] = tool['name']
            owner_map[tool['name']] = tool['owner']
            current_id += 1
        
        print(f"   ✓ Action corpus: {len(action_texts)} entities")
        print(f"     - Agents: {len(universe.get('agents', []))}")
        print(f"     - Tools: {len(universe.get('tools', []))}")
    except FileNotFoundError:
        print(f"ERROR: Universe file not found at {universe_file}.")
        return

    # 3. Build state corpus (semantic state prototypes)
    print("\nBuilding State corpus (Semantic State Prototypes)...")
    state_texts = []
    state_id_to_name = {}
    
    current_id = 0
    
    try:
        with open(states_file, 'r', encoding='utf-8') as f:
            states = json.load(f)
        
        for state in states:
            state_texts.append(state)
            state_id_to_name[str(current_id)] = state
            current_id += 1
        
        print(f"   ✓ State corpus: {len(state_texts)} semantic state prototypes")
    except FileNotFoundError:
        print(f"ERROR: Semantic State Space file not found at {states_file}.")
        return

    # 4. Encode action corpus with the action encoder
    print("\nEncoding Action corpus with ActionEncoder...")
    action_embeddings = action_encoder.encode(
        action_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128
    )
    print(f"   ✓ Action embeddings: {action_embeddings.shape}")

    # 5. Encode state corpus with the state encoder
    print("\nEncoding Semantic State corpus with StateEncoder...")
    state_embeddings = state_encoder.encode(
        state_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128
    )
    print(f"   ✓ State embeddings: {state_embeddings.shape}")

    # 6. Save artifacts
    print(f"\nSaving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save action embeddings and metadata
    np.save(os.path.join(output_dir, 'action_embeddings.npy'), action_embeddings)
    with open(os.path.join(output_dir, 'action_id_to_name.json'), 'w') as f:
        json.dump(action_id_to_name, f, indent=2)
    with open(os.path.join(output_dir, 'owner_map.json'), 'w') as f:
        json.dump(owner_map, f, indent=2)
    
    # Save state embeddings and metadata
    np.save(os.path.join(output_dir, 'state_embeddings.npy'), state_embeddings)
    with open(os.path.join(output_dir, 'state_id_to_name.json'), 'w') as f:
        json.dump(state_id_to_name, f, indent=2)
    
    # Save architecture info
    architecture_info = {
        "architecture": "Dual-Encoder (Two-Tower)",
        "state_encoder_path": state_encoder_path,
        "action_encoder_path": action_encoder_path,
        "num_actions": len(action_texts),
        "num_semantic_state_prototypes": len(state_texts),
        "embedding_dim": action_embeddings.shape[1],
        "guidance_strategy": "Query with StateEncoder, Search in Action space using semantic context"
    }
    with open(os.path.join(output_dir, 'architecture_info.json'), 'w') as f:
        json.dump(architecture_info, f, indent=2)
    
    print(f"   ✓ action_embeddings.npy saved ({action_embeddings.shape})")
    print(f"   ✓ state_embeddings.npy saved ({state_embeddings.shape})")
    print(f"   ✓ action_id_to_name.json saved")
    print(f"   ✓ state_id_to_name.json saved")
    print(f"   ✓ owner_map.json saved")
    print(f"   ✓ architecture_info.json saved")
    
    print(f"\n--- Dual Embedding Space successfully built and saved to '{output_dir}' ---")
    
    # Commit changes to volume
    from modal import Volume
    vol = Volume.from_name("vhas-finetuned-output")
    vol.commit()

@app.local_entrypoint()
def main():
    """Local entry to trigger the Modal functions from a local machine."""
    print("\n" + "="*70)
    print("STAGE 1D: BUILD DUAL EMBEDDING SPACES")
    print("="*70)
    print("\nArchitecture: Two-Tower (StateEncoder + ActionEncoder)")
    print("\nThis step builds two parallel embedding maps from two tuned models:")
    print("  - Model A (Base → Fine-tune): embedding_space_model_a_dual/")
    print("  - Model B (Base → Pre-train → Fine-tune): embedding_space_model_b_dual/")
    print("\nEach output contains:")
    print("  - action_embeddings.npy: matrix (N_actions, D) - Agents + Tools")
    print("  - state_embeddings.npy: matrix (N_states, D) - Semantic state prototypes")
    print("  - action_id_to_name.json: ID → Action name")
    print("  - state_id_to_name.json: ID → Semantic state prototype text")
    print("  - owner_map.json: Tool → Owner Agent")
    print("  - architecture_info.json: metadata")
    print("="*70 + "\n")

    # Check that the required definition files are present in the volume
    import subprocess
    import sys

    print("Checking for required definition files in the volume...")
    result = subprocess.run(
        ["modal", "volume", "ls", "vhas-training-data", "/definitions"],
        capture_output=True,
        text=True,
    )

    if UNIVERSE_FILE.name not in result.stdout or STATES_FILE.name not in result.stdout:
        print("\nERROR: Required definition files not found in the volume.")
        print("Please upload the definition files first:")
        print("  $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json "
              f"{UNIVERSE_FILE.as_posix()}")
        print("  $ modal volume put vhas-training-data data/clinical_states/clinical_states.json "
              f"{STATES_FILE.as_posix()}")
        sys.exit(1)

    print("Definitions present in volume. Proceeding...\n")

    # Launch build for Model A
    print("Building dual embedding space for Model A (Base → Fine-tune)...")
    build_dual_space_for_model.remote(
        state_encoder_path=MODEL_A_STATE_PATH.as_posix(),
        action_encoder_path=MODEL_A_ACTION_PATH.as_posix(),
        universe_file=UNIVERSE_FILE.as_posix(),
        states_file=STATES_FILE.as_posix(),
        output_dir=OUTPUT_A_DIR.as_posix(),
    )

    # Launch build for Model B
    print("Building dual embedding space for Model B (Base → Pre-train → Fine-tune)...")
    build_dual_space_for_model.remote(
        state_encoder_path=MODEL_B_STATE_PATH.as_posix(),
        action_encoder_path=MODEL_B_ACTION_PATH.as_posix(),
        universe_file=UNIVERSE_FILE.as_posix(),
        states_file=STATES_FILE.as_posix(),
        output_dir=OUTPUT_B_DIR.as_posix(),
    )

    print("\n" + "="*70)
    print("ALL DUAL EMBEDDING SPACES HAVE BEEN QUEUED")
    print("="*70)
    print("To download results locally, run:")
    print("  $ modal volume get vhas-finetuned-output embedding_space_model_a_dual/ ../output/embedding_space_model_a_dual")
    print("  $ modal volume get vhas-finetuned-output embedding_space_model_b_dual/ ../output/embedding_space_model_b_dual")
    print("" )

