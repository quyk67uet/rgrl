"""
GIAI ĐOẠN 1D: XÂY DỰNG DUAL EMBEDDING SPACE (Two-Tower Architecture)
====================================================================

Script này sử dụng Dual-Encoder (StateEncoder + ActionEncoder) đã được fine-tuned 
để tạo ra HAI embedding spaces riêng biệt:
1. State Embedding Space: Chứa embeddings của Clinical States
2. Action Embedding Space: Chứa embeddings của Agents và Tools

KIẾN TRÚC:
- StateEncoder mã hóa States
- ActionEncoder mã hóa Actions (Agents + Tools)
- Guidance sẽ search trong Action space bằng State query

WORKFLOW:
1. Upload các file định nghĩa lên volume
   $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json
   $ modal volume put vhas-training-data data/clinical_states/clinical_states.json /definitions/clinical_states.json

2. Chạy script trên Modal
   $ modal run build_embedding_space_dual.py

3. Download embedding spaces về local
   $ modal volume get vhas-finetuned-output embedding_space_model_a_dual/ ../output/embedding_space_model_a_dual
   $ modal volume get vhas-finetuned-output embedding_space_model_b_dual/ ../output/embedding_space_model_b_dual

OUTPUT STRUCTURE (cho mỗi model):
embedding_space_model_a_dual/
  ├── state_embeddings.npy       # States embeddings (N_states, 768)
  ├── action_embeddings.npy      # Actions embeddings (N_actions, 768)
  ├── state_id_to_name.json      # Map từ ID -> State text
  ├── action_id_to_name.json     # Map từ ID -> Action name
  └── owner_map.json             # Map từ tool name -> owner agent
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import modal

# Định nghĩa Modal App và Image
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

# Kết nối các Modal Volumes
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=False)
finetuned_output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=False)

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
    """
    Tải Dual-Encoder (StateEncoder + ActionEncoder) và tạo ra hai embedding spaces riêng biệt.
    
    Args:
        state_encoder_path: Path to StateEncoder model
        action_encoder_path: Path to ActionEncoder model
        universe_file: Path to vhas_universe.json (agents + tools)
        states_file: Path to clinical_states.json
        output_dir: Output directory for embedding spaces
    """
    print(f"\n--- Building Dual Embedding Space ---")
    print(f"StateEncoder: {state_encoder_path}")
    print(f"ActionEncoder: {action_encoder_path}")

    # 1. Tải Dual-Encoder models
    try:
        print("\nLoading Dual-Encoder models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_encoder = SentenceTransformer(state_encoder_path, device=device)
        action_encoder = SentenceTransformer(action_encoder_path, device=device)
        print(f"   ✓ StateEncoder loaded ({state_encoder.get_sentence_embedding_dimension()} dims)")
        print(f"   ✓ ActionEncoder loaded ({action_encoder.get_sentence_embedding_dimension()} dims)")
    except Exception as e:
        print(f"ERROR: Could not load models. Error: {e}")
        return

    # 2. Xây dựng Action Corpus (Agents + Tools)
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

    # 3. Xây dựng State Corpus (Clinical States)
    print("\nBuilding State corpus (Clinical States)...")
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
        
        print(f"   ✓ State corpus: {len(state_texts)} clinical states")
    except FileNotFoundError:
        print(f"ERROR: States file not found at {states_file}.")
        return

    # 4. Mã hóa Action Corpus với ActionEncoder
    print("\nEncoding Action corpus with ActionEncoder...")
    action_embeddings = action_encoder.encode(
        action_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128
    )
    print(f"   ✓ Action embeddings: {action_embeddings.shape}")

    # 5. Mã hóa State Corpus với StateEncoder
    print("\nEncoding State corpus with StateEncoder...")
    state_embeddings = state_encoder.encode(
        state_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128
    )
    print(f"   ✓ State embeddings: {state_embeddings.shape}")

    # 6. Lưu lại các "tài sản"
    print(f"\nSaving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Action embeddings and metadata
    np.save(os.path.join(output_dir, 'action_embeddings.npy'), action_embeddings)
    with open(os.path.join(output_dir, 'action_id_to_name.json'), 'w') as f:
        json.dump(action_id_to_name, f, indent=2)
    with open(os.path.join(output_dir, 'owner_map.json'), 'w') as f:
        json.dump(owner_map, f, indent=2)
    
    # Save State embeddings and metadata
    np.save(os.path.join(output_dir, 'state_embeddings.npy'), state_embeddings)
    with open(os.path.join(output_dir, 'state_id_to_name.json'), 'w') as f:
        json.dump(state_id_to_name, f, indent=2)
    
    # Save architecture info
    architecture_info = {
        "architecture": "Dual-Encoder (Two-Tower)",
        "state_encoder_path": state_encoder_path,
        "action_encoder_path": action_encoder_path,
        "num_actions": len(action_texts),
        "num_states": len(state_texts),
        "embedding_dim": action_embeddings.shape[1],
        "guidance_strategy": "Query with StateEncoder, Search in ActionEncoder space"
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
    """
    Local entrypoint - chạy từ máy local để trigger các Modal functions
    """
    # Các file định nghĩa (mounted từ training_data_vol)
    UNIVERSE_FILE = '/definitions/definitions/vhas_universe.json'
    STATES_FILE = '/definitions/definitions/clinical_states.json'
    
    # Các đường dẫn đến 2 models đã được fine-tune (mounted từ finetuned_output_vol tại /data)
    # Model A: Base → Fine-tune
    MODEL_A_STATE_PATH = '/data/model_a_state_encoder'
    MODEL_A_ACTION_PATH = '/data/model_a_action_encoder'
    
    # Model B: Base → Pre-train → Fine-tune
    MODEL_B_STATE_PATH = '/data/model_b_state_encoder'
    MODEL_B_ACTION_PATH = '/data/model_b_action_encoder'
    
    # Các thư mục output tương ứng
    OUTPUT_A_DIR = '/data/embedding_space_model_a_dual'
    OUTPUT_B_DIR = '/data/embedding_space_model_b_dual'
    
    print("\n" + "="*70)
    print("GIAI ĐOẠN 1D: XÂY DỰNG DUAL EMBEDDING SPACE")
    print("="*70)
    print("\nKiến trúc: Two-Tower (StateEncoder + ActionEncoder)")
    print("\nBước này sẽ tạo ra 2 'bản đồ GPS song song' từ 2 mô hình đã fine-tuned:")
    print("  - Model A (Base → Fine-tune): embedding_space_model_a_dual/")
    print("  - Model B (Base → Pre-train → Fine-tune): embedding_space_model_b_dual/")
    print("\nMỗi bản đồ sẽ chứa:")
    print("  - action_embeddings.npy: Ma trận (N_actions, 768) - Agents + Tools")
    print("  - state_embeddings.npy: Ma trận (N_states, 768) - Clinical States")
    print("  - action_id_to_name.json: Ánh xạ ID → Action name")
    print("  - state_id_to_name.json: Ánh xạ ID → State text")
    print("  - owner_map.json: Ánh xạ Tool → Owner Agent")
    print("  - architecture_info.json: Thông tin kiến trúc")
    print("="*70 + "\n")
    
    # Kiểm tra các file định nghĩa đã được upload chưa
    import subprocess
    import sys
    
    print("📋 Checking if definition files exist in volume...")
    result = subprocess.run(
        ["modal", "volume", "ls", "vhas-training-data", "/definitions"],
        capture_output=True,
        text=True
    )
    
    if "vhas_universe.json" not in result.stdout or "clinical_states.json" not in result.stdout:
        print("\n❌ ERROR: Definition files not found in volume!")
        print("\nBạn cần upload các file định nghĩa trước:")
        print("  $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json")
        print("  $ modal volume put vhas-training-data data/clinical_states/clinical_states.json /definitions/clinical_states.json")
        sys.exit(1)
    
    print("✓ Definition files found in volume\n")
    
    # Chạy cho Model A
    print("\n🚀 Building dual embedding space for Model A (Base → Fine-tune)...")
    build_dual_space_for_model.remote(
        state_encoder_path=MODEL_A_STATE_PATH,
        action_encoder_path=MODEL_A_ACTION_PATH,
        universe_file=UNIVERSE_FILE,
        states_file=STATES_FILE,
        output_dir=OUTPUT_A_DIR
    )
    
    # Chạy cho Model B
    print("\n🚀 Building dual embedding space for Model B (Base → Pre-train → Fine-tune)...")
    build_dual_space_for_model.remote(
        state_encoder_path=MODEL_B_STATE_PATH,
        action_encoder_path=MODEL_B_ACTION_PATH,
        universe_file=UNIVERSE_FILE,
        states_file=STATES_FILE,
        output_dir=OUTPUT_B_DIR
    )

    print("\n" + "="*70)
    print("✅ ALL DUAL EMBEDDING SPACES HAVE BEEN GENERATED!")
    print("="*70)
    print("\n📦 Để download về local, chạy:")
    print("  $ modal volume get vhas-finetuned-output embedding_space_model_a_dual/ ../output/embedding_space_model_a_dual")
    print("  $ modal volume get vhas-finetuned-output embedding_space_model_b_dual/ ../output/embedding_space_model_b_dual")
    print("\n🎉 GIAI ĐOẠN 1 (Data Preparation & Dual-Encoder Training) ĐÃ HOÀN THÀNH!")
    print("="*70 + "\n")

