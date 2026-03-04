"""
GIAI ĐOẠN 1D: XÂY DỰNG EMBEDDING SPACE (BUILD "BẢN ĐỒ GPS")
=============================================================

Script này sử dụng các mô hình Encoder đã được fine-tuned để tạo ra embedding space
hoàn chỉnh cho VHAS Orchestrator. Chạy trên Modal GPU để tăng tốc inference.

WORKFLOW:
1. Upload các file định nghĩa (vhas_universe.json, clinical_states.json) lên volume
   $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json
   $ modal volume put vhas-training-data data/clinical_states/clinical_states.json /definitions/clinical_states.json

2. Chạy script trên Modal
   $ modal run build_embedding_space.py

3. Download embedding spaces về local
   $ modal volume get vhas-finetuned-output embedding_space_base ../output/embedding_space_base
   $ modal volume get vhas-finetuned-output embedding_space_pretrained ../output/embedding_space_pretrained

OUTPUT STRUCTURE:
embedding_space_base/
  ├── embeddings.npy       # Numpy array (N, 768) chứa vectors
  ├── id_to_name.json      # Map từ ID -> tên entity
  └── owner_map.json       # Map từ tool name -> owner agent

embedding_space_pretrained/
  ├── embeddings.npy
  ├── id_to_name.json
  └── owner_map.json
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import modal

# Định nghĩa Modal App và Image
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

# Kết nối các Modal Volumes
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=False)
finetuned_output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=False)

@app.function(
    image=image,
    gpu="T4",  # Sử dụng T4 GPU cho inference nhanh
    timeout=1800,  # 30 phút
    volumes={
        "/definitions": training_data_vol,
        "/data": finetuned_output_vol,  # Mount 1 lần, chứa cả models và output
    },
)
def build_space_for_model(model_path: str, universe_file: str, states_file: str, output_dir: str):
    """
    Tải một Encoder model đã huấn luyện và sử dụng nó để tạo ra embedding space.
    """
    print(f"\n--- Building Embedding Space for model at: {model_path} ---")

    # 1. Tải mô hình Encoder đã được fine-tune
    try:
        print("Loading fine-tuned encoder model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder_model = SentenceTransformer(model_path, device=device)
    except Exception as e:
        print(f"ERROR: Could not load model from {model_path}. Error: {e}")
        return

    # 2. Xây dựng Corpus Hợp nhất (Agents + Tools + States)
    print("Building unified corpus...")
    corpus_texts = []
    id_to_name = {}
    owner_map = {}
    
    current_id = 0
    
    # Tải Agents và Tools
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

    # Tải Clinical States
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

    # 3. Mã hóa toàn bộ Corpus
    print("Encoding the entire corpus... (This may take a moment)")
    corpus_embeddings = encoder_model.encode(
        corpus_texts, 
        convert_to_numpy=True, # Chuyển thẳng sang numpy array
        show_progress_bar=True,
        batch_size=128 # Dùng batch size lớn cho inference
    )

    # 4. Lưu lại các "tài sản"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'embeddings.npy'), corpus_embeddings)
    with open(os.path.join(output_dir, 'id_to_name.json'), 'w') as f:
        json.dump(id_to_name, f, indent=2)
    with open(os.path.join(output_dir, 'owner_map.json'), 'w') as f:
        json.dump(owner_map, f, indent=2)
            
    print(f"--- Embedding Space successfully built and saved to '{output_dir}' ---")
    
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
    
    # Các đường dẫn đến 2 model đã được fine-tune (mounted từ finetuned_output_vol tại /data)
    MODEL_A_PATH = '/data/model_a'
    MODEL_B_PATH = '/data/model_b'
    
    # Các thư mục output tương ứng (sẽ ghi vào /data trong cùng volume)
    OUTPUT_A_DIR = '/data/embedding_space_base'
    OUTPUT_B_DIR = '/data/embedding_space_pretrained'
    
    print("\n" + "="*70)
    print("GIAI ĐOẠN 1D: XÂY DỰNG EMBEDDING SPACE")
    print("="*70)
    print("\nBước này sẽ tạo ra 2 'bản đồ GPS' từ 2 mô hình đã fine-tuned:")
    print("  - Model A (Base → Fine-tune): embedding_space_base/")
    print("  - Model B (Base → Pre-train → Fine-tune): embedding_space_pretrained/")
    print("\nMỗi bản đồ sẽ chứa:")
    print("  - embeddings.npy: Ma trận (N, 768) với N = Agents + Tools + States")
    print("  - id_to_name.json: Ánh xạ ID → Tên entity")
    print("  - owner_map.json: Ánh xạ Tool → Owner Agent")
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
    print("\n🚀 Building embedding space for Model A (Base → Fine-tune)...")
    build_space_for_model.remote(
        model_path=MODEL_A_PATH,
        universe_file=UNIVERSE_FILE,
        states_file=STATES_FILE,
        output_dir=OUTPUT_A_DIR
    )
    
    # Chạy cho Model B  
    print("\n🚀 Building embedding space for Model B (Base → Pre-train → Fine-tune)...")
    build_space_for_model.remote(
        model_path=MODEL_B_PATH,
        universe_file=UNIVERSE_FILE,
        states_file=STATES_FILE,
        output_dir=OUTPUT_B_DIR
    )

    print("\n" + "="*70)
    print("✅ ALL EMBEDDING SPACES HAVE BEEN GENERATED!")
    print("="*70)
    print("\n📦 Để download về local, chạy:")
    print("  $ modal volume get vhas-finetuned-output embedding_space_base ../output/embedding_space_base")
    print("  $ modal volume get vhas-finetuned-output embedding_space_pretrained ../output/embedding_space_pretrained")
    print("\n🎉 GIAI ĐOẠN 1 (Data Preparation & Training) ĐÃ HOÀN THÀNH!")
    print("="*70 + "\n")