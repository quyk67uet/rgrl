"""
GIAI ĐOẠN 1D (REVISED): XÂY DỰNG EMBEDDING SPACE - NO STATES VERSION
======================================================================

Script này tạo embedding space CHỈ với Agents + Tools, KHÔNG bao gồm States.

RATIONALE:
- States là INPUT cho guidance (query), không phải OUTPUT (actions)
- Bao gồm states trong embedding space gây "pha loãng" kết quả tìm kiếm
- Với 46 entities (5 agents + 6 tools + 35 states), guidance chỉ tìm được 2-3 agents
- Loại bỏ states → chỉ còn 11 entities → guidance luôn tìm đủ 5 agents

WORKFLOW:
1. Upload file định nghĩa (chỉ cần vhas_universe.json)
   $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json

2. Chạy script trên Modal
   $ modal run build_embedding_space_no_states.py

3. Download embedding spaces về local
   $ modal volume get vhas-finetuned-output embedding_space_base_no_states ./output/embedding_space_base_no_states
   $ modal volume get vhas-finetuned-output embedding_space_pretrained_no_states ./output/embedding_space_pretrained_no_states

OUTPUT STRUCTURE:
embedding_space_base_no_states/
  ├── embeddings.npy       # Numpy array (11, 768) - chỉ Agents + Tools
  ├── id_to_name.json      # Map từ ID -> tên entity
  └── owner_map.json       # Map từ tool name -> owner agent

embedding_space_pretrained_no_states/
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
def build_space_for_model(model_path: str, universe_file: str, output_dir: str):
    """
    Tải một Encoder model đã huấn luyện và tạo embedding space CHỈ với Agents + Tools.
    
    Args:
        model_path: Path to the encoder model
        universe_file: Path to vhas_universe.json (agents + tools)
        output_dir: Directory to save the embedding space
    """
    print(f"\n--- Building Embedding Space (NO STATES) for model at: {model_path} ---")

    # 1. Tải mô hình Encoder đã được fine-tune
    try:
        print("Loading fine-tuned encoder model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder_model = SentenceTransformer(model_path, device=device)
    except Exception as e:
        print(f"ERROR: Could not load model from {model_path}. Error: {e}")
        return

    # 2. Xây dựng Corpus CHỈ với Agents + Tools (NO STATES)
    print("Building corpus (Agents + Tools only, NO states)...")
    corpus_texts = []
    id_to_name = {}
    owner_map = {}
    
    current_id = 0
    
    # Tải Agents và Tools
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

    # REMOVED: Clinical States section
    # States are query inputs (not actionable entities)
    # Including them dilutes search results and causes guidance to fail
        
    print(f"✓ Corpus built with {len(corpus_texts)} total entities:")
    print(f"  - Agents: {num_agents}")
    print(f"  - Tools: {num_tools}")
    print(f"  - States: 0 (excluded by design)")

    # 3. Mã hóa toàn bộ Corpus
    print("Encoding the entire corpus... (This may take a moment)")
    corpus_embeddings = encoder_model.encode(
        corpus_texts, 
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128
    )

    # 4. Lưu lại các "tài sản"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'embeddings.npy'), corpus_embeddings)
    with open(os.path.join(output_dir, 'id_to_name.json'), 'w') as f:
        json.dump(id_to_name, f, indent=2)
    with open(os.path.join(output_dir, 'owner_map.json'), 'w') as f:
        json.dump(owner_map, f, indent=2)
            
    print(f"--- Embedding Space successfully built and saved to '{output_dir}' ---")
    print(f"    Total embeddings: {corpus_embeddings.shape[0]}")
    print(f"    Embedding dimension: {corpus_embeddings.shape[1]}")
    
    # Commit changes to volume
    from modal import Volume
    vol = Volume.from_name("vhas-finetuned-output")
    vol.commit()

@app.local_entrypoint()
def main():
    """
    Local entrypoint - chạy từ máy local để trigger các Modal functions
    """
    # File định nghĩa (mounted từ training_data_vol)
    UNIVERSE_FILE = '/definitions/definitions/vhas_universe.json'
    
    # Các đường dẫn đến 2 model đã được fine-tune
    MODEL_A_PATH = '/data/model_a'
    MODEL_B_PATH = '/data/model_b'
    
    # Các thư mục output mới (với suffix _no_states)
    OUTPUT_A_DIR = '/data/embedding_space_base_no_states'
    OUTPUT_B_DIR = '/data/embedding_space_pretrained_no_states'
    
    print("\n" + "="*70)
    print("GIAI ĐOẠN 1D (REVISED): XÂY DỰNG EMBEDDING SPACE - NO STATES")
    print("="*70)
    print("\n🎯 Mục tiêu: Tạo embedding space CHỈ với actionable entities")
    print("\nBước này sẽ tạo ra 2 'bản đồ GPS' từ 2 mô hình đã fine-tuned:")
    print("  - Model A: embedding_space_base_no_states/")
    print("  - Model B: embedding_space_pretrained_no_states/")
    print("\nMỗi bản đồ sẽ chứa:")
    print("  - embeddings.npy: Ma trận (11, 768) với 11 = 5 Agents + 6 Tools")
    print("  - id_to_name.json: Ánh xạ ID → Tên entity")
    print("  - owner_map.json: Ánh xạ Tool → Owner Agent")
    print("\n💡 Lợi ích:")
    print("  - Loại bỏ 35 states (76% corpus) gây nhiễu")
    print("  - Guidance sẽ LUÔN tìm đủ 5 agents")
    print("  - Không cần hard-code force-add SummaryAgent")
    print("  - Kết quả tìm kiếm chính xác hơn")
    print("="*70 + "\n")
    
    # Kiểm tra file định nghĩa đã được upload chưa
    import subprocess
    import sys
    
    print("📋 Checking if definition file exists in volume...")
    result = subprocess.run(
        ["modal", "volume", "ls", "vhas-training-data", "/definitions"],
        capture_output=True,
        text=True
    )
    
    if "vhas_universe.json" not in result.stdout:
        print("\n❌ ERROR: vhas_universe.json not found in volume!")
        print("\nBạn cần upload file định nghĩa trước:")
        print("  $ modal volume put vhas-training-data vhas-demo/backend/vhas_universe.json /definitions/vhas_universe.json")
        sys.exit(1)
    
    print("✓ Definition file found in volume\n")
    
    # Chạy cho Model A
    print("\n🚀 Building embedding space for Model A (Base → Fine-tune)...")
    build_space_for_model.remote(
        model_path=MODEL_A_PATH,
        universe_file=UNIVERSE_FILE,
        output_dir=OUTPUT_A_DIR
    )
    
    # Chạy cho Model B  
    print("\n🚀 Building embedding space for Model B (Base → Pre-train → Fine-tune)...")
    build_space_for_model.remote(
        model_path=MODEL_B_PATH,
        universe_file=UNIVERSE_FILE,
        output_dir=OUTPUT_B_DIR
    )

    print("\n" + "="*70)
    print("✅ ALL EMBEDDING SPACES (NO STATES) HAVE BEEN GENERATED!")
    print("="*70)
    print("\n📦 Để download về local, chạy:")
    print("  $ modal volume get vhas-finetuned-output embedding_space_base_no_states/ ./output/embedding_space_base_no_states")
    print("  $ modal volume get vhas-finetuned-output embedding_space_pretrained_no_states/ ./output/embedding_space_pretrained_no_states")
    print("\n🔄 Tiếp theo:")
    print("  1. Embedding spaces đã được tạo trên Modal volume vhas-finetuned-output")
    print("  2. Update train_orchestrator_sb3.py để dùng embedding spaces mới")
    print("  3. Chạy lại training experiments")
    print("\n🎉 Embedding spaces mới sẽ giải quyết vấn đề guidance!")
    print("="*70 + "\n")

