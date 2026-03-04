# scripts/visualize_embeddings.py
import json
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_labels(id_to_name: dict, universe_file: str, states_file: str) -> list:
    """Tạo nhãn chi tiết cho mỗi thực thể trong corpus với grouping hợp lý."""
    
    # Tải định nghĩa để biết agent nào sở hữu tool nào
    with open(universe_file, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    tool_to_owner = {tool['name']: tool['owner'] for tool in universe.get('tools', [])}
    agent_names = {agent['name'] for agent in universe.get('agents', [])}

    labels = []
    for i in range(len(id_to_name)):
        name = id_to_name[str(i)]
        if name in agent_names:
            # Mỗi Agent có nhãn riêng
            labels.append(f"Agent: {name}")
        elif name in tool_to_owner:
            # Mỗi Tool có nhãn riêng với owner
            owner = tool_to_owner[name]
            labels.append(f"Tool: {name} ({owner})")
        else: # It's a clinical state - group theo workflow phase
            # Phân loại states theo giai đoạn workflow
            if "Initial State" in name:
                labels.append("State: Initial Arrival")
            elif "Triage completed" in name:
                labels.append("State: Triage Phase")
            elif "Initial vitals assessed" in name:
                labels.append("State: Initial Assessment")
            elif "Initial medication dispensed" in name:
                labels.append("State: Initial Treatment")
            elif "Post-intervention vitals" in name:
                labels.append("State: Post-Treatment")
            elif "Full medication reconciled" in name:
                labels.append("State: Medication Review")
            elif "Final summary ready" in name:
                labels.append("State: Discharge Phase")
            else:
                labels.append("State: Other")
    return labels

def visualize_embedding_space(embedding_dir: str, universe_file: str, states_file: str, output_image_file: str):
    """Tải embedding space và trực quan hóa bằng t-SNE."""
    print(f"\n--- Visualizing Embedding Space from: {embedding_dir} ---")

    # 1. Tải Dữ liệu
    try:
        embeddings = np.load(os.path.join(embedding_dir, 'embeddings.npy'))
        with open(os.path.join(embedding_dir, 'id_to_name.json'), 'r') as f:
            id_to_name = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Embedding files not found in {embedding_dir}. Skipping.")
        return

    # 2. Tạo Nhãn
    labels = create_labels(id_to_name, universe_file, states_file)
    
    # 3. Chạy t-SNE
    print("Running t-SNE... (This can take a minute)")
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 4. Chuẩn bị dữ liệu để vẽ
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })

    # 5. Vẽ Biểu đồ
    print("Generating plot...")
    plt.figure(figsize=(18, 14))
    
    # Định nghĩa bảng màu với 18 màu hoàn toàn khác biệt
    distinct_colors = [
        '#FF0000',  # Đỏ tươi
        '#00FF00',  # Xanh lá neon
        '#0000FF',  # Xanh dương thuần
        '#FFD700',  # Vàng kim
        '#FF1493',  # Hồng đậm
        '#00CED1',  # Xanh ngọc
        '#FF4500',  # Cam đỏ
        '#9400D3',  # Tím đậm
        '#32CD32',  # Xanh lá lime
        '#FF69B4',  # Hồng nhạt
        '#1E90FF',  # Xanh dodger
        '#FFA500',  # Cam
        '#8B008B',  # Tím magenta đậm
        '#00FA9A',  # Xanh mint
        '#DC143C',  # Đỏ crimson
        '#4169E1',  # Xanh royal
        '#FF8C00',  # Cam đậm
        '#8A2BE2',  # Tím blue-violet
    ]
    
    unique_labels = sorted(df['label'].unique())
    color_palette = {}
    
    for idx, label in enumerate(unique_labels):
        color_palette[label] = distinct_colors[idx % len(distinct_colors)]
    
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='label',
        palette=color_palette,
        s=180,
        alpha=0.85,
        edgecolor='white',
        linewidth=1.5
    )
    
    plt.title(f't-SNE Visualization of VHAS Embedding Space\n({os.path.basename(embedding_dir)})', fontsize=18, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title='Entity Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu hình ảnh
    plt.savefig(output_image_file, bbox_inches='tight')
    print(f"Plot saved to '{output_image_file}'")
    plt.close()

if __name__ == "__main__":
    import os
    
    # Xác định đường dẫn tương đối từ vị trí script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', 'output', 'embedding_space_base')
    pretrained_dir = os.path.join(script_dir, '..', 'output', 'embedding_space_pretrained')
    
    # Đường dẫn đến các file định nghĩa
    UNIVERSE_FILE = os.path.join(script_dir, '..', '..', '..', 'vhas-demo', 'backend', 'vhas_universe.json')
    STATES_FILE = os.path.join(script_dir, '..', '..', 'clinical_states', 'clinical_states.json')
    
    # Output images
    OUTPUT_DIR = os.path.join(script_dir, '..', 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("GIAI ĐOẠN 1 - EVALUATION: t-SNE VISUALIZATION")
    print("="*80)
    print("\n📊 Câu hỏi Định tính: Bản đồ có trông giống quy trình lâm sàng hợp lý không?\n")
    
    # Chạy cho "base" model (Model A)
    print("\n🎨 Generating visualization for Model A...")
    visualize_embedding_space(
        embedding_dir=base_dir,
        universe_file=UNIVERSE_FILE,
        states_file=STATES_FILE,
        output_image_file=os.path.join(OUTPUT_DIR, 'tsne_model_a_base.png')
    )
    
    # Chạy cho "pretrained" model (Model B)
    print("\n🎨 Generating visualization for Model B...")
    visualize_embedding_space(
        embedding_dir=pretrained_dir,
        universe_file=UNIVERSE_FILE,
        states_file=STATES_FILE,
        output_image_file=os.path.join(OUTPUT_DIR, 'tsne_model_b_pretrained.png')
    )
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETED")
    print("="*80)
    print(f"\n📁 Images saved to: {OUTPUT_DIR}")
    print("   - tsne_model_a_base.png")
    print("   - tsne_model_b_pretrained.png")