"""
Visualization script for the dual-encoder architecture.
======================================================

This script visualizes both embedding spaces produced by the dual encoder:
- Semantic State embedding space (from StateEncoder)
- Action embedding space (from ActionEncoder)

It uses t-SNE to reduce embeddings from 768D to 2D and renders the plots.

WORKFLOW:
1. Run directly on Modal (no local embedding download needed)
    $ modal run visualize_embeddings.py

OUTPUT:
- tsne_model_a_dual_state_space.png: Semantic State space for Model A
- tsne_model_a_dual_action_space.png: Action space for Model A
- tsne_model_b_dual_state_space.png: Semantic State space for Model B
- tsne_model_b_dual_action_space.png: Action space for Model B
- tsne_visualization.png: Combined view (Semantic State + Action)
- tsne_embeddings_2d.csv: Consolidated 2D coordinates for both models

Download with:
    modal volume get vhas-finetuned-output tsne_*.png ./results/
"""

import json
import os
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Modal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import modal

# Define Modal App and image
app = modal.App("vhas-visualize-dual-embeddings")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install([
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
    ])
)

# Connect Modal volumes
finetuned_output_vol = modal.Volume.from_name("vhas-finetuned-output", create_if_missing=False)
training_data_vol = modal.Volume.from_name("vhas-training-data", create_if_missing=False)


@app.function(
    image=image,
    timeout=1800,
    volumes={
        "/data": finetuned_output_vol,
        "/definitions": training_data_vol,
    },
)
def visualize_dual_encoder_model(
    embedding_space_dir: str,
    model_name: str,
    universe_file: str,
    output_prefix: str
):
    """
    Visualize Dual-Encoder embedding spaces.
    
    Args:
        embedding_space_dir: Path to dual embedding space directory
        model_name: Name of model (e.g., "Model A", "Model B")
        universe_file: Path to vhas_universe.json
        output_prefix: Prefix for output files (e.g., "model_a_dual")
    """
    print("\n" + "="*80)
    print(f"VISUALIZING {model_name}")
    print("="*80)
    
    # 1. Load embeddings and metadata
    print(f"\n📥 Loading embeddings from {embedding_space_dir}...")
    try:
        state_embeddings = np.load(os.path.join(embedding_space_dir, 'state_embeddings.npy'))
        action_embeddings = np.load(os.path.join(embedding_space_dir, 'action_embeddings.npy'))
        
        with open(os.path.join(embedding_space_dir, 'state_id_to_name.json'), 'r') as f:
            state_id_to_name = json.load(f)
        
        with open(os.path.join(embedding_space_dir, 'action_id_to_name.json'), 'r') as f:
            action_id_to_name = json.load(f)
        
        with open(os.path.join(embedding_space_dir, 'owner_map.json'), 'r') as f:
            owner_map = json.load(f)
        
        print(f"   ✓ State embeddings: {state_embeddings.shape}")
        print(f"   ✓ Action embeddings: {action_embeddings.shape}")
    except FileNotFoundError as e:
        print(f"   ❌ ERROR: {e}")
        return
    
    # 2. Load universe
    print(f"\n📋 Loading universe from {universe_file}...")
    try:
        with open(universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        
        agent_names = {agent['name'] for agent in universe.get('agents', [])}
        print(f"   ✓ Agents: {len(agent_names)}")
    except FileNotFoundError as e:
        print(f"   ❌ ERROR: {e}")
        return
    
    # 3. Create labels for actions
    action_labels = []
    for i in range(len(action_id_to_name)):
        name = action_id_to_name[str(i)]
        if name in agent_names:
            action_labels.append(f"Agent: {name}")
        elif name in owner_map:
            owner = owner_map[name]
            action_labels.append(f"Tool: {name} ({owner})")
        else:
            action_labels.append(f"Unknown: {name}")
    
    # 4. Create labels for states (group by operational phase)
    state_labels = []
    for i in range(len(state_id_to_name)):
        state_text = state_id_to_name[str(i)]
        if "Initial State" in state_text:
            state_labels.append("Semantic State: Initial Arrival")
        elif "Triage completed" in state_text:
            state_labels.append("Semantic State: Triage Phase")
        elif "Initial vitals assessed" in state_text:
            state_labels.append("Semantic State: Initial Assessment")
        elif "Initial medication dispensed" in state_text:
            state_labels.append("Semantic State: Initial Treatment")
        elif "Post-intervention vitals" in state_text:
            state_labels.append("Semantic State: Post-Treatment")
        elif "Full medication reconciled" in state_text:
            state_labels.append("Semantic State: Medication Review")
        elif "Final summary ready" in state_text:
            state_labels.append("Semantic State: Discharge Phase")
        else:
            state_labels.append("Semantic State: Other")
    
    # 5. Visualize action space
    print("\n🎨 Generating t-SNE visualization for Action Space...")
    visualize_single_space(
        embeddings=action_embeddings,
        labels=action_labels,
        title=f"Action Embedding Space - {model_name}",
        output_file=f"/data/tsne_{output_prefix}_action_space.png"
    )
    
    # 6. Visualize semantic state space
    print("\n🎨 Generating t-SNE visualization for Semantic State Space...")
    visualize_single_space(
        embeddings=state_embeddings,
        labels=state_labels,
        title=f"Semantic State Embedding Space - {model_name}",
        output_file=f"/data/tsne_{output_prefix}_state_space.png"
    )
    
    # 7. Visualize combined space (Semantic State + Action)
    print("\n🎨 Generating t-SNE visualization for Combined Space...")
    combined_embeddings = np.vstack([state_embeddings, action_embeddings])
    combined_labels = state_labels + action_labels
    
    visualize_single_space(
        embeddings=combined_embeddings,
        labels=combined_labels,
        title=f"Combined Embedding Space (Semantic State + Action) - {model_name}",
        output_file=f"/data/tsne_visualization.png" if os.path.isdir('/data') else Path(__file__).resolve().parents[2] / 'ai' / 'results' / 'figures' / 'tsne_visualization.png',
        figsize=(20, 16)
    )

    # Save consolidated 2D coordinates and labels to a shared CSV for downstream analysis.
    try:
        # Recompute t-SNE with a deterministic seed for exported coordinates.
        tsne = TSNE(n_components=2, perplexity=min(30, combined_embeddings.shape[0] - 1), random_state=42, max_iter=1000)
        emb2d = tsne.fit_transform(combined_embeddings)
        model_label = "Model A" if "Model A" in model_name else "Model B"
        df = pd.DataFrame({
            'x': emb2d[:, 0],
            'y': emb2d[:, 1],
            'label': combined_labels,
            'model': model_label,
        })

        # Prefer writing into container volume path (/data) when running on Modal, else write to repo results.
        out_csv = Path('/data') / 'tsne_embeddings_2d.csv' if os.path.isdir('/data') else Path(__file__).resolve().parents[2] / 'ai' / 'results' / 'tsne_embeddings_2d.csv'
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if out_csv.exists():
            existing_df = pd.read_csv(out_csv)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(out_csv, index=False, encoding='utf-8')
        print(f"   ✓ TSNE coords saved to {out_csv}")
    except Exception as exc:
        print(f"   ⚠️  Failed to save TSNE CSV: {exc}")
    
    print(f"\n✅ Visualizations saved:")
    print(f"   - /data/tsne_{output_prefix}_action_space.png")
    print(f"   - /data/tsne_{output_prefix}_state_space.png")
    print(f"   - /data/tsne_visualization.png")
    
    # Commit to volume
    from modal import Volume
    vol = Volume.from_name("vhas-finetuned-output")
    vol.commit()


def visualize_single_space(embeddings, labels, title, output_file, figsize=(18, 14)):
    """
    Helper function to visualize a single embedding space.
    """
    # Run t-SNE
    print(f"   Running t-SNE on {embeddings.shape[0]} embeddings...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, embeddings.shape[0] - 1),
        random_state=42,
        max_iter=1000
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Prepare data
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    # Define color palette
    distinct_colors = [
        '#FF0000',  # Red
        '#00FF00',  # Green
        '#0000FF',  # Blue
        '#FFD700',  # Gold
        '#FF1493',  # Deep Pink
        '#00CED1',  # Dark Turquoise
        '#FF4500',  # Orange Red
        '#9400D3',  # Dark Violet
        '#32CD32',  # Lime Green
        '#FF69B4',  # Hot Pink
        '#1E90FF',  # Dodger Blue
        '#FFA500',  # Orange
        '#8B008B',  # Dark Magenta
        '#00FA9A',  # Medium Spring Green
        '#DC143C',  # Crimson
        '#4169E1',  # Royal Blue
        '#FF8C00',  # Dark Orange
        '#8A2BE2',  # Blue Violet
        '#ADFF2F',  # Green Yellow
        '#FF6347',  # Tomato
    ]
    
    unique_labels = sorted(df['label'].unique())
    color_palette = {}
    for idx, label in enumerate(unique_labels):
        color_palette[label] = distinct_colors[idx % len(distinct_colors)]
    
    # Plot
    print(f"   Generating plot...")
    plt.figure(figsize=figsize)
    
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
    
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title='Entity Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"   ✓ Plot saved to {output_file}")
    plt.close()


@app.local_entrypoint()
def main():
    """Local entrypoint - run visualizations for both models."""
    print("\n" + "="*80)
    print("DUAL-ENCODER VISUALIZATION: t-SNE EMBEDDING SPACES")
    print("="*80)
    print("\n📊 Question: Does the dual encoder produce embedding spaces with a meaningful structure?")
    print("   - Semantic State Space: state prototypes grouped by operational phases")
    print("   - Action Space: agents and tools grouped by function")
    print("   - Combined Space: state and action representations show useful cross-modal alignment")
    print("\n" + "="*80)
    
    # Paths
    UNIVERSE_FILE = '/definitions/definitions/vhas_universe.json'
    MODEL_A_DIR = '/data/embedding_space_model_a_dual'
    MODEL_B_DIR = '/data/embedding_space_model_b_dual'
    
    # Visualize Model A
    print("\n🔬 Visualizing Model A (Base → Fine-tune)...")
    visualize_dual_encoder_model.remote(
        embedding_space_dir=MODEL_A_DIR,
        model_name="Model A (Specialty)",
        universe_file=UNIVERSE_FILE,
        output_prefix="model_a_dual"
    )
    
    # Visualize Model B
    print("\n🔬 Visualizing Model B (Base → Pre-train → Fine-tune)...")
    visualize_dual_encoder_model.remote(
        embedding_space_dir=MODEL_B_DIR,
        model_name="Model B (Comprehensive)",
        universe_file=UNIVERSE_FILE,
        output_prefix="model_b_dual"
    )
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\n📦 Download visualizations:")
    print("   $ modal volume get vhas-finetuned-output tsne_model_a_dual_action_space.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_a_dual_state_space.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_a_dual_combined.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_b_dual_action_space.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_b_dual_state_space.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_b_dual_combined.png ./results/")
    print("\n💡 Insights:")
    print("   - Action Space: agents should form separate clusters, with tools near their owning agents")
    print("   - Semantic State Space: state prototypes should cluster by operational phases")
    print("   - Combined Space: states should be close to matching actions (cross-modal alignment)")
    print("\n" + "="*80 + "\n")

