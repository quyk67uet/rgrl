# scripts/visualize_embeddings.py
import json
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_labels(id_to_name: dict, universe_file: str, states_file: str) -> list:
    """Create detailed labels for each entity in the corpus with sensible grouping."""
    
    # Load definitions to determine which agent owns each tool
    with open(universe_file, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    tool_to_owner = {tool['name']: tool['owner'] for tool in universe.get('tools', [])}
    agent_names = {agent['name'] for agent in universe.get('agents', [])}

    labels = []
    for i in range(len(id_to_name)):
        name = id_to_name[str(i)]
        if name in agent_names:
            # Each agent gets its own label
            labels.append(f"Agent: {name}")
        elif name in tool_to_owner:
            # Each tool gets its own label with owner
            owner = tool_to_owner[name]
            labels.append(f"Tool: {name} ({owner})")
        else:  # Clinical state - group by workflow phase
            # Classify states by workflow phase
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
    """Load embedding space and visualize it with t-SNE."""
    print(f"\n--- Visualizing embedding space from: {embedding_dir} ---")

    # 1. Load data
    try:
        embeddings = np.load(os.path.join(embedding_dir, 'embeddings.npy'))
        with open(os.path.join(embedding_dir, 'id_to_name.json'), 'r') as f:
            id_to_name = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Embedding files not found in {embedding_dir}. Skipping.")
        return
    # 2. Create labels
    labels = create_labels(id_to_name, universe_file, states_file)
    
    # 3. Run t-SNE
    print("Running t-SNE... (This can take a minute)")
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 4. Prepare data for plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })

    # 5. Plot
    print("Generating plot...")
    plt.figure(figsize=(18, 14))
    
    # Define a palette of 18 distinct colors
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
    
    # Save image
    plt.savefig(output_image_file, bbox_inches='tight')
    print(f"Plot saved to '{output_image_file}'")
    plt.close()

if __name__ == "__main__":
    import os

    # Determine relative paths from the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', 'output', 'embedding_space_base')
    pretrained_dir = os.path.join(script_dir, '..', 'output', 'embedding_space_pretrained')

    # Paths to definition files
    UNIVERSE_FILE = os.path.join(script_dir, '..', '..', '..', 'vhas-demo', 'backend', 'vhas_universe.json')
    STATES_FILE = os.path.join(script_dir, '..', '..', 'clinical_states', 'clinical_states.json')

    # Output images
    OUTPUT_DIR = os.path.join(script_dir, '..', 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*80)
    print("STAGE 1 - EVALUATION: t-SNE VISUALIZATION")
    print("="*80)
    print("\nQuestion: Does the embedding map reflect expected clinical workflow structure?\n")

    # Run for the base model (Model A)
    print("\n🎨 Generating visualization for Model A...")
    visualize_embedding_space(
        embedding_dir=base_dir,
        universe_file=UNIVERSE_FILE,
        states_file=STATES_FILE,
        output_image_file=os.path.join(OUTPUT_DIR, 'tsne_model_a_base.png')
    )

    # Run for the pretrained model (Model B)
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
    print(f"\nImages saved to: {OUTPUT_DIR}")
    print("   - tsne_model_a_base.png")
    print("   - tsne_model_b_pretrained.png")