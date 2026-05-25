"""Comprehensive evaluation suite for the dual-encoder.

This script runs both evaluation and visualization steps for dual-encoder models.

Workflow:
1. Evaluation (nearest neighbors + metrics)
2. Visualization (t-SNE plots)
3. Comparison report

Run:
    $ modal run run_dual_evaluation.py

Outputs:
- Evaluation results (console)
- t-SNE visualizations (saved to volume)
- Comparison report (saved to volume)
"""

import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import modal

# Modal app definition
app = modal.App("vhas-dual-evaluation-suite")

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
def evaluate_dual_encoder_model(
    embedding_space_dir: str,
    model_name: str,
    universe_file: str
):
    """
    Evaluate a Dual-Encoder model.
    """
    print("\n" + "="*80)
    print(f"EVALUATING {model_name}")
    print("="*80)
    
    # 1. Load embeddings và metadata
    print(f"\n📥 Loading embeddings from {embedding_space_dir}...")
    try:
        state_embeddings = np.load(os.path.join(embedding_space_dir, 'state_embeddings.npy'))
        action_embeddings = np.load(os.path.join(embedding_space_dir, 'action_embeddings.npy'))
        
        with open(os.path.join(embedding_space_dir, 'state_id_to_name.json'), 'r') as f:
            state_id_to_name = json.load(f)
        
        with open(os.path.join(embedding_space_dir, 'action_id_to_name.json'), 'r') as f:
            action_id_to_name = json.load(f)
        
        print(f"   ✓ State embeddings: {state_embeddings.shape}")
        print(f"   ✓ Action embeddings: {action_embeddings.shape}")
    except FileNotFoundError as e:
        print(f"   ❌ ERROR: {e}")
        return None
    
    # 2. Load universe
    print(f"\n📋 Loading universe from {universe_file}...")
    try:
        with open(universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        
        actionable_entities = {agent['name'] for agent in universe.get('agents', [])}
        print(f"   ✓ Actionable entities (Agents): {len(actionable_entities)}")
    except FileNotFoundError as e:
        print(f"   ❌ ERROR: {e}")
        return None
    
    # 3. Create maps
    state_name_to_id = {v: int(k) for k, v in state_id_to_name.items()}
    
    # 4. Filter actionable actions
    actionable_indices = []
    actionable_names = []
    for idx, name in action_id_to_name.items():
        if name in actionable_entities:
            actionable_indices.append(int(idx))
            actionable_names.append(name)
    
    actionable_action_embeddings = action_embeddings[actionable_indices]
    print(f"   ✓ Filtered to {len(actionable_indices)} actionable actions")
    
    # 5. Test queries
    TEST_QUERIES = [
        "Patient state is Triage completed, priority is High, suspected Acute Coronary Syndrome.",
        "Patient state is Triage completed, priority is Low, suspected minor limb fracture.",
        "Patient state is Initial vitals assessed, patient is hemodynamically unstable (hypotensive and tachycardic).",
        "Patient state is Full medication reconciled, a significant drug-drug interaction was identified between the patient's home anticoagulant and a planned ED intervention.",
        "Patient state is Initial medication dispensed, morphine administered for severe abdominal pain.",
        "Patient state is Post-intervention vitals reassessed, patient remains hypotensive despite initial fluid bolus.",
        "Patient state is Initial State - Patient has just arrived at the emergency department.",
        "Patient state is Final summary ready, all interventions completed, patient stable for discharge."
    ]
    
    # 6. Run evaluation
    print("\n" + "="*80)
    print("NEAREST NEIGHBORS ANALYSIS (State → Action)")
    print("="*80)
    
    results = []
    
    for query_text in TEST_QUERIES:
        if query_text not in state_name_to_id:
            print(f"\n⚠️  Query not found: {query_text[:80]}...")
            continue
        
        query_id = state_name_to_id[query_text]
        query_embedding = state_embeddings[query_id].reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, actionable_action_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        
        top_k = 5
        top_actions = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            action_name = actionable_names[idx]
            score = similarities[idx]
            top_actions.append((action_name, score))
        
        short_query = query_text.split(',')[0] if ',' in query_text else query_text[:80]
        print(f"\n📍 Query: {short_query}...")
        print(f"   ↓ Top-{top_k} Actions:")
        
        for rank, (action_name, score) in enumerate(top_actions, 1):
            print(f"      {rank}. {action_name} (Score: {score:.4f})")
        
        results.append({
            "query": query_text,
            "top_actions": top_actions
        })
    
    # 7. Metrics
    print("\n" + "="*80)
    print("AGGREGATE METRICS")
    print("="*80)
    
    if results:
        top1_scores = [r['top_actions'][0][1] for r in results if r['top_actions']]
        top3_scores = [np.mean([a[1] for a in r['top_actions'][:3]]) for r in results if len(r['top_actions']) >= 3]
        top5_scores = [np.mean([a[1] for a in r['top_actions'][:5]]) for r in results if len(r['top_actions']) >= 5]
        
        print(f"\n📊 Average Cosine Similarity:")
        print(f"   Top-1: {np.mean(top1_scores):.4f} ± {np.std(top1_scores):.4f}")
        if top3_scores:
            print(f"   Top-3: {np.mean(top3_scores):.4f} ± {np.std(top3_scores):.4f}")
        if top5_scores:
            print(f"   Top-5: {np.mean(top5_scores):.4f} ± {np.std(top5_scores):.4f}")
    
    print("\n" + "="*80)
    print(f"✅ {model_name} EVALUATION COMPLETE")
    print("="*80)
    
    return {
        "model_name": model_name,
        "results": results,
        "metrics": {
            "avg_top1_similarity": float(np.mean(top1_scores)) if top1_scores else 0.0,
            "avg_top3_similarity": float(np.mean(top3_scores)) if top3_scores else 0.0,
            "avg_top5_similarity": float(np.mean(top5_scores)) if top5_scores else 0.0,
        }
    }


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
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    print("\n" + "="*80)
    print(f"VISUALIZING {model_name}")
    print("="*80)
    
    # Load data
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
    
    # Load universe
    with open(universe_file, 'r', encoding='utf-8') as f:
        universe = json.load(f)
    agent_names = {agent['name'] for agent in universe.get('agents', [])}
    
    # Create labels
    action_labels = []
    for i in range(len(action_id_to_name)):
        name = action_id_to_name[str(i)]
        if name in agent_names:
            action_labels.append(f"Agent: {name}")
        elif name in owner_map:
            action_labels.append(f"Tool: {name}")
        else:
            action_labels.append(f"Unknown: {name}")
    
    state_labels = []
    for i in range(len(state_id_to_name)):
        state_text = state_id_to_name[str(i)]
        if "Initial State" in state_text:
            state_labels.append("State: Initial")
        elif "Triage" in state_text:
            state_labels.append("State: Triage")
        elif "vitals" in state_text:
            state_labels.append("State: Assessment")
        elif "medication" in state_text:
            state_labels.append("State: Medication")
        elif "Final" in state_text:
            state_labels.append("State: Discharge")
        else:
            state_labels.append("State: Other")
    
    # Visualize combined space
    print("\n🎨 Generating t-SNE visualization...")
    combined_embeddings = np.vstack([state_embeddings, action_embeddings])
    combined_labels = state_labels + action_labels
    
    tsne = TSNE(n_components=2, perplexity=min(30, combined_embeddings.shape[0] - 1), random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': combined_labels
    })
    
    plt.figure(figsize=(18, 14))
    sns.scatterplot(data=df, x='x', y='y', hue='label', s=180, alpha=0.85, edgecolor='white', linewidth=1.5)
    plt.title(f'Dual-Encoder Embedding Space - {model_name}', fontsize=18, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title='Entity Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = f"/data/tsne_{output_prefix}_combined.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"   ✓ Saved to {output_file}")
    plt.close()
    
    # Commit
    from modal import Volume
    vol = Volume.from_name("vhas-finetuned-output")
    vol.commit()


@app.function(
    image=image,
    timeout=1800,
    volumes={
        "/data": finetuned_output_vol,
    },
)
def generate_comparison_report(result_a, result_b):
    """
    Tạo comparison report chi tiết giữa Model A và Model B.
    """
    import json
    from datetime import datetime
    
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("DUAL-ENCODER EVALUATION REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Model A Summary
    report.append("="*80)
    report.append("MODEL A: Base → Fine-tune (Specialist)")
    report.append("="*80)
    report.append(f"Average Top-1 Similarity: {result_a['metrics']['avg_top1_similarity']:.4f}")
    report.append(f"Average Top-3 Similarity: {result_a['metrics']['avg_top3_similarity']:.4f}")
    report.append(f"Average Top-5 Similarity: {result_a['metrics']['avg_top5_similarity']:.4f}")
    report.append("")
    
    # Model B Summary
    report.append("="*80)
    report.append("MODEL B: Base → Pre-train → Fine-tune (Generalist)")
    report.append("="*80)
    report.append(f"Average Top-1 Similarity: {result_b['metrics']['avg_top1_similarity']:.4f}")
    report.append(f"Average Top-3 Similarity: {result_b['metrics']['avg_top3_similarity']:.4f}")
    report.append(f"Average Top-5 Similarity: {result_b['metrics']['avg_top5_similarity']:.4f}")
    report.append("")
    
    # Comparison
    report.append("="*80)
    report.append("COMPARISON")
    report.append("="*80)
    
    top1_diff = result_b['metrics']['avg_top1_similarity'] - result_a['metrics']['avg_top1_similarity']
    top3_diff = result_b['metrics']['avg_top3_similarity'] - result_a['metrics']['avg_top3_similarity']
    top5_diff = result_b['metrics']['avg_top5_similarity'] - result_a['metrics']['avg_top5_similarity']
    
    report.append(f"Top-1 Difference (B - A): {top1_diff:+.4f}")
    report.append(f"Top-3 Difference (B - A): {top3_diff:+.4f}")
    report.append(f"Top-5 Difference (B - A): {top5_diff:+.4f}")
    report.append("")
    
    if top1_diff > 0:
        report.append("🏆 WINNER: Model B (Generalist)")
        report.append("   → Pre-training on ADP+T1 data improved cross-modal retrieval!")
    elif top1_diff < 0:
        report.append("🏆 WINNER: Model A (Specialist)")
        report.append("   → Domain-specific fine-tuning alone was sufficient!")
    else:
        report.append("🤝 TIE: Both models perform equally well")
    
    report.append("")
    report.append("="*80)
    report.append("KEY INSIGHTS")
    report.append("="*80)
    report.append("- Dual-Encoder tạo ra 2 embedding spaces compatible với nhau")
    report.append("- StateEncoder encode queries, ActionEncoder encode actions")
    report.append("- Cosine similarity in cross-modal space measures alignment")
    report.append("- Higher similarity = Better State → Action mapping")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open("/data/dual_encoder_evaluation_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    
    # Save JSON version (convert numpy types to Python types)
    def convert_to_python_types(obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_python_types(item) for item in obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "model_a": convert_to_python_types(result_a),
        "model_b": convert_to_python_types(result_b),
        "comparison": {
            "top1_diff": float(top1_diff),
            "top3_diff": float(top3_diff),
            "top5_diff": float(top5_diff),
            "winner": "Model B" if top1_diff > 0 else ("Model A" if top1_diff < 0 else "Tie")
        }
    }
    
    with open("/data/dual_encoder_evaluation_report.json", "w") as f:
        json.dump(json_report, f, indent=2)
    
    # Commit to volume
    from modal import Volume
    vol = Volume.from_name("vhas-finetuned-output")
    vol.commit()
    
    print("\n✅ Report saved to:")
    print("   - /data/dual_encoder_evaluation_report.txt")
    print("   - /data/dual_encoder_evaluation_report.json")


@app.local_entrypoint()
def main():
    """Run the full evaluation suite."""
    print("\n" + "="*80)
    print("DUAL-ENCODER COMPREHENSIVE EVALUATION SUITE")
    print("="*80)
    print("\nThis will run:")
    print("  1. Nearest Neighbors Analysis (State → Action)")
    print("  2. Aggregate Metrics (Cosine Similarity)")
    print("  3. t-SNE Visualizations (State, Action, Combined spaces)")
    print("  4. Comparison Report (Model A vs Model B)")
    print("\n" + "="*80)
    
    # Paths
    UNIVERSE_FILE = '/definitions/definitions/vhas_universe.json'
    MODEL_A_DIR = '/data/embedding_space_model_a_dual'
    MODEL_B_DIR = '/data/embedding_space_model_b_dual'
    
    # Step 1: Run Evaluation
    print("\n" + "="*80)
    print("STEP 1: EVALUATION (Nearest Neighbors + Metrics)")
    print("="*80)
    
    print("\n🔬 Evaluating Model A...")
    result_a = evaluate_dual_encoder_model.remote(
        embedding_space_dir=MODEL_A_DIR,
        model_name="Model A (Specialist)",
        universe_file=UNIVERSE_FILE,
    )
    
    print("\n🔬 Evaluating Model B...")
    result_b = evaluate_dual_encoder_model.remote(
        embedding_space_dir=MODEL_B_DIR,
        model_name="Model B (Generalist)",
        universe_file=UNIVERSE_FILE,
    )
    
    # Step 2: Run Visualization
    print("\n" + "="*80)
    print("STEP 2: VISUALIZATION (t-SNE Plots)")
    print("="*80)
    
    print("🎨 Visualizing Model A...")
    visualize_dual_encoder_model.remote(
        embedding_space_dir=MODEL_A_DIR,
        model_name="Model A (Specialist)",
        universe_file=UNIVERSE_FILE,
        output_prefix="model_a_dual",
    )
    
    print("🎨 Visualizing Model B...")
    visualize_dual_encoder_model.remote(
        embedding_space_dir=MODEL_B_DIR,
        model_name="Model B (Generalist)",
        universe_file=UNIVERSE_FILE,
        output_prefix="model_b_dual",
    )
    
    # Step 3: Generate Comparison Report
    print("\n" + "="*80)
    print("STEP 3: COMPARISON REPORT")
    print("="*80)
    
    if result_a and result_b:
        generate_comparison_report.remote(result_a, result_b)
    else:
        print("⚠️  Skipping report generation due to missing evaluation results")
    
    # Final Summary
    print("\n" + "="*80)
    print("✅ EVALUATION SUITE COMPLETE!")
    print("="*80)
    print("\n📦 Download results:")
    print("\n1. Comparison Report:")
    print("   $ modal volume get vhas-finetuned-output dual_encoder_evaluation_report.txt ./results/")
    print("   $ modal volume get vhas-finetuned-output dual_encoder_evaluation_report.json ./results/")
    print("\n2. Visualizations:")
    print("   $ modal volume get vhas-finetuned-output tsne_model_a_dual_action_space.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_a_dual_state_space.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_a_dual_combined.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_b_dual_action_space.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_b_dual_state_space.png ./results/")
    print("   $ modal volume get vhas-finetuned-output tsne_model_b_dual_combined.png ./results/")
    print("\n" + "="*80 + "\n")

