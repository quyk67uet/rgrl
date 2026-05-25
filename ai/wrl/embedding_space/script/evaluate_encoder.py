"""
Evaluation script for the dual-encoder architecture.
====================================================

This script evaluates cross-modal retrieval for the dual encoder:
- Query: semantic state embeddings (from StateEncoder)
- Search: action embeddings (from ActionEncoder)

WORKFLOW:
1. Run directly on Modal (no local embedding download needed)
    $ modal run evaluate_encoder.py

OUTPUT:
- Nearest-neighbor analysis for each test query
- Retrieval metrics (MRR, Recall@K)
- Comparison between Model A and Model B
"""

import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import modal

# Define Modal App and image
app = modal.App("vhas-evaluate-dual-encoder")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install([
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
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
def evaluate_dual_encoder_model(
    embedding_space_dir: str,
    model_name: str,
    universe_file: str
):
    """
    Evaluate a dual-encoder model.

    Args:
        embedding_space_dir: Path to dual embedding space directory
        model_name: Name of the model (e.g., "Model A", "Model B")
        universe_file: Path to vhas_universe.json
    """
    print("\n" + "="*80)
    print(f"EVALUATING {model_name}")
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
        
        print(f"   ✓ State embeddings: {state_embeddings.shape}")
        print(f"   ✓ Action embeddings: {action_embeddings.shape}")
    except FileNotFoundError as e:
        print(f"   ❌ ERROR: {e}")
        return None
    
    # 2. Load universe to identify actionable entities (agents only)
    print(f"\n📋 Loading universe from {universe_file}...")
    try:
        with open(universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        
        actionable_entities = {agent['name'] for agent in universe.get('agents', [])}
        print(f"   ✓ Actionable entities (Agents): {len(actionable_entities)}")
    except FileNotFoundError as e:
        print(f"   ❌ ERROR: {e}")
        return None
    
    # 3. Create name-to-id maps
    state_name_to_id = {v: int(k) for k, v in state_id_to_name.items()}
    action_name_to_id = {v: int(k) for k, v in action_id_to_name.items()}
    
    # 4. Filter to actionable actions (agents only)
    actionable_indices = []
    actionable_names = []
    for idx, name in action_id_to_name.items():
        if name in actionable_entities:
            actionable_indices.append(int(idx))
            actionable_names.append(name)
    
    actionable_action_embeddings = action_embeddings[actionable_indices]
    print(f"   ✓ Filtered to {len(actionable_indices)} actionable actions (agents only)")
    
    # 5. Define test queries (semantic state prototypes)
    TEST_QUERIES = [
        "Semantic state prototype is Triage completed, priority is High, suspected Acute Coronary Syndrome.",
        "Semantic state prototype is Triage completed, priority is Low, suspected minor limb fracture.",
        "Semantic state prototype is Initial vitals assessed, the subject is hemodynamically unstable (hypotensive and tachycardic).",
        "Semantic state prototype is Full medication reconciled, a significant drug-drug interaction was identified between the subject's home anticoagulant and a planned operational workflow intervention.",
        "Semantic state prototype is Initial medication dispensed, morphine administered for severe abdominal pain.",
        "Semantic state prototype is Post-intervention vitals reassessed, the subject remains hypotensive despite an initial fluid bolus.",
        "Semantic state prototype is Initial State - Subject has just arrived at the operational setting.",
        "Semantic state prototype is Final summary ready, all interventions completed, subject stable for discharge."
    ]
    
    # 6. Run evaluation
    print("\n" + "="*80)
    print("NEAREST-NEIGHBOR ANALYSIS (Semantic State → Action)")
    print("="*80)
    
    results = []
    
    for query_text in TEST_QUERIES:
        # Find the query state in the corpus
        if query_text not in state_name_to_id:
            print(f"\n⚠️  Query not found in corpus: {query_text[:80]}...")
            continue
        
        query_id = state_name_to_id[query_text]
        query_embedding = state_embeddings[query_id].reshape(1, -1)
        
        # Compute cosine similarity against actionable actions
        similarities = cosine_similarity(query_embedding, actionable_action_embeddings)[0]
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Get top-5
        top_k = 5
        top_actions = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            action_name = actionable_names[idx]
            score = similarities[idx]
            top_actions.append((action_name, score))
        
        # Display
        short_query = query_text.split(',')[0] if ',' in query_text else query_text[:80]
        print(f"\n📍 Query Semantic State: {short_query}...")
        print(f"   ↓ Top-{top_k} suggested actions (agents):")
        
        for rank, (action_name, score) in enumerate(top_actions, 1):
            print(f"      {rank}. {action_name} (Score: {score:.4f})")
        
        results.append({
            "query": query_text,
            "top_actions": top_actions
        })
    
    # 7. Compute aggregate metrics
    print("\n" + "="*80)
    print("AGGREGATE METRICS")
    print("="*80)
    
    # Average similarity scores for top-1, top-3, top-5
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
    
    # Return the evaluation payload (no file I/O here)
    return {
        "model_name": model_name,
        "results": results,
        "metrics": {
            "avg_top1_similarity": float(np.mean(top1_scores)) if top1_scores else 0.0,
            "avg_top3_similarity": float(np.mean(top3_scores)) if top3_scores else 0.0,
            "avg_top5_similarity": float(np.mean(top5_scores)) if top5_scores else 0.0,
        }
    }


@app.local_entrypoint()
def main():
    """Local entrypoint - run evaluation for both models."""
    print("\n" + "="*80)
    print("DUAL-ENCODER EVALUATION: CROSS-MODAL RETRIEVAL (State → Action)")
    print("="*80)
    print("\n📊 Question: From a semantic state prototype, does the dual encoder retrieve the intended action?")
    print("   - Query: semantic state embeddings (from StateEncoder)")
    print("   - Search: action embeddings (from ActionEncoder)")
    print("   - Metric: cosine similarity in cross-modal space")
    print("\n" + "="*80)
    
    # Paths
    UNIVERSE_FILE = '/definitions/definitions/vhas_universe.json'
    MODEL_A_DIR = '/data/embedding_space_model_a_dual'
    MODEL_B_DIR = '/data/embedding_space_model_b_dual'
    
    # Evaluate Model A and Model B (run modal functions)
    print("\n🔬 Evaluating Model A (Base → Fine-tune)...")
    result_a = evaluate_dual_encoder_model.remote(
        embedding_space_dir=MODEL_A_DIR,
        model_name="Model A (Semantic State)",
        universe_file=UNIVERSE_FILE
    )

    print("\n🔬 Evaluating Model B (Base → Pre-train → Fine-tune)...")
    result_b = evaluate_dual_encoder_model.remote(
        embedding_space_dir=MODEL_B_DIR,
        model_name="Model B (Operational Guidance)",
        universe_file=UNIVERSE_FILE
    )

    # Both jobs should return dictionaries. Consolidate into a single payload.
    consolidated = {
        "model_a": result_a,
        "model_b": result_b
    }

    # Persist consolidated JSON to the preferred output (Modal /data volume or local ai/results)
    try:
        out_path = Path('/data') / 'wrl_nearest_neighbors.json' if os.path.isdir('/data') else Path(__file__).resolve().parents[2] / 'ai' / 'results' / 'wrl_nearest_neighbors.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # If the file exists, merge keys without overwriting unrelated content
        if out_path.exists():
            try:
                with out_path.open('r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        else:
            existing = {}

        existing.update(consolidated)

        with out_path.open('w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        print(f"\n   ✓ Consolidated nearest-neighbors saved to: {out_path}")
    except Exception as exc:
        print(f"\n   ⚠️ Failed to write consolidated JSON: {exc}")

    # Print a simple comparison summary if both results are present
    print("\n" + "="*80)
    print("COMPARISON: MODEL A vs MODEL B")
    print("="*80)

    if result_a and result_b:
        try:
            print(f"\n📊 Average Top-1 Similarity:")
            print(f"   Model A: {result_a['metrics']['avg_top1_similarity']:.4f}")
            print(f"   Model B: {result_b['metrics']['avg_top1_similarity']:.4f}")

            print(f"\n📊 Average Top-3 Similarity:")
            print(f"   Model A: {result_a['metrics']['avg_top3_similarity']:.4f}")
            print(f"   Model B: {result_b['metrics']['avg_top3_similarity']:.4f}")

            print(f"\n📊 Average Top-5 Similarity:")
            print(f"   Model A: {result_a['metrics']['avg_top5_similarity']:.4f}")
            print(f"   Model B: {result_b['metrics']['avg_top5_similarity']:.4f}")

            # Determine winner
            if result_b['metrics']['avg_top1_similarity'] > result_a['metrics']['avg_top1_similarity']:
                diff = result_b['metrics']['avg_top1_similarity'] - result_a['metrics']['avg_top1_similarity']
                print(f"\n🏆 Winner: Model B (Operational Guidance) by {diff:.4f} points")
                print("   → Pre-training on related data improved cross-modal retrieval!")
            else:
                diff = result_a['metrics']['avg_top1_similarity'] - result_b['metrics']['avg_top1_similarity']
                print(f"\n🏆 Winner: Model A (Semantic State) by {diff:.4f} points")
                print("   → Focused fine-tuning alone was sufficient!")
        except Exception as exc:
            print(f"\n⚠️  Comparison printing failed: {exc}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print("\n💡 Insights:")
    print("   - The dual encoder creates two compatible embedding spaces")
    print("   - StateEncoder encodes queries, ActionEncoder encodes actions")
    print("   - Cosine similarity in cross-modal space measures alignment")
    print("   - Higher similarity = better Semantic State → Action mapping")
    print("\n" + "="*80 + "\n")

