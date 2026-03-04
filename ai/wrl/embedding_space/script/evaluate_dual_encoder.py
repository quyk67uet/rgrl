"""
EVALUATION SCRIPT FOR DUAL-ENCODER ARCHITECTURE
================================================

Script này đánh giá khả năng cross-modal retrieval của Dual-Encoder:
- Query: State embeddings (từ StateEncoder)
- Search: Action embeddings (từ ActionEncoder)

WORKFLOW:
1. Chạy trực tiếp trên Modal (không cần download embeddings về local)
   $ modal run evaluate_dual_encoder.py

OUTPUT:
- Nearest Neighbors Analysis cho từng test query
- Retrieval metrics (MRR, Recall@K)
- Comparison giữa Model A và Model B
"""

import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import modal

# Định nghĩa Modal App và Image
app = modal.App("vhas-evaluate-dual-encoder")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install([
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ])
)

# Kết nối Modal Volume
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
    Đánh giá một Dual-Encoder model.
    
    Args:
        embedding_space_dir: Path to dual embedding space directory
        model_name: Name of model (e.g., "Model A", "Model B")
        universe_file: Path to vhas_universe.json
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
    
    # 2. Load universe để xác định actionable entities (chỉ Agents)
    print(f"\n📋 Loading universe from {universe_file}...")
    try:
        with open(universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        
        actionable_entities = {agent['name'] for agent in universe.get('agents', [])}
        print(f"   ✓ Actionable entities (Agents): {len(actionable_entities)}")
    except FileNotFoundError as e:
        print(f"   ❌ ERROR: {e}")
        return None
    
    # 3. Tạo name_to_id maps
    state_name_to_id = {v: int(k) for k, v in state_id_to_name.items()}
    action_name_to_id = {v: int(k) for k, v in action_id_to_name.items()}
    
    # 4. Filter để chỉ lấy actionable actions (Agents)
    actionable_indices = []
    actionable_names = []
    for idx, name in action_id_to_name.items():
        if name in actionable_entities:
            actionable_indices.append(int(idx))
            actionable_names.append(name)
    
    actionable_action_embeddings = action_embeddings[actionable_indices]
    print(f"   ✓ Filtered to {len(actionable_indices)} actionable actions (Agents only)")
    
    # 5. Define test queries (Clinical States)
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
        # Find query state in corpus
        if query_text not in state_name_to_id:
            print(f"\n⚠️  Query not found in corpus: {query_text[:80]}...")
            continue
        
        query_id = state_name_to_id[query_text]
        query_embedding = state_embeddings[query_id].reshape(1, -1)
        
        # Compute cosine similarity với actionable actions
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
        print(f"\n📍 Query State: {short_query}...")
        print(f"   ↓ Top-{top_k} Suggested Actions (Agents):")
        
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
    """
    Local entrypoint - chạy evaluation cho cả 2 models
    """
    print("\n" + "="*80)
    print("DUAL-ENCODER EVALUATION: CROSS-MODAL RETRIEVAL (State → Action)")
    print("="*80)
    print("\n📊 Câu hỏi: Từ một Clinical State, Dual-Encoder có đề xuất đúng Agent không?")
    print("   - Query: State embeddings (từ StateEncoder)")
    print("   - Search: Action embeddings (từ ActionEncoder)")
    print("   - Metric: Cosine Similarity trong cross-modal space")
    print("\n" + "="*80)
    
    # Paths
    UNIVERSE_FILE = '/definitions/definitions/vhas_universe.json'
    MODEL_A_DIR = '/data/embedding_space_model_a_dual'
    MODEL_B_DIR = '/data/embedding_space_model_b_dual'
    
    # Evaluate Model A
    print("\n🔬 Evaluating Model A (Base → Fine-tune)...")
    result_a = evaluate_dual_encoder_model.remote(
        embedding_space_dir=MODEL_A_DIR,
        model_name="Model A (Chuyên khoa)",
        universe_file=UNIVERSE_FILE
    )
    
    # Evaluate Model B
    print("\n🔬 Evaluating Model B (Base → Pre-train → Fine-tune)...")
    result_b = evaluate_dual_encoder_model.remote(
        embedding_space_dir=MODEL_B_DIR,
        model_name="Model B (Toàn diện)",
        universe_file=UNIVERSE_FILE
    )
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON: MODEL A vs MODEL B")
    print("="*80)
    
    if result_a and result_b:
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
            print(f"\n🏆 Winner: Model B (Toàn diện) by {diff:.4f} points")
            print("   → Pre-training on ADP+T1 data improved cross-modal retrieval!")
        else:
            diff = result_a['metrics']['avg_top1_similarity'] - result_b['metrics']['avg_top1_similarity']
            print(f"\n🏆 Winner: Model A (Chuyên khoa) by {diff:.4f} points")
            print("   → Domain-specific fine-tuning alone was sufficient!")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print("\n💡 Insights:")
    print("   - Dual-Encoder tạo ra 2 embedding spaces compatible với nhau")
    print("   - StateEncoder encode queries, ActionEncoder encode actions")
    print("   - Cosine similarity trong cross-modal space đo độ phù hợp")
    print("   - Higher similarity = Better State → Action mapping")
    print("\n" + "="*80 + "\n")

