# scripts/evaluate_encoder.py
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingEvaluator:
    def __init__(self, embedding_dir: str, universe_file: str):
        self.embedding_dir = embedding_dir
        self.universe_file = universe_file
        self.embeddings = None
        self.id_to_name = None
        self.name_to_id = None
        self.actionable_entities = None
        self._load_data()

    def _load_data(self):
        """Load necessary files from the embedding space and identify actionable entities."""
        try:
            self.embeddings = np.load(os.path.join(self.embedding_dir, 'embeddings.npy'))
            with open(os.path.join(self.embedding_dir, 'id_to_name.json'), 'r') as f:
                self.id_to_name = json.load(f)
            # Build reverse lookup map for fast lookup
            self.name_to_id = {v: int(k) for k, v in self.id_to_name.items()}

            # Load universe to determine actionable entities (agents only)
            with open(self.universe_file, 'r', encoding='utf-8') as f:
                universe = json.load(f)

            # Only agents are considered actionable; tools are not returned directly
            self.actionable_entities = {agent['name'] for agent in universe.get('agents', [])}
            
        except FileNotFoundError as e:
            raise IOError(f"Required files not found: {e}")

    def find_nearest_neighbors(self, query_name: str, top_k: int = 5):
        """
        Find the top-k nearest actionable actions (agents) for a state.
        Filter results to actionable entities (agents), excluding states and tools.
        """
        if query_name not in self.name_to_id:
            return [f"'{query_name}' not found in corpus."]
        
        query_id = self.name_to_id[query_name]
        query_vector = self.embeddings[query_id].reshape(1, -1)
        
        # Compute cosine similarity against the full corpus
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # --- FILTER STEP: consider only actionable entities (agents) ---
        action_scores = []
        for i in range(len(self.id_to_name)):
            entity_name = self.id_to_name[str(i)]
            if entity_name in self.actionable_entities:
                action_scores.append((similarities[i], i))
        
        # Sort by score descending
        sorted_actions = sorted(action_scores, key=lambda x: x[0], reverse=True)
        
        # Get top-k actionable neighbors
        top_indices = [idx for score, idx in sorted_actions[:top_k]]
        
        neighbors = []
        for idx in top_indices:
            name = self.id_to_name[str(idx)]
            score = similarities[idx]
            neighbors.append(f"{name} (Score: {score:.4f})")
            
        return neighbors

def run_evaluation(evaluator: EmbeddingEvaluator, query_set: list):
    """Run evaluation on a set of queries and print results."""
    print(f"\n--- Evaluating State → Action Mapping ---")
    for query in query_set:
        # Display shortened query for readability
        short_query = query.split(',')[0] if ',' in query else query[:80]
        print(f"\n📍 Query State: {short_query}...")
        print(f"   ↓ Suggested Actions (Agents):")
        
        neighbors = evaluator.find_nearest_neighbors(query, top_k=5)
        for i, neighbor in enumerate(neighbors):
            print(f"      {i+1}. {neighbor}")

if __name__ == "__main__":
    import os
    
    # Determine relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', 'output', 'embedding_space_base')
    pretrained_dir = os.path.join(script_dir, '..', 'output', 'embedding_space_pretrained')
    
    # Path to vhas_universe.json
    UNIVERSE_FILE = os.path.join(script_dir, '..', '..', '..', 'vhas-demo', 'backend', 'vhas_universe.json')
    
    # Test queries - CLINICAL STATES only (State → Action mapping)
    TEST_QUERIES = [
        "Patient state is Triage completed, priority is High, suspected Acute Coronary Syndrome.",
        "Patient state is Triage completed, priority is Low, suspected minor limb fracture.",
        "Patient state is Initial vitals assessed, patient is hemodynamically unstable (hypotensive and tachycardic).",
        "Patient state is Full medication reconciled, a significant drug-drug interaction was identified between the patient's home anticoagulant and a planned ED intervention.",
        "Patient state is Initial medication dispensed, morphine administered for severe abdominal pain."
    ]

    print("="*80)
    print("STAGE 1 - EVALUATION: NEAREST NEIGHBORS ANALYSIS")
    print("="*80)
    print("\nQuestion: From this state, which agents are reasonable choices?")
    print("   (Evaluating State → Action mapping only, excluding state noise)\n")
    
    # Run for the base model (Model A)
    try:
        evaluator_base = EmbeddingEvaluator(base_dir, UNIVERSE_FILE)
        print("\n" + "="*80)
        print("MODEL A: Base → Fine-tune (Specialist)")
        print("="*80)
        run_evaluation(evaluator_base, TEST_QUERIES)
    except IOError as e:
        print(f"❌ ERROR: {e}")
        
    # Run for the pretrained model (Model B)
    try:
        evaluator_pretrained = EmbeddingEvaluator(pretrained_dir, UNIVERSE_FILE)
        print("\n" + "="*80)
        print("MODEL B: Base → Pre-train → Fine-tune (Generalist)")
        print("="*80)
        run_evaluation(evaluator_pretrained, TEST_QUERIES)
    except IOError as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETED")
    print("="*80)
    print("\nInsights:")
    print("   - Nearest neighbors include only agents (actionable entities)")
    print("   - States and tools are filtered out from results")
    print("   - This reflects the State → Action task for the Orchestrator")
