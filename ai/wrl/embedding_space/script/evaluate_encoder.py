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
        """Tải các file cần thiết từ embedding space và xác định actionable entities."""
        try:
            self.embeddings = np.load(os.path.join(self.embedding_dir, 'embeddings.npy'))
            with open(os.path.join(self.embedding_dir, 'id_to_name.json'), 'r') as f:
                self.id_to_name = json.load(f)
            # Tạo map ngược để tra cứu nhanh
            self.name_to_id = {v: int(k) for k, v in self.id_to_name.items()}
            
            # Tải universe để xác định các actionable entities (chỉ Agents)
            with open(self.universe_file, 'r', encoding='utf-8') as f:
                universe = json.load(f)
            
            # Chỉ Agents là actionable - Tools không được trả về trực tiếp
            self.actionable_entities = {agent['name'] for agent in universe.get('agents', [])}
            
        except FileNotFoundError as e:
            raise IOError(f"Required files not found: {e}")

    def find_nearest_neighbors(self, query_name: str, top_k: int = 5):
        """
        Tìm k hành động (Agents) gần nhất cho một state.
        LỌC để chỉ trả về actionable entities (Agents), loại bỏ States và Tools.
        """
        if query_name not in self.name_to_id:
            return [f"'{query_name}' not found in corpus."]
        
        query_id = self.name_to_id[query_name]
        query_vector = self.embeddings[query_id].reshape(1, -1)
        
        # Tính cosine similarity với toàn bộ corpus
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # --- BƯỚC LỌC: Chỉ xem xét các actionable entities (Agents) ---
        action_scores = []
        for i in range(len(self.id_to_name)):
            entity_name = self.id_to_name[str(i)]
            if entity_name in self.actionable_entities:
                action_scores.append((similarities[i], i))
        
        # Sắp xếp theo điểm số giảm dần
        sorted_actions = sorted(action_scores, key=lambda x: x[0], reverse=True)
        
        # Lấy top_k kết quả
        top_indices = [idx for score, idx in sorted_actions[:top_k]]
        
        neighbors = []
        for idx in top_indices:
            name = self.id_to_name[str(idx)]
            score = similarities[idx]
            neighbors.append(f"{name} (Score: {score:.4f})")
            
        return neighbors

def run_evaluation(evaluator: EmbeddingEvaluator, query_set: list):
    """Chạy đánh giá trên một bộ câu hỏi và in kết quả."""
    print(f"\n--- Evaluating State → Action Mapping ---")
    for query in query_set:
        # Hiển thị query rút gọn để dễ đọc
        short_query = query.split(',')[0] if ',' in query else query[:80]
        print(f"\n📍 Query State: {short_query}...")
        print(f"   ↓ Suggested Actions (Agents):")
        
        neighbors = evaluator.find_nearest_neighbors(query, top_k=5)
        for i, neighbor in enumerate(neighbors):
            print(f"      {i+1}. {neighbor}")

if __name__ == "__main__":
    import os
    
    # Xác định đường dẫn tương đối từ vị trí script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', 'output', 'embedding_space_base')
    pretrained_dir = os.path.join(script_dir, '..', 'output', 'embedding_space_pretrained')
    
    # Đường dẫn đến vhas_universe.json
    UNIVERSE_FILE = os.path.join(script_dir, '..', '..', '..', 'vhas-demo', 'backend', 'vhas_universe.json')
    
    # Bộ câu hỏi kiểm thử - CHỈ CLINICAL STATES (State → Action mapping)
    TEST_QUERIES = [
        "Patient state is Triage completed, priority is High, suspected Acute Coronary Syndrome.",
        "Patient state is Triage completed, priority is Low, suspected minor limb fracture.",
        "Patient state is Initial vitals assessed, patient is hemodynamically unstable (hypotensive and tachycardic).",
        "Patient state is Full medication reconciled, a significant drug-drug interaction was identified between the patient's home anticoagulant and a planned ED intervention.",
        "Patient state is Initial medication dispensed, morphine administered for severe abdominal pain."
    ]

    print("="*80)
    print("GIAI ĐOẠN 1 - EVALUATION: NEAREST NEIGHBORS ANALYSIS")
    print("="*80)
    print("\n📊 Câu hỏi Định lượng: Từ State này, những Agent nào là lựa chọn hợp lý?")
    print("   (Chỉ đánh giá State → Action mapping, loại bỏ State noise)\n")
    
    # Chạy cho "base" model (Model A)
    try:
        evaluator_base = EmbeddingEvaluator(base_dir, UNIVERSE_FILE)
        print("\n" + "="*80)
        print("MODEL A: Base → Fine-tune (Specialist)")
        print("="*80)
        run_evaluation(evaluator_base, TEST_QUERIES)
    except IOError as e:
        print(f"❌ ERROR: {e}")
        
    # Chạy cho "pretrained" model (Model B)
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
    print("\n💡 Insights:")
    print("   - Nearest neighbors chỉ bao gồm Agents (actionable entities)")
    print("   - States và Tools đã được lọc ra khỏi kết quả")
    print("   - Điều này phản ánh đúng nhiệm vụ State → Action của Orchestrator")
