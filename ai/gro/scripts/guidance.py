# backend/guidance.py
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class GuidanceMechanism:
    def __init__(self, encoder_path: str, embedding_space_path: str):
        """
        Khởi tạo cơ chế hướng dẫn bằng cách tải tất cả các tài sản cần thiết.
        """
        print("--- Initializing Guidance Mechanism ---")
        
        # 1. Tải Encoder Model đã được fine-tune
        print(f"Loading encoder model from: {encoder_path}")
        self.encoder = SentenceTransformer(encoder_path)
        
        # 2. Tải các file từ Embedding Space
        print(f"Loading embedding space from: {embedding_space_path}")
        self.embeddings = np.load(os.path.join(embedding_space_path, 'embeddings.npy'))
        
        with open(os.path.join(embedding_space_path, 'id_to_name.json'), 'r') as f:
            self.id_to_name = json.load(f)
        self.name_to_id = {v: int(k) for k, v in self.id_to_name.items()}
            
        with open(os.path.join(embedding_space_path, 'owner_map.json'), 'r') as f:
            self.owner_map = json.load(f)
            
        # Tạo một set các tên agent để kiểm tra nhanh
        self.agent_names = set(self.owner_map.values())
        
        print("--- Guidance Mechanism Initialized Successfully ---")

    def propose_actions(self, current_state: str, top_k: int = 5, search_depth: int = 20) -> list[str]:
        """
        Implement thuật toán Tool-to-Agent Retrieval để đề xuất top-k agent.
        
        Args:
            current_state: Mô tả text của Trạng thái Lâm sàng hiện tại.
            top_k: Số lượng agent cuối cùng cần đề xuất.
            search_depth: Số lượng thực thể (agent+tool) cần truy xuất ban đầu.
        """
        
        # --- Bước 1: Lấy Vector Truy vấn ---
        query_vector = self.encoder.encode(current_state, convert_to_tensor=False).reshape(1, -1)
        
        # --- Bước 2: Tìm kiếm trên Toàn bộ "Bản đồ" ---
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # --- Bước 3: Lấy ra Top-N Ứng viên ban đầu ---
        # Lấy ra các thực thể có điểm số cao nhất, bao gồm cả agent và tool
        top_n_indices = np.argsort(similarities)[::-1][:search_depth]
        
        # --- Bước 4: "Di chuyển ngược" (Traversal) và Lọc ---
        proposed_agents = set()
        
        for idx in top_n_indices:
            if len(proposed_agents) >= top_k:
                break # Đã đủ số lượng cần thiết
                
            entity_name = self.id_to_name.get(str(idx))
            if not entity_name:
                continue
            
            # Kịch bản 1: Nếu thực thể là một Agent
            if entity_name in self.agent_names:
                proposed_agents.add(entity_name)
            
            # Kịch bản 2: Nếu thực thể là một Tool
            elif entity_name in self.owner_map:
                owner_agent = self.owner_map[entity_name]
                proposed_agents.add(owner_agent)
        
        # --- Bước 5: Trả về Kết quả ---
        return list(proposed_agents)