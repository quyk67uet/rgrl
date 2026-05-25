# backend/guidance.py
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class GuidanceMechanism:
    def __init__(self, encoder_path: str, embedding_space_path: str):
        """
        Initialize the guidance mechanism by loading all required assets.
        """
        print("--- Initializing Guidance Mechanism ---")
        
        # 1. Load the fine-tuned encoder model
        print(f"Loading encoder model from: {encoder_path}")
        self.encoder = SentenceTransformer(encoder_path)
        
        # 2. Load files from the embedding space
        print(f"Loading embedding space from: {embedding_space_path}")
        self.embeddings = np.load(os.path.join(embedding_space_path, 'embeddings.npy'))
        
        with open(os.path.join(embedding_space_path, 'id_to_name.json'), 'r') as f:
            self.id_to_name = json.load(f)
        self.name_to_id = {v: int(k) for k, v in self.id_to_name.items()}
            
        with open(os.path.join(embedding_space_path, 'owner_map.json'), 'r') as f:
            self.owner_map = json.load(f)
            
        # Cache agent names for fast lookup
        self.agent_names = set(self.owner_map.values())
        
        print("--- Guidance Mechanism Initialized Successfully ---")

    def propose_actions(self, current_state: str, top_k: int = 5, search_depth: int = 20) -> list[str]:
        """
        Use tool-to-agent retrieval to propose top-k agents.
        
        Args:
            current_state: Text description of the current clinical state.
            top_k: Number of agents to return.
            search_depth: Number of initial entities to retrieve.
        """
        
        # --- Step 1: Build the query vector ---
        query_vector = self.encoder.encode(current_state, convert_to_tensor=False).reshape(1, -1)
        
        # --- Step 2: Search the full embedding space ---
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # --- Step 3: Get the initial top-N candidates ---
        # Includes both agents and tools
        top_n_indices = np.argsort(similarities)[::-1][:search_depth]
        
        # --- Step 4: Traverse back to agents and filter ---
        proposed_agents = set()
        
        for idx in top_n_indices:
            if len(proposed_agents) >= top_k:
                break # Enough results
                
            entity_name = self.id_to_name.get(str(idx))
            if not entity_name:
                continue
            
            # Case 1: the entity is an agent
            if entity_name in self.agent_names:
                proposed_agents.add(entity_name)
            
            # Case 2: the entity is a tool
            elif entity_name in self.owner_map:
                owner_agent = self.owner_map[entity_name]
                proposed_agents.add(owner_agent)
        
        # --- Step 5: Return the result ---
        return list(proposed_agents)