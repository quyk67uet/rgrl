# backend/guidance_dual.py
"""
Dual-Encoder Guidance Mechanism (Two-Tower Architecture)

ARCHITECTURE:
- StateEncoder: encodes clinical states into query vectors
- ActionEncoder: encodes actions (agents + tools) into the embedding space
- Guidance: query with StateEncoder and search in the ActionEncoder space

ADVANTAGES:
- Eliminates the "surface similarity curse"
- States are never compared against other states in search
- Searches directly in the action space
- Fast and accurate
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DualGuidanceMechanism:
    def __init__(
        self,
        state_encoder_path: str,
        action_encoder_path: str,
        embedding_space_path: str
    ):
        """
        Initialize the dual-encoder guidance mechanism.
        
        Args:
            state_encoder_path: Path to StateEncoder model
            action_encoder_path: Path to ActionEncoder model (not used for inference, only for reference)
            embedding_space_path: Path to dual embedding space directory
        """
        print("--- Initializing Dual-Encoder Guidance Mechanism ---")
        
        # 1. Load StateEncoder for query states
        print(f"Loading StateEncoder from: {state_encoder_path}")
        self.state_encoder = SentenceTransformer(state_encoder_path)
        print(f"   ✓ StateEncoder loaded ({self.state_encoder.get_sentence_embedding_dimension()} dims)")
        
        # 2. Load precomputed action embeddings
        print(f"Loading action embeddings from: {embedding_space_path}")
        self.action_embeddings = np.load(os.path.join(embedding_space_path, 'action_embeddings.npy'))
        print(f"   ✓ Action embeddings loaded: {self.action_embeddings.shape}")
        
        # 3. Load action metadata
        with open(os.path.join(embedding_space_path, 'action_id_to_name.json'), 'r') as f:
            self.action_id_to_name = json.load(f)
        self.action_name_to_id = {v: int(k) for k, v in self.action_id_to_name.items()}
        
        with open(os.path.join(embedding_space_path, 'owner_map.json'), 'r') as f:
            self.owner_map = json.load(f)
        
        # Cache agent names for fast lookup
        self.agent_names = set(self.owner_map.values())
        
        # 4. Load architecture info (optional)
        architecture_file = os.path.join(embedding_space_path, 'architecture_info.json')
        if os.path.exists(architecture_file):
            with open(architecture_file, 'r') as f:
                self.architecture_info = json.load(f)
            print(f"   ✓ Architecture: {self.architecture_info.get('architecture', 'Unknown')}")
        
        print(f"   ✓ Total actions in space: {len(self.action_id_to_name)}")
        print(f"   ✓ Total agents: {len(self.agent_names)}")
        print("--- Dual-Encoder Guidance Mechanism Initialized Successfully ---")

    def propose_actions(self, current_state: str, top_k: int = 5) -> list[str]:
        """
        Use dual-encoder guidance to propose top-k agents.
        
        WORKFLOW:
        1. Encode current_state with StateEncoder -> query_vector
        2. Compute cosine similarity between query_vector and action_embeddings
        3. Take the top-N actions with the highest scores
        4. Traverse from tools to agents if needed
        5. Return top-k agents
        
        Args:
            current_state: Text description of the current clinical state.
            top_k: Number of agents to return.
        
        Returns:
            List[str]: Proposed agent names
        """
        
        # --- Step 1: Encode the state with StateEncoder ---
        query_vector = self.state_encoder.encode(current_state, convert_to_tensor=False).reshape(1, -1)
        
        # --- Step 2: Search the action embedding space ---
        # Key point: search actions only, not states
        similarities = cosine_similarity(query_vector, self.action_embeddings)[0]
        
        # --- Step 3: Get the top-N actions ---
        # Only actions are in the space, so no large search depth is needed
        search_depth = min(top_k * 3, len(self.action_id_to_name))  # Auto-scale
        top_n_indices = np.argsort(similarities)[::-1][:search_depth]
        
        # --- Step 4: Traverse from tools to agents ---
        proposed_agents = set()
        
        for idx in top_n_indices:
            if len(proposed_agents) >= top_k:
                break
            
            action_name = self.action_id_to_name.get(str(idx))
            if not action_name:
                continue
            
            # Case 1: action is an agent
            if action_name in self.agent_names:
                proposed_agents.add(action_name)
            
            # Case 2: action is a tool -> map to its owner agent
            elif action_name in self.owner_map:
                owner_agent = self.owner_map[action_name]
                proposed_agents.add(owner_agent)
        
        # --- Step 5: Return the result ---
        return list(proposed_agents)
    
    def get_action_embedding(self, action_name: str) -> np.ndarray:
        """
        Get the embedding for a specific action (for debugging/analysis).
        
        Args:
            action_name: Name of an agent or tool
        
        Returns:
            np.ndarray: Embedding vector
        """
        if action_name not in self.action_name_to_id:
            raise ValueError(f"Action '{action_name}' not found in embedding space")
        
        action_id = self.action_name_to_id[action_name]
        return self.action_embeddings[action_id]
    
    def get_top_actions_with_scores(self, current_state: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Return the top-k actions with scores (for debugging/analysis).
        
        Returns:
            List[Tuple[str, float]]: [(action_name, similarity_score), ...]
        """
        query_vector = self.state_encoder.encode(current_state, convert_to_tensor=False).reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.action_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            action_name = self.action_id_to_name.get(str(idx))
            if action_name:
                results.append((action_name, float(similarities[idx])))
        
        return results

