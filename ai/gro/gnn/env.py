# gnn/env.py
"""
ClinicalWorkflowEnv for GNN-based RL training.

This environment returns HeteroData observations (graph-structured) instead of flat vectors.
It integrates with the VHAS_GNN_Wrapper to bridge to RSL-RL's TensorDict requirements.
"""

import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
import json
import random
import os

# Import guidance mechanism
try:
    import sys
    sys.path.insert(0, '../scripts')
    from guidance import GuidanceMechanism
    GUIDANCE_AVAILABLE = True
except ImportError:
    GUIDANCE_AVAILABLE = False
    print("Warning: GuidanceMechanism not available. Action masking will use all actions.")


class ClinicalWorkflowEnv(gym.Env):
    """
    Clinical Workflow Environment for GNN-based Orchestrator Training.
    
    Key Features:
    - Returns HeteroData observations (graph structure)
    - Dynamic action masking via GuidanceMechanism
    - Lookup-based state transitions from Knowledge Base
    - Multi-component reward function
    
    Observation Space: HeteroData with node types (agent, state, tool) and edge types
    Action Space: Discrete(num_agents)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        encoder_model: SentenceTransformer,
        scenarios_data_dir: str,
        kb_path: str = "data/simulation_kb.json",
        guidance_encoder_path: str = None,
        guidance_embedding_space_path: str = None,
        use_guidance: bool = True,
        top_k_guidance: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        print("=" * 80)
        print("🔧 Initializing ClinicalWorkflowEnv (GNN Mode)")
        print("=" * 80)
        
        self.device = device
        self.encoder = encoder_model
        self.embedding_dim = encoder_model.get_sentence_embedding_dimension()
        self.top_k_guidance = top_k_guidance
        
        # 1. Load Knowledge Base
        print(f"📚 Loading Knowledge Base from: {kb_path}")
        with open(kb_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        print(f"   ✓ Loaded {len(self.knowledge_base['state_transitions'])} state transitions")
        
        # 2. Initialize Guidance Mechanism (for action masking)
        self.guidance = None
        if use_guidance and GUIDANCE_AVAILABLE and guidance_encoder_path and guidance_embedding_space_path:
            try:
                print(f"🧭 Loading Guidance Mechanism...")
                self.guidance = GuidanceMechanism(
                    encoder_path=guidance_encoder_path,
                    embedding_space_path=guidance_embedding_space_path
                )
                print(f"   ✓ Guidance initialized (top_k={top_k_guidance})")
            except Exception as e:
                print(f"   ⚠️  Could not load Guidance: {e}")
                self.guidance = None
        else:
            print(f"   ℹ️  Guidance disabled (all actions allowed)")
        
        # 3. Load Expert Traces
        print(f"📖 Loading expert traces from: {scenarios_data_dir}")
        self.all_traces_data = self._load_expert_traces(scenarios_data_dir)
        print(f"   ✓ Loaded {len(self.all_traces_data)} traces")
        
        # 4. Define Action Space
        self.agent_names = ['TriageAgent', 'EHRAgent', 'DispensationAgent', 'ReconciliationAgent', 'SummaryAgent']
        self.num_agents = len(self.agent_names)
        self.num_actions = self.num_agents  # RSL-RL expects this attribute
        self.action_space = gym.spaces.Discrete(self.num_agents)
        self.action_to_name = {i: name for i, name in enumerate(self.agent_names)}
        self.name_to_action = {name: i for i, name in self.action_to_name.items()}
        
        # 5. Observation Space (HeteroData - described for compatibility)
        # Note: RSL-RL will use TensorDict from wrapper, but we define this for gym compatibility
        self.observation_space = gym.spaces.Dict({
            'graph': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # Placeholder
        })
        
        # 6. Episode tracking
        self._current_step = 0
        self.current_expert_trace = None
        self.current_state_text = None
        self.current_graph = None  # Current HeteroData observation
        
        # 7. VHAS Universe (Agent-Tool mappings)
        self._load_vhas_universe()
        
        # 8. PRE-COMPUTE AND CACHE EMBEDDINGS (OPTIMIZATION)
        # Agent and tool embeddings never change, so we cache them once
        print("🔄 Pre-computing agent and tool embeddings (optimization)...")
        self._cache_embeddings()
        print("   ✓ Embeddings cached")
        
        print("=" * 80)
        print("✅ ClinicalWorkflowEnv Initialized Successfully (GNN Mode)")
        print(f"   • {self.num_agents} agents")
        print(f"   • {len(self.all_traces_data)} training traces")
        print(f"   • Observation type: HeteroData (graph)")
        print(f"   • Action masking: {'ENABLED' if self.guidance else 'DISABLED'}")
        print("=" * 80)
    
    def _load_vhas_universe(self):
        """Load agent-tool mappings from vhas_universe.json"""
        # This defines which tools belong to which agents
        self.agent_tools = {
            'TriageAgent': ['calculate_ews_score'],
            'EHRAgent': ['assess_vitals', 'add_new_note'],
            'DispensationAgent': ['prepare_prescription'],
            'ReconciliationAgent': ['check_drug_interaction', 'get_medication_history'],
            'SummaryAgent': []  # No tools
        }
    
    def _cache_embeddings(self):
        """
        Pre-compute and cache agent and tool embeddings.
        These never change, so we only encode them once during initialization.
        This provides ~3x speedup in _construct_hetero_data.
        """
        # Cache agent embeddings
        agent_descriptions = [f"Agent: {name}" for name in self.agent_names]
        agent_embeddings_list = self.encoder.encode(
            agent_descriptions,
            convert_to_tensor=True,
            device=self.device,
            batch_size=len(agent_descriptions),  # Batch encode all agents at once
            show_progress_bar=False
        )
        self.agent_embeddings = {
            name: emb for name, emb in zip(self.agent_names, agent_embeddings_list)
        }
        
        # Cache tool embeddings
        tool_descriptions = []
        tool_to_agent_map = {}  # Maps tool index to agent index
        
        for agent_idx, agent_name in enumerate(self.agent_names):
            for tool_name in self.agent_tools.get(agent_name, []):
                tool_desc = f"Tool: {tool_name}"
                tool_descriptions.append(tool_desc)
                tool_to_agent_map[len(tool_descriptions) - 1] = agent_idx
        
        if tool_descriptions:
            tool_embeddings_list = self.encoder.encode(
                tool_descriptions,
                convert_to_tensor=True,
                device=self.device,
                batch_size=len(tool_descriptions),
                show_progress_bar=False
            )
            self.tool_embeddings = {
                desc.split(": ")[1]: emb  # Extract tool name from "Tool: {name}"
                for desc, emb in zip(tool_descriptions, tool_embeddings_list)
            }
            self.tool_to_agent_map = tool_to_agent_map
        else:
            self.tool_embeddings = {}
            self.tool_to_agent_map = {}
    
    def _load_expert_traces(self, scenarios_data_dir: str) -> list:
        """Load and parse all traces from batch folders."""
        all_traces = []
        
        if not os.path.isdir(scenarios_data_dir):
            raise FileNotFoundError(f"Directory not found: {scenarios_data_dir}")
        
        batch_folders = sorted([
            f for f in os.listdir(scenarios_data_dir)
            if f.startswith('batch_') and os.path.isdir(os.path.join(scenarios_data_dir, f))
        ])
        
        for batch_folder in batch_folders:
            batch_num = batch_folder.split('_')[1]
            trace_file = os.path.join(scenarios_data_dir, batch_folder, f'traces_{batch_num}.json')
            if os.path.exists(trace_file):
                with open(trace_file, 'r', encoding='utf-8') as f:
                    traces = json.load(f)
                    all_traces.extend(traces)
        
        return all_traces
    
    def _construct_hetero_data(self, state_text: str) -> HeteroData:
        """
        Construct a HeteroData graph observation from the current state.
        
        Graph Structure:
        - Nodes: current state, all possible agents, tools for valid agents
        - Edges: state->agents (triggers), agents->future states (produces), agents->tools (calls)
        
        This creates an "anticipatory flow anticipation network" (AFAN).
        """
        data = HeteroData()
        
        # --- 1. STATE NODE ---
        state_embedding = self.encoder.encode(state_text, convert_to_tensor=True, device=self.device)
        data['state'].x = state_embedding.unsqueeze(0)  # Shape: [1, embedding_dim]
        
        # --- 2. AGENT NODES (USE CACHED EMBEDDINGS) ---
        # Use pre-computed agent embeddings (much faster!)
        agent_embeddings = [self.agent_embeddings[name] for name in self.agent_names]
        data['agent'].x = torch.stack(agent_embeddings)  # Shape: [num_agents, embedding_dim]
        
        # --- 3. TOOL NODES (USE CACHED EMBEDDINGS) ---
        tool_list = []
        tool_to_agent_map = {}  # Maps tool index to agent index
        
        for agent_idx, agent_name in enumerate(self.agent_names):
            for tool_name in self.agent_tools.get(agent_name, []):
                # Use cached tool embedding
                emb = self.tool_embeddings[tool_name]
                tool_list.append(emb)
                tool_to_agent_map[len(tool_list) - 1] = agent_idx
        
        if tool_list:
            data['tool'].x = torch.stack(tool_list)  # Shape: [num_tools, embedding_dim]
        else:
            # Empty tool set (fallback)
            data['tool'].x = torch.zeros((0, self.embedding_dim), device=self.device)
        
        # --- 4. EDGES: STATE -> AGENT (triggers) ---
        # Connect current state to all agents (full bipartite)
        num_agents = len(self.agent_names)
        state_to_agent_edges = torch.tensor([
            [0] * num_agents,  # Source: always node 0 (the state)
            list(range(num_agents))  # Destination: all agent indices
        ], dtype=torch.long, device=self.device)
        data['state', 'triggers', 'agent'].edge_index = state_to_agent_edges
        
        # --- 5. EDGES: AGENT -> STATE (produces) ---
        # For now, we create a simple "anticipation" edge from each agent back to state
        # (In a more sophisticated version, this could predict future states)
        agent_to_state_edges = torch.tensor([
            list(range(num_agents)),  # Source: all agents
            [0] * num_agents  # Destination: state node
        ], dtype=torch.long, device=self.device)
        data['agent', 'produces', 'state'].edge_index = agent_to_state_edges
        
        # --- 6. EDGES: AGENT -> TOOL (calls) ---
        if tool_list:
            agent_to_tool_src = []
            agent_to_tool_dst = []
            for tool_idx, agent_idx in tool_to_agent_map.items():
                agent_to_tool_src.append(agent_idx)
                agent_to_tool_dst.append(tool_idx)
            
            agent_to_tool_edges = torch.tensor([
                agent_to_tool_src,
                agent_to_tool_dst
            ], dtype=torch.long, device=self.device)
            data['agent', 'calls', 'tool'].edge_index = agent_to_tool_edges
        else:
            # No tools - create empty edge index
            data['agent', 'calls', 'tool'].edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        
        return data
    
    def _get_action_mask(self) -> np.ndarray:
        """
        Generate action mask based on GuidanceMechanism.
        
        Returns:
            np.ndarray: Boolean mask of shape (num_agents,) where True = valid action
        
        Logic:
            1. If guidance available: Get top-k agent proposals
            2. Always include SummaryAgent (for termination)
            3. Fallback: Allow all actions if no guidance
        """
        mask = np.zeros(self.num_agents, dtype=np.uint8)
        
        if self.guidance is not None and self.current_state_text is not None:
            try:
                # Get top-k agent proposals from guidance
                candidate_names = self.guidance.propose_actions(
                    self.current_state_text,
                    top_k=self.top_k_guidance
                )
                
                # Mark proposed agents as valid
                for agent_name in candidate_names:
                    if agent_name in self.name_to_action:
                        agent_idx = self.name_to_action[agent_name]
                        mask[agent_idx] = 1
                
                # CRITICAL: Always allow SummaryAgent (for episode termination)
                if 'SummaryAgent' in self.name_to_action:
                    summary_idx = self.name_to_action['SummaryAgent']
                    mask[summary_idx] = 1
                
                # Fallback: If no valid actions, allow all
                if mask.sum() == 0:
                    mask[:] = 1
                    
            except Exception as e:
                print(f"⚠️  Error in action masking: {e}")
                mask[:] = 1  # Fallback to all actions
        else:
            # No guidance - allow all actions
            mask[:] = 1
        
        return mask
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to start a new episode.
        
        Returns:
            obs: HeteroData - Graph observation
            info: dict - Contains 'action_mask' and other metadata
        """
        super().reset(seed=seed)
        
        # 1. Select random expert trace
        selected_trace = random.choice(self.all_traces_data)
        
        # 2. Extract expert sequence (State, Action) pairs
        expert_sequence = []
        for span in selected_trace['spans']:
            attrs = span.get('attributes', {})
            if attrs.get('vhas.span.type') == 'orchestrator_decision':
                state = attrs.get('vhas.orchestrator.input_state')
                action = attrs.get('vhas.orchestrator.action_selected')
                if state and action:
                    expert_sequence.append((state, action))
        
        self.current_expert_trace = expert_sequence
        self._current_step = 0
        
        # 3. Get initial state and construct graph
        self.current_state_text = self.current_expert_trace[0][0]
        self.current_graph = self._construct_hetero_data(self.current_state_text)
        
        # 4. Generate action mask
        action_mask = self._get_action_mask()
        
        # 5. Build info dictionary
        info = {
            'action_mask': action_mask,
            'current_state_text': self.current_state_text,
            'expert_action': self.current_expert_trace[0][1],
            'trace_length': len(self.current_expert_trace),
            'episode_step': self._current_step
        }
        
        return self.current_graph, info
    
    def step(self, action: int):
        """
        Execute action and return next observation.
        
        Args:
            action: int - Index of agent to call
        
        Returns:
            obs: HeteroData - Next graph observation
            reward: float - Reward for this transition
            done: bool - Whether episode is complete
            truncated: bool - Whether episode was truncated (always False for us)
            info: dict - Contains 'action_mask' and reward breakdown
        """
        agent_name = self.action_to_name[action]
        
        # --- 1. LOOKUP STATE TRANSITION ---
        transition_key = json.dumps([self.current_state_text, agent_name], ensure_ascii=False)
        
        if transition_key in self.knowledge_base["state_transitions"]:
            next_state_text = self.knowledge_base["state_transitions"][transition_key]
            status = "observed"
        else:
            next_state_text = f"State: Unknown outcome from exploratory action '{agent_name}'."
            status = "exploratory"
        
        # --- 2. COMPUTE MULTI-COMPONENT REWARD ---
        
        # 2.1 Efficiency Reward
        time_elapsed = 2
        R_eff = -time_elapsed
        
        # 2.2 Conformance Reward (Imitation) - OPTIMIZED for better learning
        R_conf = 0
        if self._current_step < len(self.current_expert_trace):
            expert_action = self.current_expert_trace[self._current_step][1]
            if agent_name == expert_action:
                R_conf = 30  # INCREASED from 20 to 30: Stronger reward for following expert
            else:
                R_conf = -5  # Penalty for deviation (unchanged)
        
        # Penalty for exploratory actions - REDUCED to encourage exploration
        if status == "exploratory":
            R_conf -= 8  # REDUCED from -15 to -8: Less harsh penalty for exploration
        
        # 2.3 Terminal Reward
        self._current_step += 1
        max_steps = 15
        done = (agent_name == "SummaryAgent") or (self._current_step >= max_steps)
        
        R_term = 0
        if done:
            # Success: ended with SummaryAgent at correct step
            is_successful = (agent_name == "SummaryAgent" and 
                           self._current_step == len(self.current_expert_trace))
            
            if is_successful:
                R_term = 300  # Large reward for perfect completion (unchanged)
            else:
                # REDUCED penalties to prevent agent from learning to terminate early
                R_term = -100  # REDUCED from -200: Less harsh penalty for failure
                # Extra penalty for premature termination - REDUCED
                if self._current_step < 3:
                    R_term -= 50  # REDUCED from -100: Less penalty for premature termination
                # Reward for longer episodes (encourage not terminating too early)
                elif self._current_step >= 4:
                    R_term += 20  # Small bonus for longer episodes
        
        # Total weighted reward
        w_eff = 0.1
        w_conf = 2.0
        reward = (w_eff * R_eff) + (w_conf * R_conf) + R_term
        
        # --- 3. UPDATE STATE AND CONSTRUCT NEXT GRAPH ---
        self.current_state_text = next_state_text
        self.current_graph = self._construct_hetero_data(next_state_text)
        
        # --- 4. GENERATE ACTION MASK FOR NEXT STATE ---
        action_mask = self._get_action_mask()
        
        # --- 5. BUILD INFO DICTIONARY ---
        info = {
            'action_mask': action_mask,
            'current_state_text': next_state_text,
            'status': status,
            'episode_step': self._current_step,
            'rewards': {
                'efficiency': w_eff * R_eff,
                'conformance': w_conf * R_conf,
                'terminal': R_term,
                'total': reward
            }
        }
        
        truncated = False  # We don't use truncation
        
        return self.current_graph, reward, done, truncated, info
    
    def render(self, mode='human'):
        """Optional: Render the current state."""
        if mode == 'human':
            print(f"\n{'='*70}")
            print(f"Step: {self._current_step}")
            print(f"State: {self.current_state_text[:100]}...")
            print(f"Graph: {self.current_graph}")
            print(f"{'='*70}\n")
    
    def close(self):
        """Cleanup resources."""
        pass
