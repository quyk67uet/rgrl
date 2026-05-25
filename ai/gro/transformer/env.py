# transformer/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
import random

# Maximum history length for Transformer input
MAX_HISTORY_LENGTH = 20

# Import GuidanceMechanism (optional)
try:
    from guidance import GuidanceMechanism
    SINGLE_GUIDANCE_AVAILABLE = True
except ImportError:
    SINGLE_GUIDANCE_AVAILABLE = False

try:
    from guidance_dual import DualGuidanceMechanism
    DUAL_GUIDANCE_AVAILABLE = True
except ImportError:
    DUAL_GUIDANCE_AVAILABLE = False

if not SINGLE_GUIDANCE_AVAILABLE and not DUAL_GUIDANCE_AVAILABLE:
    print("Warning: No GuidanceMechanism available. Running without guidance.")

class ClinicalWorkflowEnv(gym.Env):
    """
    Simulation env for VHAS, following Gymnasium conventions.
    It is a lookup machine: it does not think and has no medical if/else logic.
    It only looks up state transitions from a knowledge base built from gold traces.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, 
                 encoder_model, 
                 scenarios_data_dir: str, 
                 kb_path: str = "data/simulation_kb.json",
                 guidance_encoder_path: str = None,
                 guidance_embedding_space_path: str = None,
                 guidance_action_encoder_path: str = None,  # NEW: For dual-encoder
                 use_guidance: bool = True,
                 use_dual_encoder: bool = False):  # NEW: Flag to use dual-encoder
        super().__init__()
        
        print("--- Initializing ClinicalWorkflowEnv (Lookup-Based) ---")
        
        # 1. Load the knowledge base with state transitions
        self.encoder = encoder_model
        with open(kb_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        print(f"   Loaded KB with {len(self.knowledge_base['state_transitions'])} state transitions")
        
        # 2. Initialize the guidance mechanism (optional)
        self.guidance = None
        self.use_dual_encoder = use_dual_encoder
        
        if use_guidance and guidance_encoder_path and guidance_embedding_space_path:
            try:
                if use_dual_encoder and DUAL_GUIDANCE_AVAILABLE and guidance_action_encoder_path:
                    # Use dual-encoder guidance
                    self.guidance = DualGuidanceMechanism(
                        state_encoder_path=guidance_encoder_path,
                        action_encoder_path=guidance_action_encoder_path,
                        embedding_space_path=guidance_embedding_space_path
                    )
                    print(f"   ✅ Dual-Encoder Guidance Mechanism initialized (Two-Tower search)")
                elif SINGLE_GUIDANCE_AVAILABLE:
                    # Use single-encoder guidance (legacy)
                    self.guidance = GuidanceMechanism(
                        encoder_path=guidance_encoder_path,
                        embedding_space_path=guidance_embedding_space_path
                    )
                    print(f"   ✅ Single-Encoder Guidance Mechanism initialized (Tool-to-Agent search)")
                else:
                    print(f"   ⚠️  Warning: No guidance mechanism available")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not initialize GuidanceMechanism: {e}")
                import traceback
                traceback.print_exc()
                self.guidance = None
        else:
            print(f"   ℹ️  Guidance Mechanism disabled (use_guidance={use_guidance})")
        
        # 3. Load expert traces from batch folders
        self.all_traces_data = self._load_expert_traces(scenarios_data_dir)
        
        # 4. Define the action space
        # Extract agent names from traces instead of AGENT_REGISTRY_SIM
        self.agent_names = ['TriageAgent', 'EHRAgent', 'DispensationAgent', 'ReconciliationAgent', 'SummaryAgent']
        self.num_agents = len(self.agent_names)
        self.action_space = spaces.Discrete(self.num_agents)
        self.action_to_name = {i: name for i, name in enumerate(self.agent_names)}
        self.name_to_action = {name: i for i, name in self.action_to_name.items()}
        
        # 5. Define the observation space
        # Changed: now returns sequences instead of single vectors
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.embedding_dim = embedding_dim
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(MAX_HISTORY_LENGTH, embedding_dim),  # Sequence shape
            dtype=np.float32
        )
        
        # 6. Episode tracking variables
        self._current_step = 0
        self.current_expert_trace = None  # Expert (state, action) sequence
        self.current_state_text = None    # Current state in the episode
        self.current_history = []  # History of (state, action) texts for the Transformer
        
        print(f"--- ClinicalWorkflowEnv Initialized Successfully with {len(self.all_traces_data)} traces. ---")

    def _load_expert_traces(self, scenarios_data_dir: str) -> list:
        """Load and parse all traces from batch folders."""
        all_traces = []
        print(f"Loading expert traces from {scenarios_data_dir}...")
        
        if not os.path.isdir(scenarios_data_dir):
            raise FileNotFoundError(f"Directory not found at: {scenarios_data_dir}")

        # Find the batch folders
        batch_folders = sorted([f for f in os.listdir(scenarios_data_dir) 
                               if f.startswith('batch_') and os.path.isdir(os.path.join(scenarios_data_dir, f))])
        
        for batch_folder in batch_folders:
            batch_num = batch_folder.split('_')[1]
            trace_file = os.path.join(scenarios_data_dir, batch_folder, f'traces_{batch_num}.json')
            if os.path.exists(trace_file):
                with open(trace_file, 'r', encoding='utf-8') as f:
                    traces = json.load(f)
                    all_traces.extend(traces)
        
        print(f"Loaded {len(all_traces)} traces from {len(batch_folders)} batches.")
        return all_traces

    def _get_obs(self) -> np.ndarray:
        """
        Generate sequence observation from current history.
        
        Returns:
            np.ndarray: Shape (MAX_HISTORY_LENGTH, embedding_dim)
                       Padded with zeros at the beginning if history < MAX_HISTORY_LENGTH
        """
        # Encode all items in history
        if not self.current_history:
            # Fallback: empty history (should not happen)
            embeddings = np.zeros((MAX_HISTORY_LENGTH, self.embedding_dim), dtype=np.float32)
            return embeddings
        
        # Batch encode all history items
        history_embeddings = self.encoder.encode(
            self.current_history,
            convert_to_numpy=True,
            show_progress_bar=False
        )  # Shape: (len(history), embedding_dim)
        
        # Pad to MAX_HISTORY_LENGTH
        current_length = len(history_embeddings)
        
        if current_length >= MAX_HISTORY_LENGTH:
            # Keep the last MAX_HISTORY_LENGTH items
            embeddings = history_embeddings[-MAX_HISTORY_LENGTH:]
        else:
            # Pad the front with zeros
            padding = np.zeros((MAX_HISTORY_LENGTH - current_length, self.embedding_dim), dtype=np.float32)
            embeddings = np.vstack([padding, history_embeddings])
        
        return embeddings.astype(np.float32)

    def get_candidate_actions(self, current_state_text: str = None, top_k: int = 5) -> dict:
        """
        Get candidate actions for the current state using the guidance mechanism.
        
        Args:
            current_state_text: State text to query (defaults to self.current_state_text)
            top_k: Number of candidates to propose
        
        Returns:
            dict: {
                'candidate_names': List[str],  # Agent names
                'candidate_indices': List[int],  # Action-space indices
                'action_mask': np.ndarray  # Binary action mask
            }
        """
        state_text = current_state_text or self.current_state_text
        
        if self.guidance is None:
            # Fallback: return all agents with equal priority
            return {
                'candidate_names': self.agent_names,
                'candidate_indices': list(range(self.num_agents)),
                'action_mask': np.ones(self.num_agents, dtype=bool)
            }
        
        # Use the guidance mechanism with tool-to-agent search
        candidate_agent_names = self.guidance.propose_actions(state_text, top_k=top_k)
        
        # Convert candidate names to action indices
        candidate_indices = []
        valid_candidates = []
        
        for agent_name in candidate_agent_names:
            if agent_name in self.name_to_action:
                candidate_indices.append(self.name_to_action[agent_name])
                valid_candidates.append(agent_name)
        
        # Create the action mask
        action_mask = np.zeros(self.num_agents, dtype=bool)
        if candidate_indices:
            action_mask[candidate_indices] = True
        else:
            # Fallback if no valid candidates are found
            action_mask[:] = True
            valid_candidates = self.agent_names
            candidate_indices = list(range(self.num_agents))
        
        return {
            'candidate_names': valid_candidates,
            'candidate_indices': candidate_indices,
            'action_mask': action_mask
        }



    def reset(self, seed=None, options=None):
        """Start a new episode."""
        super().reset(seed=seed)
        
        # 1. Pick a random trace from the reference set
        selected_trace = random.choice(self.all_traces_data)
        
        # 2. Extract the expert (state, action) sequence
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
        
        # 3. Initialize history with the initial state
        self.current_state_text = self.current_expert_trace[0][0]
        self.current_history = [self.current_state_text]  # Start with the initial state
        
        # 4. Generate the sequence observation
        observation = self._get_obs()
        info = {
            "current_state_text": self.current_state_text,
            "expert_action": self.current_expert_trace[0][1],
            "trace_length": len(self.current_expert_trace)
        }
        
        return observation, info

    def step(self, action: int):
        """Execute an action by looking up the state transition in the KB."""
        
        agent_name_to_call = self.action_to_name[action]
        
        # --- Step 1: Look up the state transition ---
        transition_key = json.dumps([self.current_state_text, agent_name_to_call], ensure_ascii=False)
        
        if transition_key in self.knowledge_base["state_transitions"]:
            # Found in the KB: observed action
            next_state_text = self.knowledge_base["state_transitions"][transition_key]
            status = "observed"
        else:
            # Not found: unseen exploratory action
            next_state_text = f"State: Unknown outcome from exploratory action '{agent_name_to_call}'."
            status = "exploratory"
        
        # Append the action and new state to history
        self.current_history.append(f"Action: {agent_name_to_call}")
        self.current_history.append(next_state_text)
        
        # Generate the sequence observation from full history
        observation = self._get_obs()
        
        # --- Step 2: Compute the multi-part reward ---
        
        # 2.1. Efficiency Reward (R_eff)
        # Lower the penalty to avoid encouraging early termination
        time_elapsed = 2
        R_eff = -time_elapsed
        
        # 2.2. Conformance Reward (R_conf) - "Imitation Reward"
        R_conf = 0
        if self._current_step < len(self.current_expert_trace):
            expert_action = self.current_expert_trace[self._current_step][1]
            if agent_name_to_call == expert_action:
                R_conf = 20  # Reward compliance
            else:
                R_conf = -5  # Lower the penalty to encourage exploration
        
        # Penalize exploratory actions not present in the KB
        if status == "exploratory":
            R_conf -= 15
        
        # 2.3. Terminal Reward (R_term)
        self._current_step += 1
        max_steps = 15
        done = (agent_name_to_call == "SummaryAgent") or (self._current_step >= max_steps)
        
        R_term = 0
        if done:
            # Check whether the workflow succeeded
            # Success means ending with SummaryAgent and following the expert trace
            is_successful = (agent_name_to_call == "SummaryAgent" and 
                             self._current_step == len(self.current_expert_trace))
            
            if is_successful:
                R_term = 300  # Large reward for successful completion
            else:
                R_term = -200  # Strong penalty for failure
                # Extra penalty for premature termination (cheat)
                if self._current_step < 3:
                    R_term -= 100  # Extra penalty if terminated too early
        
        # --- Aggregate reward ---
        # Weight settings (adjusted for better balance)
        w_eff = 0.1       # Increased from 0.05 -> 0.1
        w_conf = 2.0      # Increased from 1.5 -> 2.0 to prioritize conformance
        
        reward = (w_eff * R_eff) + (w_conf * R_conf) + R_term
        
        # Update the state for the next step
        self.current_state_text = next_state_text
        
        info = {
            "current_state_text": next_state_text,
            "status": status,
            "step": self._current_step,
            "rewards": {
                "efficiency": w_eff * R_eff,
                "conformance": w_conf * R_conf,
                "terminal": R_term,
                "total": reward
            }
        }
        
        return observation, reward, done, False, info