# simulation/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
import random
from typing import Any

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
    Môi trường Mô phỏng cho VHAS, tuân thủ chuẩn Gymnasium.
    Môi trường này là một "Cỗ máy Tra cứu" (Lookup Machine) - nó KHÔNG suy nghĩ,
    KHÔNG có logic if/else về y tế. Nó chỉ tra cứu state transitions từ Knowledge Base
    được biên dịch từ 2000 "dấu vết vàng".
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, 
                 encoder_model, 
                 traces_filepath: str, 
                 kb_path: str = "data/simulation_kb.json",
                 guidance_encoder_path: str = None,
                 guidance_embedding_space_path: str = None,
                 guidance_action_encoder_path: str = None,  # NEW: For dual-encoder
                 use_guidance: bool = True,
                 use_dual_encoder: bool = False):  # NEW: Flag to use dual-encoder
        super().__init__()
        
        print("--- Initializing ClinicalWorkflowEnv (Lookup-Based) ---")
        
        # 1. Tải Knowledge Base với state transitions
        self.encoder = encoder_model
        with open(kb_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        print(f"   Loaded KB with {len(self.knowledge_base['state_transitions'])} state transitions")
        
        # 2. Khởi tạo Guidance Mechanism (optional)
        self.guidance = None
        self.use_dual_encoder = use_dual_encoder
        
        if use_guidance and guidance_encoder_path and guidance_embedding_space_path:
            try:
                if use_dual_encoder and DUAL_GUIDANCE_AVAILABLE and guidance_action_encoder_path:
                    # Use Dual-Encoder Guidance
                    self.guidance = DualGuidanceMechanism(
                        state_encoder_path=guidance_encoder_path,
                        action_encoder_path=guidance_action_encoder_path,
                        embedding_space_path=guidance_embedding_space_path
                    )
                    print(f"   ✅ Dual-Encoder Guidance Mechanism initialized (Two-Tower search)")
                elif SINGLE_GUIDANCE_AVAILABLE:
                    # Use Single-Encoder Guidance (legacy)
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
        
        # 3. Tải expert traces từ unified JSON file
        self.all_traces_data = self._load_expert_traces(traces_filepath)
        
        # 4. Định nghĩa Không gian Hành động (Action Space)
        # Trích xuất agent names từ traces thay vì từ AGENT_REGISTRY_SIM
        self.agent_names = ['TriageAgent', 'EHRAgent', 'DispensationAgent', 'ReconciliationAgent', 'SummaryAgent']
        self.num_agents = len(self.agent_names)
        self.action_space = spaces.Discrete(self.num_agents)
        self.action_to_name = {i: name for i, name in enumerate(self.agent_names)}
        self.name_to_action = {name: i for i, name in self.action_to_name.items()}
        
        # 5. Định nghĩa Không gian Trạng thái (Observation Space)
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32)
        
        # 6. Các biến theo dõi của một "episode"
        self._current_step = 0
        self.current_expert_trace = None  # Chuỗi (State, Action) của chuyên gia
        self.current_state_text = None    # State hiện tại trong episode
        
        print(f"--- ClinicalWorkflowEnv Initialized Successfully with {len(self.all_traces_data)} traces. ---")

    def _load_expert_traces(self, traces_filepath: str) -> list:
        """Tải và parse expert traces từ một unified JSON file."""
        print(f"Loading expert traces from {traces_filepath}...")

        if not os.path.isfile(traces_filepath):
            raise FileNotFoundError(f"Trace file not found at: {traces_filepath}")

        with open(traces_filepath, 'r', encoding='utf-8') as f:
            traces = json.load(f)

        if not isinstance(traces, list):
            raise ValueError("Trace file must contain a JSON list of traces.")

        print(f"Loaded {len(traces)} traces from unified file.")
        return traces

    def _get_obs(self, clinical_state_text: str) -> np.ndarray:
        """Biến một chuỗi trạng thái thành một vector observation."""
        return self.encoder.encode(clinical_state_text)

    def get_candidate_actions(
        self,
        current_state_text: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Lấy danh sách candidate actions cho state hiện tại sử dụng Guidance Mechanism.
        
        Args:
            current_state_text: State text để query (nếu None, dùng self.current_state_text)
            top_k: Số lượng candidates đề xuất
        
        Returns:
            dict: {
                'candidate_names': List[str],  # Agent names
                'candidate_indices': List[int],  # Action space indices
                'action_mask': np.ndarray  # Binary mask for action space
            }
        """
        state_text = current_state_text or self.current_state_text
        
        if self.guidance is None:
            # Fallback: trả về tất cả agents với equal priority
            return {
                'candidate_names': self.agent_names,
                'candidate_indices': list(range(self.num_agents)),
                'action_mask': np.ones(self.num_agents, dtype=bool)
            }
        
        # Sử dụng Guidance Mechanism với Tool-to-Agent search
        candidate_agent_names = self.guidance.propose_actions(state_text, top_k=top_k)
        
        # Convert candidate names to action indices
        candidate_indices = []
        valid_candidates = []
        
        for agent_name in candidate_agent_names:
            if agent_name in self.name_to_action:
                candidate_indices.append(self.name_to_action[agent_name])
                valid_candidates.append(agent_name)
        
        # Create action mask
        action_mask = np.zeros(self.num_agents, dtype=bool)
        if candidate_indices:
            action_mask[candidate_indices] = True
        else:
            # Fallback nếu không có valid candidates
            action_mask[:] = True
            valid_candidates = self.agent_names
            candidate_indices = list(range(self.num_agents))
        
        return {
            'candidate_names': valid_candidates,
            'candidate_indices': candidate_indices,
            'action_mask': action_mask
        }



    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Bắt đầu một episode mới."""
        super().reset(seed=seed)
        
        # 1. Chọn ngẫu nhiên một trace từ "đáp án"
        selected_trace = random.choice(self.all_traces_data)
        
        # 2. Trích xuất chuỗi chuyên gia (State, Action)
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
        
        # 3. Trả về trạng thái ban đầu
        self.current_state_text = self.current_expert_trace[0][0]
        observation = self._get_obs(self.current_state_text)
        info = {
            "current_state_text": self.current_state_text,
            "expert_action": self.current_expert_trace[0][1],
            "trace_length": len(self.current_expert_trace)
        }
        
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Thực thi một hành động bằng cách TRA CỨU state transition trong KB."""
        
        agent_name_to_call = self.action_to_name[action]
        
        # --- BƯỚC 1: TRA CỨU STATE TRANSITION ---
        transition_key = json.dumps([self.current_state_text, agent_name_to_call], ensure_ascii=False)
        
        if transition_key in self.knowledge_base["state_transitions"]:
            # Tìm thấy trong KB - đây là hành động đã được quan sát
            next_state_text = self.knowledge_base["state_transitions"][transition_key]
            observation = self._get_obs(next_state_text)
            status = "observed"
        else:
            # KHÔNG tìm thấy - đây là hành động "khám phá" chưa từng thấy
            next_state_text = f"State: Unknown outcome from exploratory action '{agent_name_to_call}'."
            observation = self._get_obs(next_state_text)
            status = "exploratory"
        
        # --- BƯỚC 2: TÍNH TOÁN REWARD ĐA THÀNH PHẦN ---
        
        # 2.1. Efficiency Reward (R_eff)
        # Giảm penalty để không khuyến khích terminate sớm
        time_elapsed = 2
        R_eff = -time_elapsed
        
        # 2.2. Conformance Reward (R_conf) - "Imitation Reward"
        R_conf = 0
        if self._current_step < len(self.current_expert_trace):
            expert_action = self.current_expert_trace[self._current_step][1]
            if agent_name_to_call == expert_action:
                R_conf = 20  # Tăng thưởng vì tuân thủ
            else:
                R_conf = -5  # Giảm phạt để khuyến khích explore
        
        # Phạt cho các hành động "mò mẫm" không có trong KB
        if status == "exploratory":
            R_conf -= 15
        
        # 2.3. Terminal Reward (R_term)
        self._current_step += 1
        max_steps = 15
        done = (agent_name_to_call == "SummaryAgent") or (self._current_step >= max_steps)
        
        R_term = 0
        if done:
            # Kiểm tra xem workflow có thành công không
            # Tiêu chí thành công: kết thúc bằng SummaryAgent VÀ đi theo đúng dấu vết chuyên gia
            is_successful = (agent_name_to_call == "SummaryAgent" and 
                             self._current_step == len(self.current_expert_trace))
            
            if is_successful:
                R_term = 300  # Tăng thưởng rất lớn khi hoàn thành xuất sắc
            else:
                R_term = -200  # Tăng phạt nặng nếu thất bại
                # Extra penalty cho premature termination (cheat)
                if self._current_step < 3:
                    R_term -= 100  # Phạt thêm nếu terminate quá sớm
        
        # --- TỔNG HỢP REWARD ---
        # Định nghĩa các trọng số (adjusted for better balance)
        w_eff = 0.1       # Tăng lại từ 0.05 -> 0.1
        w_conf = 2.0      # Tăng từ 1.5 -> 2.0 để prioritize conformance
        
        reward = (w_eff * R_eff) + (w_conf * R_conf) + R_term
        
        # Cập nhật state cho bước tiếp theo
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