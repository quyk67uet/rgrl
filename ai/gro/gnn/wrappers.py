import gymnasium as gym
import torch
import numpy as np
from tensordict import TensorDict
from torch_geometric.data import HeteroData

class VHAS_GNN_Wrapper(gym.Wrapper):
    def __init__(self, env, max_nodes=50, max_edges=100, embedding_dim=768):
        super().__init__(env)
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.embedding_dim = embedding_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- 1. Định nghĩa Observation Space tĩnh cho RSL-RL ---
        
        # Node Features Spaces (Float32)
        node_spaces = {
            f"{name}_x": gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(max_nodes, embedding_dim), 
                dtype=np.float32
            )
            for name in ["agent", "state", "tool"]
        }
        
        # Node Masks 
        node_masks = {
            f"{name}_mask": gym.spaces.Box(
                low=0, high=1, 
                shape=(max_nodes,), 
                dtype=np.uint8
            )
            for name in ["agent", "state", "tool"]
        }

        # Edge Index Spaces (Int64)
        edge_types = [
            "state__triggers__agent",
            "agent__produces__state",
            "agent__calls__tool"
        ]
        edge_spaces = {
            f"{etype}_index": gym.spaces.Box(
                low=0, high=max_nodes, 
                shape=(2, max_edges), 
                dtype=np.int64
            )
            for etype in edge_types
        }
        
        # Edge Masks 
        edge_masks = {
            f"{etype}_mask": gym.spaces.Box(
                low=0, high=1, 
                shape=(max_edges,), 
                dtype=np.uint8
            )
            for etype in edge_types
        }

        # Gộp tất cả vào Dict
        self.observation_space = gym.spaces.Dict({
            **node_spaces,
            **node_masks,
            **edge_spaces,
            **edge_masks,
            # Thêm key mới cho action mask
            "action_mask": gym.spaces.Box(0, 1, (self.env.num_actions,), dtype=np.uint8)
        })

    def _pad_tensor(self, tensor, max_len, is_edge_index=False):
        """Hàm tiện ích để padding tensor"""
        
        # Chuẩn bị mask kiểu Bool để xử lý logic, nhưng sẽ cast về uint8 khi trả về
        mask_dtype = torch.uint8 

        if is_edge_index:
            # Edge index có shape [2, num_edges]
            current_len = tensor.shape[1]
            if current_len > max_len:
                # Truncate nếu vượt quá
                return (
                    tensor[:, :max_len], 
                    torch.ones(max_len, dtype=mask_dtype, device=self.device)
                )
            
            # Padding
            padded = torch.zeros((2, max_len), dtype=torch.long, device=self.device)
            padded[:, :current_len] = tensor
            
            # Tạo mask (1: thật, 0: đệm)
            mask = torch.zeros(max_len, dtype=mask_dtype, device=self.device)
            mask[:current_len] = 1
            return padded, mask
        else:
            # Node features có shape [num_nodes, dim]
            current_len = tensor.shape[0]
            if current_len > max_len:
                return (
                    tensor[:max_len], 
                    torch.ones(max_len, dtype=mask_dtype, device=self.device)
                )

            padded = torch.zeros((max_len, tensor.shape[1]), dtype=tensor.dtype, device=self.device)
            padded[:current_len] = tensor
            
            mask = torch.zeros(max_len, dtype=mask_dtype, device=self.device)
            mask[:current_len] = 1
            return padded, mask

    def _process_hetero_data(self, data: HeteroData, action_mask: np.ndarray):
        """Biến đổi HeteroData động thành TensorDict tĩnh"""
        processed_dict = {}

        # 1. Xử lý Nodes (Agent, State, Tool)
        for node_type in ["agent", "state", "tool"]:
            if node_type in data.x_dict:
                x = data[node_type].x
            else:
                x = torch.zeros((0, self.embedding_dim), device=self.device)
            
            padded_x, mask = self._pad_tensor(x, self.max_nodes)
            processed_dict[f"{node_type}_x"] = padded_x
            processed_dict[f"{node_type}_mask"] = mask

        # 2. Xử lý Edges
        edge_mapping = {
            ("state", "triggers", "agent"): "state__triggers__agent",
            ("agent", "produces", "state"): "agent__produces__state",
            ("agent", "calls", "tool"): "agent__calls__tool"
        }

        for pyg_edge_type, str_edge_type in edge_mapping.items():
            if pyg_edge_type in data.edge_index_dict:
                edge_index = data[pyg_edge_type].edge_index
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
            padded_edge, mask = self._pad_tensor(edge_index, self.max_edges, is_edge_index=True)
            processed_dict[f"{str_edge_type}_index"] = padded_edge
            processed_dict[f"{str_edge_type}_mask"] = mask
				
				# --- GÓI ACTION MASK VÀO TENSORDICT ---
        processed_dict["action_mask"] = torch.from_numpy(action_mask).to(self.device)
        
        # Quan trọng: RSL-RL mong đợi TensorDict trả về từ env
        # batch_size=[] nghĩa là đây là observation của 1 môi trường đơn lẻ
        return TensorDict(processed_dict, batch_size=[])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # obs là HeteroData, info chứa action_mask
        action_mask = info.get('action_mask', np.ones(self.env.num_actions, dtype=np.uint8))
        return self._process_hetero_data(obs, action_mask), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # obs là HeteroData, info chứa action_mask
        action_mask = info.get('action_mask', np.ones(self.env.num_actions, dtype=np.uint8))
        return self._process_hetero_data(obs, action_mask), reward, done, truncated, info