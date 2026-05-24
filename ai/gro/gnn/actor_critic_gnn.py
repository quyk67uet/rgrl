import torch
import torch.nn as nn
from torch.nn import LayerNorm
from tensordict import TensorDict
from torch.distributions import Categorical
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import HeteroConv, GATv2Conv, global_mean_pool

from rsl_rl.networks import MLP

class ActorCriticGNN(nn.Module):
    """
    Custom GNN-based Actor-Critic compatible với RSL-RL.
    
    This class implements a graph neural network policy for VHAS Orchestrator,
    using HeteroData observations and action masking.
    """
    # RSL-RL compatibility: Required attribute to indicate non-recurrent policy
    is_recurrent: bool = False
    
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict,
        num_actions: int,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation: str = "elu",
        **kwargs,
    ):
        """
        Custom GNN-based Actor-Critic compatible với RSL-RL.

        Lưu ý quan trọng:
        - OnPolicyRunner sẽ gọi ctor với chữ ký:

              ActorCriticGNN(obs, obs_groups, env.num_actions, **policy_cfg)

          nên 3 tham số đầu tiên PHẢI là (obs, obs_groups, num_actions).
        - Các tham số như num_actor_obs, num_critic_obs sẽ được tính nội bộ
          từ TensorDict obs, không truyền từ bên ngoài.
        """
        super().__init__()

        # Validate và convert num_actions thành int
        if not isinstance(num_actions, int):
            num_actions = int(num_actions)
        
        self.num_actions = num_actions
        self.embedding_dim = 768 
        self.gnn_hidden_dim = 128

        # --- 1. Kiến trúc GNN Backbone (AFAN) ---
        self.gnn = HeteroConv({
            ('state', 'triggers', 'agent'): GATv2Conv((-1, -1), self.gnn_hidden_dim, add_self_loops=False),
            ('agent', 'produces', 'state'): GATv2Conv((-1, -1), self.gnn_hidden_dim, add_self_loops=False),
            ('agent', 'calls', 'tool'): GATv2Conv((-1, -1), self.gnn_hidden_dim, add_self_loops=False),
        }, aggr='mean')
        
        # Tạo một LayerNorm riêng cho mỗi loại nút
        self.node_norms = nn.ModuleDict({
            'agent': LayerNorm(self.gnn_hidden_dim),
            'state': LayerNorm(self.gnn_hidden_dim),
            'tool': LayerNorm(self.gnn_hidden_dim)
        })

        # --- 2. [VÁ LỖI A2] Embedding cho Đồ thị Rỗng ---
        # Một vector có thể học được, đại diện cho trạng thái "không có thông tin"
        self.empty_graph_embedding = nn.Parameter(torch.randn(1, self.gnn_hidden_dim * 3))

        # --- 3. Actor & Critic Heads ---
        # Đảm bảo hidden_dims là list/tuple (MLP yêu cầu iterable)
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256]
        if not isinstance(actor_hidden_dims, (list, tuple)):
            actor_hidden_dims = list(actor_hidden_dims) if hasattr(actor_hidden_dims, '__iter__') else [actor_hidden_dims]
        
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256]
        if not isinstance(critic_hidden_dims, (list, tuple)):
            critic_hidden_dims = list(critic_hidden_dims) if hasattr(critic_hidden_dims, '__iter__') else [critic_hidden_dims]
        
        self.actor = MLP(self.gnn_hidden_dim * 3, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(self.gnn_hidden_dim * 3, 1, critic_hidden_dims, activation)
        
        self.distribution = None
        print(f"ActorCriticGNN initialized (Robustness Patches Applied).")

    def _reconstruct_hetero_batch(self, obs: TensorDict) -> Batch:
        """
        Hàm cốt lõi: Chuyển đổi TensorDict -> PyG Batch với INDEX REMAPPING.
        Đảm bảo topo đồ thị chính xác 100% sau khi unpadding.
        """
        batch_size = obs.shape[0]
        data_list = []
        device = obs.device

        # Cấu hình Mapping
        node_types = ["agent", "state", "tool"]
        # Lưu ý: key của edge_mapping phải khớp với key trong TensorDict từ Wrapper
        edge_mapping = {
            "state__triggers__agent": ("state", "triggers", "agent"),
            "agent__produces__state": ("agent", "produces", "state"),
            "agent__calls__tool": ("agent", "calls", "tool")
        }

        # Lấy kích thước tối đa từ shape của tensor đầu vào (để tạo bảng mapping)
        # Giả sử shape là [batch_size, max_nodes, feat_dim]
        max_nodes = obs["agent_x"].shape[1] 

        for i in range(batch_size):
            data = HeteroData()
            
            # --- 1. UNPADDING NODES & TẠO MAPPING ---
            # id_mappings: Dict[node_type, Tensor]
            # Lưu bảng tra cứu: index_cũ (trong padding) -> index_mới (trong đồ thị sạch)
            id_mappings = {}

            for n_type in node_types:
                # Mask: [max_nodes] (bool hoặc uint8)
                mask = obs[f"{n_type}_mask"][i].bool()
                num_valid_nodes = mask.sum().item()
                data[n_type].num_nodes = num_valid_nodes # Ghi lại số lượng node thật

                if num_valid_nodes > 0:
                    # Lấy feature node thật
                    data[n_type].x = obs[f"{n_type}_x"][i][mask]
                    # TẠO MAPPING TABLE
                    # Khởi tạo bảng mapping với giá trị -1 (invalid)
                    mapping = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
                    # Lấy danh sách index cũ (các vị trí mask == True)
                    old_indices = torch.nonzero(mask, as_tuple=True)[0]
                    # Gán index mới (0, 1, 2...) cho các vị trí đó
                    mapping[old_indices] = torch.arange(num_valid_nodes, device=device)
                    
                    id_mappings[n_type] = mapping
                else:
                    # Nếu không có node nào loại này, tạo mapping rỗng để tránh lỗi key error sau này
                    id_mappings[n_type] = torch.full((max_nodes,), -1, dtype=torch.long, device=device)

            # --- 2. UNPADDING EDGES & REMAPPING ---
            for str_key, pyg_key in edge_mapping.items():
                src_type, _, dst_type = pyg_key
                
                # Check xem node đích có tồn tại không trước khi xử lý cạnh
                if data[src_type].num_nodes == 0 or data[dst_type].num_nodes == 0:
                    continue # Bỏ qua loại cạnh này nếu không có node tương ứng
                
                # Mask cạnh: [max_edges]
                edge_mask = obs[f"{str_key}_mask"][i].bool()
                
                if edge_mask.sum() > 0:
                    # Lấy edge_index cũ: [2, num_valid_edges]
                    raw_edge_index = obs[f"{str_key}_index"][i][:, edge_mask].long()
                    
                    src_old = raw_edge_index[0]
                    dst_old = raw_edge_index[1]

                    # REMAP: Tra cứu index mới từ bảng mapping
                    src_mapping = id_mappings[src_type]
                    dst_mapping = id_mappings[dst_type]

                    src_new = src_mapping[src_old]
                    dst_new = dst_mapping[dst_old]

                    # VALIDATE: Chỉ giữ lại các cạnh nối giữa các node hợp lệ
                    # (Cạnh hợp lệ là cạnh mà cả src và dst đều map ra giá trị >= 0)
                    valid_edges_mask = (src_new >= 0) & (dst_new >= 0)

                    if valid_edges_mask.sum() > 0:
                        final_src = src_new[valid_edges_mask]
                        final_dst = dst_new[valid_edges_mask]
                        data[pyg_key].edge_index = torch.stack([final_src, final_dst], dim=0)
            
            data_list.append(data)

        # --- TẠO PYG BATCH ---
        # Batch.from_data_list sẽ tự động xử lý offset cho các index đã được remap sạch sẽ
        pyg_batch = Batch.from_data_list(data_list)
        
        return pyg_batch

    def _extract_features(self, obs: TensorDict) -> torch.Tensor:
        """
        Flow: TensorDict -> Remap & Batch -> GNN -> Pooling
        """
        pyg_batch = self._reconstruct_hetero_batch(obs)
        
        # Chạy GNN
        x_dict = self.gnn(pyg_batch.x_dict, pyg_batch.edge_index_dict)
        
        for node_type, embeddings in x_dict.items():
            x_dict[node_type] = self.node_norms[node_type](embeddings)

        # --- GLOBAL POOLING (OPTIMIZED: Pre-allocate tensor) ---
        batch_size = obs.shape[0]
        if not x_dict or not any(
            pyg_batch[node_type].num_nodes > 0 for node_type in x_dict
        ):
            return self.empty_graph_embedding.to(obs.device).expand(batch_size, -1)

        # Pre-allocate output tensor for better performance
        pooled_embeds = torch.zeros(
            (batch_size, 3 * self.gnn_hidden_dim), 
            device=obs.device, 
            dtype=x_dict[list(x_dict.keys())[0]].dtype
        )
        
        node_types = ["agent", "state", "tool"]
        for idx, node_type in enumerate(node_types):
            if node_type in x_dict and pyg_batch[node_type].num_nodes > 0:
                # Thực hiện pooling cho loại nút này
                embeds = x_dict[node_type]
                batch_idx = pyg_batch[node_type].batch
                pooled = global_mean_pool(embeds, batch_idx)
                
                # Xử lý trường hợp một số đồ thị trong batch không có loại nút này
                # Tạo một tensor đầy đủ và điền vào
                unique_indices = torch.unique(batch_idx)
                start_idx = idx * self.gnn_hidden_dim
                end_idx = (idx + 1) * self.gnn_hidden_dim
                pooled_embeds[unique_indices, start_idx:end_idx] = pooled
        
        # Final graph embedding (already concatenated)
        final_graph_embedding = pooled_embeds  # Shape: [batch_size, 3 * gnn_hidden_dim]
            
        return final_graph_embedding

    def _update_distribution(self, logits: torch.Tensor, action_masks: torch.Tensor) -> None:
        """Cập nhật phân phối hành động, có áp dụng mask."""
        
        # Áp dụng mask: đặt logits của các hành động không hợp lệ thành -inf
        # Chuyển mask (uint8) thành boolean
        logits[~action_masks.bool()] = -float('inf')
        
        self.distribution = Categorical(logits=logits)

    # --- RSL-RL Interface Methods ---
    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        # 1. Trích xuất đặc trưng
        features = self._extract_features(obs)
        
        # 2. Lấy action_mask từ TensorDict obs
        action_masks = obs["action_mask"]
        
        # 3. Tính Logits từ Actor Head
        logits = self.actor(features)
        
        # 4. Tạo phân phối (đã có mask) và lấy mẫu
        self._update_distribution(logits, action_masks)
        # Sample discrete action indices: shape [batch]
        action_indices = self.distribution.sample()
        # Convert to one-hot vectors so that action shape matches [batch, num_actions]
        actions_one_hot = torch.nn.functional.one_hot(
            action_indices, num_classes=self.num_actions
        ).float()
        return actions_one_hot

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Trả về log-prob của các action đã thực hiện.

        - Trong training, `actions` là one-hot vector [batch, num_actions].
        - Ta chuyển về chỉ số (argmax) trước khi gọi log_prob của Categorical.
        """
        if actions.dim() == 2 and actions.shape[1] == self.num_actions:
            action_indices = actions.argmax(dim=-1)
        else:
            action_indices = actions
        return self.distribution.log_prob(action_indices)
    
    # --- RSL-RL Required Properties ---
    @property
    def action_mean(self) -> torch.Tensor:
        """
        Mean of the action distribution.

        Để tương thích với RSL-RL (PPO), ta trả về tensor shape [batch_size, num_actions].
        Với Categorical, một lựa chọn hợp lý là dùng xác suất (probs) của từng action.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        # Shape: [batch_size, num_actions]
        return self.distribution.probs
    
    @property
    def action_std(self) -> torch.Tensor:
        """
        Standard deviation of the action distribution.

        PPO trong RSL-RL giả định phân phối Gaussian liên tục và dùng `action_std`
        cho tính KL divergence. Với Categorical, ta trả về tensor 1s cùng shape với
        `action_mean` để giữ tương thích shape, dù giá trị không có ý nghĩa vật lý.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        probs = self.distribution.probs
        # Shape: [batch_size, num_actions]
        return torch.ones_like(probs)
    
    @property
    def entropy(self) -> torch.Tensor:
        """
        Entropy of the action distribution.
        Used for exploration bonus in PPO.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        return self.distribution.entropy()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        features = self._extract_features(obs)
        logits = self.actor(features)
        return torch.argmax(logits, dim=-1)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        features = self._extract_features(obs)
        return self.critic(features)
    
    def update_normalization(self, obs: TensorDict) -> None:
        """
        Update normalization statistics for observations.
        
        RSL-RL calls this method during training to update running statistics.
        For GNN policy, we don't use observation normalization (graph-structured input),
        so this is a no-op.
        """
        # No normalization needed for graph-structured observations
        pass
    
    def reset(self, dones=None):
        """
        Reset policy state (e.g., hidden states for recurrent policies).
        For feedforward GNN policy, this is a no-op.
        """
        pass