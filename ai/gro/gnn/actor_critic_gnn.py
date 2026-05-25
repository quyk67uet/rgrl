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
    Custom GNN-based Actor-Critic for RSL-RL.
    
    This class implements a graph neural network policy for VHAS Orchestrator,
    using HeteroData observations and action masking.
    """
    # RSL-RL compatibility: marks a non-recurrent policy
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
                Custom GNN-based Actor-Critic for RSL-RL.

                Notes:
                - OnPolicyRunner calls the ctor with:

              ActorCriticGNN(obs, obs_groups, env.num_actions, **policy_cfg)

                    so the first 3 args must be (obs, obs_groups, num_actions).
                - num_actor_obs and num_critic_obs are computed from TensorDict obs.
        """
        super().__init__()

                # Validate and cast num_actions
        if not isinstance(num_actions, int):
            num_actions = int(num_actions)
        
        self.num_actions = num_actions
        self.embedding_dim = 768 
        self.gnn_hidden_dim = 128

        # --- 1. GNN backbone (AFAN) ---
        self.gnn = HeteroConv({
            ('state', 'triggers', 'agent'): GATv2Conv((-1, -1), self.gnn_hidden_dim, add_self_loops=False),
            ('agent', 'produces', 'state'): GATv2Conv((-1, -1), self.gnn_hidden_dim, add_self_loops=False),
            ('agent', 'calls', 'tool'): GATv2Conv((-1, -1), self.gnn_hidden_dim, add_self_loops=False),
        }, aggr='mean')
        
        # Separate LayerNorm per node type
        self.node_norms = nn.ModuleDict({
            'agent': LayerNorm(self.gnn_hidden_dim),
            'state': LayerNorm(self.gnn_hidden_dim),
            'tool': LayerNorm(self.gnn_hidden_dim)
        })

        # --- 2. [A2 fix] Empty-graph embedding ---
        # Learnable vector for the "no information" state
        self.empty_graph_embedding = nn.Parameter(torch.randn(1, self.gnn_hidden_dim * 3))

        # --- 3. Actor and critic heads ---
        # Ensure hidden_dims is iterable
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
        Core step: TensorDict -> PyG Batch with index remapping.
        Keeps graph topology correct after unpadding.
        """
        batch_size = obs.shape[0]
        data_list = []
        device = obs.device

        # Mapping setup
        node_types = ["agent", "state", "tool"]
        # edge_mapping keys must match the wrapper output
        edge_mapping = {
            "state__triggers__agent": ("state", "triggers", "agent"),
            "agent__produces__state": ("agent", "produces", "state"),
            "agent__calls__tool": ("agent", "calls", "tool")
        }

        # Get max size from the input tensor shape
        # Assumes [batch_size, max_nodes, feat_dim]
        max_nodes = obs["agent_x"].shape[1] 

        for i in range(batch_size):
            data = HeteroData()
            
            # --- 1. Unpad nodes and build mapping ---
            # id_mappings: Dict[node_type, Tensor]
            # Store old padded index -> new clean graph index
            id_mappings = {}

            for n_type in node_types:
                # Mask: [max_nodes] (bool or uint8)
                mask = obs[f"{n_type}_mask"][i].bool()
                num_valid_nodes = mask.sum().item()
                data[n_type].num_nodes = num_valid_nodes # Save real node count

                if num_valid_nodes > 0:
                    # Get real node features
                    data[n_type].x = obs[f"{n_type}_x"][i][mask]
                    # Build mapping table
                    # Initialize mapping with -1 (invalid)
                    mapping = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
                    # Get old indices where mask is True
                    old_indices = torch.nonzero(mask, as_tuple=True)[0].to(device)
                    # Assign new indices (0, 1, 2...)
                    mapping[old_indices] = torch.arange(num_valid_nodes, device=device)
                    
                    id_mappings[n_type] = mapping
                else:
                    # Empty mapping avoids later key errors
                    id_mappings[n_type] = torch.full((max_nodes,), -1, dtype=torch.long, device=device)

            # --- 2. Unpad edges and remap ---
            for str_key, pyg_key in edge_mapping.items():
                src_type, _, dst_type = pyg_key
                
                # Skip if either endpoint type is missing
                if data[src_type].num_nodes == 0 or data[dst_type].num_nodes == 0:
                    continue # Skip this edge type
                
                # Edge mask: [max_edges]
                edge_mask = obs[f"{str_key}_mask"][i].bool()
                
                if edge_mask.sum() > 0:
                    # Get old edge_index: [2, num_valid_edges]
                    raw_edge_index = obs[f"{str_key}_index"][i][:, edge_mask].long()
                    
                    src_old = raw_edge_index[0]
                    dst_old = raw_edge_index[1]

                    # Remap using the index tables
                    src_mapping = id_mappings[src_type]
                    dst_mapping = id_mappings[dst_type]

                    src_new = src_mapping[src_old]
                    dst_new = dst_mapping[dst_old]

                    # Keep only valid edges
                    # A valid edge maps both src and dst to >= 0
                    valid_edges_mask = (src_new >= 0) & (dst_new >= 0)

                    if valid_edges_mask.sum() > 0:
                        final_src = src_new[valid_edges_mask]
                        final_dst = dst_new[valid_edges_mask]
                        data[pyg_key].edge_index = torch.stack(
                            [final_src, final_dst], dim=0
                        ).to(device)
            
            data_list.append(data)

        # --- Build PyG batch ---
        # Batch.from_data_list handles offsets automatically
        pyg_batch = Batch.from_data_list(data_list)
        
        return pyg_batch

    def _extract_features(self, obs: TensorDict) -> torch.Tensor:
        """
        Flow: TensorDict -> Remap & Batch -> GNN -> Pooling
        """
        pyg_batch = self._reconstruct_hetero_batch(obs)
        
        # Run GNN
        x_dict = self.gnn(pyg_batch.x_dict, pyg_batch.edge_index_dict)
        
        for node_type, embeddings in x_dict.items():
            x_dict[node_type] = self.node_norms[node_type](embeddings)

        # --- Global pooling (optimized: pre-allocate tensor) ---
        batch_size = obs.shape[0]
        if not x_dict or not any(
            pyg_batch[node_type].num_nodes > 0 for node_type in x_dict
        ):
            return self.empty_graph_embedding.to(obs.device).expand(batch_size, -1)

        # Pre-allocate output tensor
        pooled_embeds = torch.zeros(
            (batch_size, 3 * self.gnn_hidden_dim), 
            device=obs.device, 
            dtype=x_dict[list(x_dict.keys())[0]].dtype
        )
        
        node_types = ["agent", "state", "tool"]
        for idx, node_type in enumerate(node_types):
            if node_type in x_dict and pyg_batch[node_type].num_nodes > 0:
                # Pool this node type
                embeds = x_dict[node_type]
                batch_idx = pyg_batch[node_type].batch
                pooled = global_mean_pool(embeds, batch_idx)
                
                # Handle batches missing this node type
                # Fill the corresponding slice
                unique_indices = torch.unique(batch_idx)
                start_idx = idx * self.gnn_hidden_dim
                end_idx = (idx + 1) * self.gnn_hidden_dim
                pooled_embeds[unique_indices, start_idx:end_idx] = pooled
        
        # Final graph embedding
        final_graph_embedding = pooled_embeds  # Shape: [batch_size, 3 * gnn_hidden_dim]
            
        return final_graph_embedding

    def _update_distribution(self, logits: torch.Tensor, action_masks: torch.Tensor) -> None:
        """Update the masked action distribution."""
        
        # Mask invalid actions by setting logits to -inf
        # Convert mask (uint8) to bool
        valid_mask = action_masks.bool()
        if valid_mask.sum() == 0:
            print("Warning: action_masks contains no valid actions; skipping mask to avoid NaNs.")
        else:
            logits[~valid_mask] = -float('inf')
        
        self.distribution = Categorical(logits=logits)

    # --- RSL-RL interface methods ---
    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        # 1. Extract features
        features = self._extract_features(obs)
        
        # 2. Get action_mask from TensorDict
        action_masks = obs["action_mask"]
        
        # 3. Compute logits from the actor head
        logits = self.actor(features)
        
        # 4. Build the masked distribution and sample
        self._update_distribution(logits, action_masks)
        # Sample discrete action indices: shape [batch]
        action_indices = self.distribution.sample()
        # Convert to one-hot vectors
        actions_one_hot = torch.nn.functional.one_hot(
            action_indices, num_classes=self.num_actions
        ).float()
        return actions_one_hot

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Return log-probabilities for the given actions.

        - During training, `actions` is a one-hot vector [batch, num_actions].
        - Convert to indices before calling Categorical.log_prob.
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

        Return a [batch_size, num_actions] tensor for RSL-RL compatibility.
        For Categorical, probabilities are a reasonable choice.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        # Shape: [batch_size, num_actions]
        return self.distribution.probs
    
    @property
    def action_std(self) -> torch.Tensor:
        """
        Standard deviation of the action distribution.

        PPO expects a Gaussian `action_std` for KL computation.
        For Categorical, return ones with the same shape as `action_mean`.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act() first.")
        probs = self.distribution.probs
        # Shape: [batch_size, num_actions]
        return torch.ones_like(probs, device=probs.device, requires_grad=False).detach()
    
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
        
        RSL-RL calls this during training to update running stats.
        Graph inputs do not use observation normalization, so this is a no-op.
        """
        # No normalization needed for graph inputs
        pass
    
    def reset(self, dones=None):
        """
        Reset policy state (e.g., hidden states for recurrent policies).
        Feedforward GNN policy: no-op.
        """
        pass