import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineHead(nn.Module):
    """
    Angular similarity classifier. Measures the proximity of features 
    to learned class prototypes on a hypersphere.
    """
    def __init__(self, emb_dim, num_classes, init_tau=10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.02)
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))

    def forward(self, h):
        # Normalize both features and weight prototypes to unit length
        h = F.normalize(h, p=2, dim=-1)
        W = F.normalize(self.W, p=2, dim=-1)
        # Result is cosine similarity scaled by temperature tau
        return self.tau * (h @ W.t())

class MOELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, context_dim, config):
        super(MOELayer, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.expert_architecture = config.get('expert_architecture', 'MLP')
        self.gate_type = config.get('gate_type', 'context_feature_demo')
        self.mixture_mode = config.get('mixture_mode', 'logits')
        self.use_shared_expert = config.get('use_shared_expert', False)
        self.return_aux = config['return_aux']
        
        # 1. Determine Gate Input Dimension
        gate_in_dim = self._get_gate_in_dim(input_dim, context_dim, config)
        
        # 2. The Gating Network
        self.gate = nn.Linear(gate_in_dim, num_experts)
        
        # 3. Build Experts
        expert_list = []
        for i in range(num_experts):
            expert_list.append(self._build_expert(input_dim, output_dim))
        self.experts = nn.ModuleList(expert_list)

    def _get_gate_in_dim(self, input_dim, context_dim, config):
        """Calculates dimension based on what we are feeding the gate."""
        mapping = {
            'context_only': context_dim,
            'feature_only': input_dim,
            'demographic_only': config.get('demo_emb_dim', 16),
            'context_feature': context_dim + input_dim,
            'context_feature_demo': context_dim + input_dim + config.get('demo_emb_dim', 16)
        }
        return mapping.get(self.gate_type, context_dim + input_dim)

    def _build_expert(self, input_dim, output_dim):
        """Factory function for different expert architectures."""
        if self.expert_architecture == "MLP":
            return nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, output_dim)
            )
        elif self.expert_architecture == "linear":
            return nn.Linear(input_dim, output_dim)
        elif self.expert_architecture == "cosine":
            # MLP projection followed by Cosine Similarity
            return nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim // 2),
                CosineHead(emb_dim=input_dim // 2, num_classes=output_dim)
            )
        else:
            raise ValueError(f"Unknown expert architecture: {self.expert_architecture}")

    def _prepare_gate_input(self, x, u, d):
        """Concatenates available signals for the gate."""
        if self.gate_type == 'context_only': return u
        if self.gate_type == 'feature_only': return x
        if self.gate_type == 'demographic_only': return d
        if self.gate_type == 'context_feature': return torch.cat([u, x], dim=1)
        # Default: all three
        if d is not None:
            return torch.cat([u, x, d], dim=1)
        return torch.cat([u, x], dim=1)

    def forward(self, x, u, demographics_emb=None):
        """
        x: Query features (B, input_dim)
        u: Context embedding (B, context_dim)
        demographics_emb: Embedded demographic vector (B, demo_dim)
        """
        # 1. Get routing weights
        g_in = self._prepare_gate_input(x, u, demographics_emb)
        gate_logits = self.gate(g_in)
        weights = F.softmax(gate_logits, dim=1)

        # 2. Shared Expert logic
        if self.use_shared_expert:
            # We treat Expert 0 as the 'Global' expert
            shared_weight = 0.4
            weights = weights * (1 - shared_weight)
            weights[:, 0] = weights[:, 0] + shared_weight

        # 3. Get outputs from all experts
        # expert_outputs shape: (Batch, Num_Experts, Output_Dim)
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1)

        # 4. Mix outputs
        if self.mixture_mode == 'logits':
            # Weighted sum of expert outputs (Standard MoE)
            out = torch.sum(weights.unsqueeze(2) * expert_outputs, dim=1)
        elif self.mixture_mode == 'probs':
            # Expert Ensemble (Sum of probabilities)
            expert_probs = F.softmax(expert_outputs, dim=-1)
            out = torch.sum(weights.unsqueeze(2) * expert_probs, dim=1)
        else:
            out = torch.sum(weights.unsqueeze(2) * expert_outputs, dim=1)

        # 5. Logging Auxiliary Data
        aux = {}
        if self.return_aux:
            # Record how often each expert is used
            aux["gate_usage"] = weights.mean(dim=0).detach()
            # Record which expert was 'dominant' for each sample
            aux["routing_decisions"] = weights.argmax(dim=1).detach()

        return out, aux