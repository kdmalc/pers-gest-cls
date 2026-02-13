import torch
import torch.nn as nn
import torch.nn.functional as F

class MOELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, context_dim, config):
        super(MOELayer, self).__init__()
        self.num_experts = num_experts
        self.use_shared_expert = config['use_shared_expert']
        self.expert_architecture = config['expert_architecture']
        self.gate_type = config['gate_type']
        
        # --- 1. Gating Network ---
        # The gate input dimension depends on the type
        if self.gate_type == 'context_only':
            gate_in = context_dim
        elif self.gate_type == 'feature_only':
            gate_in = input_dim
        elif self.gate_type == 'demographic_only':
            gate_in = config['demo_emb_dim']
        elif self.gate_type == 'context_feature':
            gate_in = context_dim + input_dim
        elif self.gate_type == 'context_feature_demo':
            gate_in = context_dim + input_dim + config['demo_emb_dim']
        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")

        self.gate = nn.Linear(gate_in, num_experts)

        # --- 2. Experts ---
        expert_list = []
        for i in range(num_experts):
            if self.expert_architecture=="MLP":
                # 2-Layer MLP Expert
                expert = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, output_dim)
                )
            elif self.expert_architecture=="linear":
                expert = nn.Linear(input_dim, output_dim)
            else:
                raise ValueError("self.expert_architecture must be either MLP or linear!")
            expert_list.append(expert)
        
        self.experts = nn.ModuleList(expert_list)

    def forward(self, x, u, d=None):
        # d is the demographic embedding vector NOT the raw tabular input

        # 1. Prepare Gate Input
        if self.gate_type == 'context_only':
            g_in = u
        elif self.gate_type == 'feature_only':
            g_in = x
        elif self.gate_type == 'demographic_only':
            g_in = d
        elif self.gate_type == 'context_feature':
            g_in = torch.cat([u, x], dim=1)
        elif self.gate_type == 'context_feature_demo':
            g_in = torch.cat([u, x, d], dim=1)

        # 2. Get Routing Weights
        gate_logits = self.gate(g_in)
        weights = F.softmax(gate_logits, dim=1) # (Batch, Num_Experts)

        # 3. Shared Expert Logic
        # If enabled, expert[0] is always active, others are weighted
        if self.use_shared_expert:
            # We treat the first expert as "Shared"
            # TODO: What is this doing? Why don't we just create / select a separate expert? ...
            ## Is this the standard behaviour...
            shared_weight = 0.5 # Or make this learnable
            # Re-scale other weights to fit in the remaining 0.5
            weights = weights * (1 - shared_weight)
            weights[:, 0] = shared_weight

        # 4. Expert Computation
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1) 
        # (Batch, Num_Experts, Output_Dim)

        # 5. Weighted Sum
        weighted_output = torch.sum(weights.unsqueeze(2) * expert_outputs, dim=1)
        return weighted_output, weights