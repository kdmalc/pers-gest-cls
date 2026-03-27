import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleExpertHead(nn.Module):
    """
    Simple MLP classifier used when config['use_MOE'] is False.
    Structurally identical to one MoE expert: Linear -> ReLU -> Linear.
    Accepts (x, u, d) to match MOELayer's forward signature, but ignores u and d
    (since FiLM conditioning on demographics happens upstream in the CNN, and the
    context vector 'u' is only meaningful for routing, which we are bypassing).
    Returns (logits, {}) to stay API-compatible with MOELayer.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x, u=None, d=None):
        return self.mlp(x), {}


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

    def forward(self, x, u, d=None):
        """
        x: Query features (B, input_dim)
        u: Context embedding (B, context_dim)
        d: Embedded demographic vector (B, demo_dim) --> How is this used? ONLY for the routing function? Not actually passed through the MOE layer? 
        """
        # 1. Get routing weights
        g_in = self._prepare_gate_input(x, u, d)
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

class MultimodalCNNLSTMMOE(nn.Module):
    def __init__(self, config):
        super(MultimodalCNNLSTMMOE, self).__init__()
        self.config = config
        
        # 1. Encoders (CNN)
        self.emg_encoder = self._build_cnn(config['emg_in_ch'], config['emg_base_cnn_filters'], config['emg_cnn_layers'], config['cnn_kernel_size'], config['emg_stride'], config['groupnorm_num_groups'])
        self.emg_out_dim = config['emg_base_cnn_filters'] * (2 ** (config['emg_cnn_layers'] - 1))

        self.imu_out_dim = 0
        if config['use_imu']:
            self.imu_encoder = self._build_cnn(config['imu_in_ch'], config['imu_base_cnn_filters'], config['imu_cnn_layers'], config['cnn_kernel_size'], config['imu_stride'], config['groupnorm_num_groups'])
            self.imu_out_dim = config['imu_base_cnn_filters'] * (2 ** (config['imu_cnn_layers'] - 1))

        cnn_combined_dim = self.emg_out_dim + self.imu_out_dim

        # 2. Demographics Encoder
        self.demo_encoder = DemographicsEncoder(config['demo_in_dim'], config['demo_emb_dim']) if config.get('use_demographics', True) else None
        
        # 3. Context Projector (Now projects directly from pooled CNN features, not LSTM)
        self.context_projector = ContextEncoder(
            feature_dim=cnn_combined_dim, 
            context_dim=config['context_emb_dim'],
            pool_type=config['context_pool_type']
        )

        # 4. FiLM Conditioner
        film_cond_dim = config['context_emb_dim'] if config.get('FILM_on_context_or_demo', 'demo') == 'context' else config['demo_emb_dim']
        self.film_emg = FiLMConditioner(self.emg_out_dim, film_cond_dim)
        self.film_imu = FiLMConditioner(self.imu_out_dim, film_cond_dim) if config['use_imu'] else None

        # 5. Temporal Processing (LSTM)
        if config['use_lstm']:
            self.lstm = nn.LSTM(input_size=cnn_combined_dim, hidden_size=config['lstm_hidden'], num_layers=config['lstm_layers'], batch_first=True, bidirectional=True, dropout=config['dropout'] if config['lstm_layers'] > 1 else 0)
            self.feature_dim = config['lstm_hidden'] * 2
        else:
            self.lstm = None
            self.feature_dim = cnn_combined_dim

        # 6. Classification Head
        if config.get('use_MOE', False):
            self.moe = MOELayer(
                input_dim=self.feature_dim,
                output_dim=config['n_way'],
                num_experts=config['num_experts'],
                context_dim=config['context_emb_dim'],
                config=config,
            )
        else:
            self.moe = SingleExpertHead(
                input_dim=self.feature_dim,
                output_dim=config['n_way'],
            )

    def _build_cnn(self, in_channels, base_filters, num_layers, kernel_size, stride, gn_groups):
        layers = []
        curr_in, curr_out = in_channels, base_filters
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(curr_in, curr_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
                nn.GroupNorm(gn_groups, curr_out),
                nn.ReLU(),
                nn.Dropout(self.config['dropout'])
            ])
            curr_in, curr_out = curr_out, curr_out * 2
        return nn.Sequential(*layers)

    def _extract_cnn_features(self, x_emg, x_imu=None):
        """Pass 1: Just get the raw signals translated into feature maps."""
        e = self.emg_encoder(x_emg)
        i = self.imu_encoder(x_imu) if (self.config['use_imu'] and x_imu is not None) else None
        return e, i

    def _apply_film_and_temporal(self, e, i, condition_emb):
        """Pass 2: Apply FiLM conditioning and run through LSTM."""
        # 1. Apply FiLM
        if condition_emb is not None:
            if condition_emb.size(0) == 1 and e.size(0) > 1:
                condition_emb_expanded = condition_emb.expand(e.size(0), -1)
            else:
                condition_emb_expanded = condition_emb
                
            e = self.film_emg(e, condition_emb_expanded)
            if i is not None:
                i = self.film_imu(i, condition_emb_expanded)

        # 2. Fusion
        combined = torch.cat([e, i], dim=1) if i is not None else e

        # 3. Temporal Processing
        if self.lstm:
            combined = combined.permute(0, 2, 1) # (B, T, C)
            out, (hn, _) = self.lstm(combined)
            if self.config['use_GlobalAvgPooling']:
                return torch.mean(out, dim=1)
            else:
                return torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            return torch.mean(combined, dim=2)

    def get_context_vector(self, support_emg, support_imu=None):
        """Derive 'u' from the CNN features of the support set."""
        # Notice we extract CNN features WITHOUT FiLM conditioning here
        e, i = self._extract_cnn_features(support_emg, support_imu)
        
        # Pool the temporal dimension (T) so the context projector just gets a flat vector per sample
        e_pooled = torch.mean(e, dim=2) 
        if i is not None:
            i_pooled = torch.mean(i, dim=2)
            features = torch.cat([e_pooled, i_pooled], dim=1)
        else:
            features = e_pooled
            
        return self.context_projector(features)

    def forward(self, x_emg, x_imu=None, demographics=None, support_emg=None, support_imu=None, context_embedding=None):
        
        # Determine our Conditioning Vector for FiLM based on the config toggle
        condition_emb = None
        film_mode = self.config.get('FILM_on_context_or_demo', 'demo')
        
        if film_mode == 'demo':
            condition_emb = self.demo_encoder(demographics) if (self.demo_encoder and demographics is not None) else None
            # Even if we use demo for FiLM, we still might need the context vector 'u' for the MoE head.
            if self.config.get('use_MOE', False):
                u = self.get_context_vector(support_emg, support_imu) if context_embedding is None else context_embedding
            else:
                u = None # Ignore context completely if not using MOE and not using context for FiLM
                
        elif film_mode == 'context':
            # We use the support set to generate the context vector, which will act as our condition_emb
            if context_embedding is not None:
                condition_emb = context_embedding
            elif support_emg is not None:
                condition_emb = self.get_context_vector(support_emg, support_imu)
            else:
                condition_emb = torch.zeros(x_emg.size(0), self.config['context_emb_dim'], device=x_emg.device)
            u = condition_emb # The context vector is also 'u' for the MoE head (if used)
            
        # 1. Get raw CNN features for the Query Set
        e_query, i_query = self._extract_cnn_features(x_emg, x_imu)
        
        # 2. Condition the features and pass through LSTM
        x_features = self._apply_film_and_temporal(e_query, i_query, condition_emb)

        # 3. Final Head (MOE or Single MLP)
        if self.config.get('use_MOE', False):
            return self.moe(x_features, u, d=(self.demo_encoder(demographics) if self.demo_encoder else None))
        else:
            return self.moe(x_features)
    

class ContextEncoder(nn.Module):
    """
    Takes a batch of support gestures and creates a single 'context' vector.
    Toggles between simple averaging and Set-style Cross-Attention.
    """
    def __init__(self, feature_dim, context_dim, pool_type='mean', num_heads=4):
        super().__init__()
        self.pool_type = pool_type
        
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU()
        )

        if self.pool_type == 'attn':
            self.query = nn.Parameter(torch.randn(1, 1, context_dim))
            self.kv_proj = nn.Linear(feature_dim, context_dim)
            self.mha = nn.MultiheadAttention(embed_dim=context_dim, num_heads=num_heads, batch_first=True)

    def forward(self, support_features):
        if self.pool_type == 'mean':
            summary = torch.mean(support_features, dim=0, keepdim=True)
            return self.projector(summary)
        elif self.pool_type == 'attn':
            x = support_features.unsqueeze(0) # (1, N, Feat)
            kv = self.kv_proj(x)
            attn_out, _ = self.mha(self.query, kv, kv)
            return attn_out.squeeze(1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")


class DemographicsEncoder(nn.Module):
    def __init__(self, in_dim: int, demo_emb_dim: int = 16):
        super().__init__()
        # TODO: Why do we have hard coded maxes here? Sure we have an unspecified hidden size ig...
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, max(16, demo_emb_dim)),
            nn.GELU(),
            nn.Linear(max(16, demo_emb_dim), demo_emb_dim),
        )
    def forward(self, d):
        return self.net(d)


class FiLMConditioner(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)
    def forward(self, h, c):
        # h: (B, C, T), c: (B, cond_dim)
        gamma = self.gamma(c).unsqueeze(-1)
        beta = self.beta(c).unsqueeze(-1)
        return h * (1 + gamma) + beta