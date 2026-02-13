import torch
import torch.nn as nn
from MOE_shared import MOELayer

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

        # 2. Demographics & FiLM
        self.demo_encoder = DemographicsEncoder(config['demo_in_dim'], config['demo_emb_dim']) if config['use_demographics'] else None
        
        if config['use_film_x_demo'] and self.demo_encoder:
            self.film_emg = FiLMConditioner(self.emg_out_dim, config['demo_emb_dim'])
            self.film_imu = FiLMConditioner(self.imu_out_dim, config['demo_emb_dim']) if config['use_imu'] else None

        # 3. Temporal Processing (LSTM)
        input_feat_dim = self.emg_out_dim + self.imu_out_dim
        if config['use_lstm']:
            self.lstm = nn.LSTM(input_size=input_feat_dim, hidden_size=config['lstm_hidden'], num_layers=config['lstm_layers'], batch_first=True, bidirectional=True, dropout=config['dropout'] if config['lstm_layers'] > 1 else 0)
            self.feature_dim = config['lstm_hidden'] * 2
        else:
            self.lstm = None
            self.feature_dim = input_feat_dim

        # 4. Context Embedding (User/Set Summary)
        self.context_projector = ContextEncoder(
            feature_dim=self.feature_dim, 
            context_dim=config['context_emb_dim'],
            pool_type=config['context_pool_type']
        )

        # 5. MoE Head
        self.moe = MOELayer(input_dim=self.feature_dim, output_dim=config['num_classes'], num_experts=config['num_experts'], context_dim=config['context_emb_dim'], config=config)

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

    def _process_signals(self, x_emg, x_imu=None, d_emb=None):
        """Internal helper to pass raw signals through CNN -> FiLM -> LSTM -> Pooling"""
        # 1. CNN Encoding
        e = self.emg_encoder(x_emg)
        i = self.imu_encoder(x_imu) if (self.config['use_imu'] and x_imu is not None) else None

        # 2. FiLM Conditioning
        if self.config['use_film_x_demo'] and d_emb is not None:
            # We expand d_emb if we are processing a support set (multiple samples) 
            # while having only one demographic vector for the user.
            if d_emb.size(0) == 1 and e.size(0) > 1:
                d_emb_expanded = d_emb.expand(e.size(0), -1)
            else:
                d_emb_expanded = d_emb
                
            e = self.film_emg(e, d_emb_expanded)
            if i is not None:
                i = self.film_imu(i, d_emb_expanded)

        # 3. Fusion
        combined = torch.cat([e, i], dim=1) if i is not None else e

        # 4. Temporal Processing
        if self.lstm:
            combined = combined.permute(0, 2, 1) # (B, T, C)
            out, (hn, _) = self.lstm(combined)
            # Use GAP or concat hidden states based on config
            if self.config['use_GlobalAvgPooling']:
                return torch.mean(out, dim=1)
            else:
                return torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            # No LSTM: Standard GAP across time dimension
            return torch.mean(combined, dim=2)

    def get_context_vector(self, support_emg, support_imu=None, demographics=None):
        """Used to calculate the static user 'u' vector from a calibration/support set."""
        self.eval() 
        with torch.no_grad():
            d_emb = self.demo_encoder(demographics) if (self.demo_encoder and demographics is not None) else None
            features = self._process_signals(support_emg, support_imu, d_emb) 
            u = self.context_projector(features) 
        self.train()
        return u

    def forward(self, x_emg, x_imu=None, demographics=None, support_emg=None, support_imu=None, context_embedding=None):
        """
        Main forward pass. Handles both 'support' (context-generation) and 'query' samples.
        """
        # 1. Encode demographics once
        d_emb = self.demo_encoder(demographics) if (self.demo_encoder and demographics is not None) else None
        
        # 2. Handle context 'u'
        if context_embedding is not None:
            u = context_embedding
        elif support_emg is not None:
            # Generate u on-the-fly from the support set
            u = self.get_context_vector(support_emg, support_imu, demographics)
        else:
            # Cold start fallback
            u = torch.zeros(x_emg.size(0), self.config['context_emb_dim'], device=x_emg.device)
            
        # Ensure u matches the query batch size for the MoE gate
        if u.size(0) == 1 and x_emg.size(0) > 1:
            u = u.expand(x_emg.size(0), -1)

        # 3. Process Query Signals
        x_features = self._process_signals(x_emg, x_imu, d_emb)

        # 4. Mixture of Experts classification
        return self.moe(x_features, u, d=d_emb)