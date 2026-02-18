import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveMatchingNetwork(nn.Module):
    def __init__(self, config):
        super(AttentiveMatchingNetwork, self).__init__()
        self.config = config
        
        # --- 1. SHARED BACKBONE (CNN + FiLM + LSTM) ---
        # (Reusing your existing effective feature extraction logic)
        
        # EMG Encoder
        self.emg_encoder = self._build_cnn(
            config['emg_in_ch'], config['emg_base_cnn_filters'], 
            config['emg_cnn_layers'], config['cnn_kernel_size'], 
            config['emg_stride'], config['groupnorm_num_groups']
        )
        self.emg_out_dim = config['emg_base_cnn_filters'] * (2 ** (config['emg_cnn_layers'] - 1))

        # IMU Encoder
        self.imu_out_dim = 0
        if config['use_imu']:
            self.imu_encoder = self._build_cnn(
                config['imu_in_ch'], config['imu_base_cnn_filters'], 
                config['imu_cnn_layers'], config['cnn_kernel_size'], 
                config['imu_stride'], config['groupnorm_num_groups']
            )
            self.imu_out_dim = config['imu_base_cnn_filters'] * (2 ** (config['imu_cnn_layers'] - 1))

        # Demographics & FiLM
        self.demo_encoder = DemographicsEncoder(config['demo_in_dim'], config['demo_emb_dim']) if config['use_demographics'] else None
        
        if config['use_film_x_demo'] and self.demo_encoder:
            self.film_emg = FiLMConditioner(self.emg_out_dim, config['demo_emb_dim'])
            self.film_imu = FiLMConditioner(self.imu_out_dim, config['demo_emb_dim']) if config['use_imu'] else None

        # Temporal Processing (LSTM)
        # Note: We keep this to get a rich time-series embedding
        input_feat_dim = self.emg_out_dim + self.imu_out_dim
        if config['use_lstm']:
            self.lstm = nn.LSTM(
                input_size=input_feat_dim, 
                hidden_size=config['lstm_hidden'], 
                num_layers=config['lstm_layers'], 
                batch_first=True, 
                bidirectional=True, 
                dropout=config['dropout'] if config['lstm_layers'] > 1 else 0
            )
            self.feature_dim = config['lstm_hidden'] * 2
        else:
            self.lstm = None
            self.feature_dim = input_feat_dim

        # --- 2. NEW ATTENTION HEADS ---
        
        # Projection to a common matching dimension
        self.embed_dim = config.get('attention_dim', 128)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )

        # A. Contextualizer (Self-Attention on Support Set)
        # Allows the model to see the user's "whole style" before matching
        # (e.g. "This user generally has low EMG amplitude")
        self.support_contextualizer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=4, 
            dim_feedforward=self.embed_dim*2, 
            batch_first=True
        )

        # B. Comparator (Cross-Attention)
        # Query = Unknown Gesture, Key/Value = Support Set
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        # Temperature scaling for the softmax (helps convergence in Metric Learning)
        self.scale = nn.Parameter(torch.tensor(10.0))

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

    def _get_embeddings(self, x_emg, x_imu=None, demographics=None):
        """Passes data through CNN -> FiLM -> LSTM -> Projector"""
        # 1. Demographics
        d_emb = self.demo_encoder(demographics) if (self.demo_encoder and demographics is not None) else None
        
        # 2. CNN
        e = self.emg_encoder(x_emg)
        i = self.imu_encoder(x_imu) if (self.config['use_imu'] and x_imu is not None) else None

        # 3. FiLM
        if self.config['use_film_x_demo'] and d_emb is not None:
            # Handle broadcasting for support sets vs single queries
            if d_emb.dim() == 2 and e.dim() == 3: 
                # e is (Batch, Channels, Time), d is (Batch, Dim)
                # If Batch sizes match, direct. If d is 1 (user) and e is 10 (support), expand.
                if d_emb.size(0) == 1 and e.size(0) > 1:
                    d_emb_run = d_emb.expand(e.size(0), -1)
                else:
                    d_emb_run = d_emb
            e = self.film_emg(e, d_emb_run)
            if i is not None:
                i = self.film_imu(i, d_emb_run)

        # 4. Fusion & LSTM
        combined = torch.cat([e, i], dim=1) if i is not None else e
        if self.lstm:
            combined = combined.permute(0, 2, 1) # (B, T, C)
            out, (hn, _) = self.lstm(combined)
            # Use last hidden state as sequence embedding
            raw_feat = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            raw_feat = torch.mean(combined, dim=2)

        return self.projector(raw_feat)

    def forward(self, query_emg, support_emg, query_imu=None, support_imu=None, demographics=None):
        """
        query_emg: (Batch, Channels, Time) - The unknown gestures
        support_emg: (Batch, N_Classes, Channels, Time) - The 1-shot examples for this user
        
        Returns: Logits (Batch, N_Classes) indicating similarity to each support sample.
        """
        batch_size = query_emg.size(0)
        n_classes = support_emg.size(1)

        # 1. Encode Query
        # q_emb: (Batch, Embed_Dim)
        q_emb = self._get_embeddings(query_emg, query_imu, demographics)
        
        # 2. Encode Support
        # Flatten batch and classes to process in parallel: (Batch * N_Classes, Channels, Time)
        s_emg_flat = support_emg.view(-1, *support_emg.shape[2:])
        s_imu_flat = support_imu.view(-1, *support_imu.shape[2:]) if support_imu is not None else None
        
        # Handle demographics expansion for the flattened support set
        if demographics is not None:
            # demographics is (Batch, Dim) -> Need (Batch * N_Classes, Dim)
            d_flat = demographics.unsqueeze(1).expand(-1, n_classes, -1).reshape(-1, demographics.shape[-1])
        else:
            d_flat = None

        # s_emb_flat: (Batch * N_Classes, Embed_Dim)
        s_emb_flat = self._get_embeddings(s_emg_flat, s_imu_flat, d_flat)
        
        # Reshape back to (Batch, N_Classes, Embed_Dim)
        s_emb = s_emb_flat.view(batch_size, n_classes, -1)

        # --- 3. ATTENTION MECHANISMS ---

        # A. Contextualize the Support Set (Self-Attention)
        # This helps the model understand the "User's Manifold"
        # s_emb becomes "Context Aware Support"
        s_emb = self.support_contextualizer(s_emb) 

        # B. Cross-Attention (Matching)
        # We want to check similarity between Q and {S_1...S_10}
        
        # Prepare Q for attention: (Batch, 1, Embed_Dim)
        q_unsqueezed = q_emb.unsqueeze(1)
        
        # Multihead Attention
        # Query: q_unsqueezed
        # Key: s_emb (The support set)
        # Value: s_emb
        # attn_output: (Batch, 1, Embed_Dim)
        # attn_weights: (Batch, 1, N_Classes) -> This is essentially our classification!
        attn_out, attn_weights = self.cross_attn(query=q_unsqueezed, key=s_emb, value=s_emb)
        
        # C. Compute Similarity Scores
        # Although attn_weights gives a probability, using a cosine similarity 
        # between the Attended Query and the Keys often yields sharper gradients.
        
        # Option 1: Just use Attention Weights (Softmax is already applied inside MHA)
        # return attn_weights.squeeze(1) 
        
        # Option 2 (Better for Metric Learning): Cosine Similarity
        # We calculate distance between Query and every Support sample
        # q_emb: (B, D) -> (B, 1, D)
        # s_emb: (B, N, D)
        
        # Normalize for cosine similarity
        q_norm = F.normalize(q_emb, p=2, dim=1).unsqueeze(1)
        s_norm = F.normalize(s_emb, p=2, dim=2)
        
        # Dot product
        # (B, 1, D) @ (B, D, N) -> (B, 1, N)
        logits = torch.bmm(q_norm, s_norm.transpose(1, 2)).squeeze(1)
        
        # Scale by learnable temperature (crucial for convergence)
        return logits * self.scale

# --- Helper Classes (Same as before) ---

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
        gamma = self.gamma(c).unsqueeze(-1)
        beta = self.beta(c).unsqueeze(-1)
        return h * (1 + gamma) + beta