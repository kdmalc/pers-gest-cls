import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralSubspaceClassifier(nn.Module):
    def __init__(self, in_channels=88, seq_len=64, d_dim=50):
        super().__init__()
        self.d_dim = d_dim
        
        # Support Encoder: Takes (C * T) and produces a projection matrix (C * d_dim)
        # We use a simple MLP here, but it could be replaced with a CNN.
        self.support_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, in_channels * d_dim)
        )
        
    def forward(self, support, query):
        """
        support: (Batch, C, T)
        query:   (Batch, C, T)
        Returns: L1 distance between projected support and query
        """
        batch_size, C, T = support.shape
        
        # 1. Generate sample-conditioned projection matrix Pi: (Batch, C, d_dim)
        P = self.support_encoder(support).view(batch_size, C, self.d_dim)
        
        # 2. Project support and query into the learned subspace
        # P^T @ x -> (Batch, d_dim, C) @ (Batch, C, T) -> (Batch, d_dim, T)
        P_t = P.transpose(1, 2)
        z_s = torch.bmm(P_t, support)
        z_q = torch.bmm(P_t, query)
        
        # 3. Compute Distance (as per the document's analog to $B)
        z_s_flat = z_s.view(batch_size, -1)
        z_q_flat = z_q.view(batch_size, -1)
        
        # L1 distance
        dist = torch.abs(z_s_flat - z_q_flat).sum(dim=1)
        return dist # Lower is better (more similar)
    

class CovarianceEmbeddingNet(nn.Module):
    def __init__(self, in_channels=88, embed_dim=128):
        super().__init__()
        # Number of unique elements in a C x C covariance matrix
        self.cov_dim = in_channels * (in_channels + 1) // 2
        
        # Encoder: simple MLP on the upper triangle (Fast baseline)
        self.encoder = nn.Sequential(
            nn.Linear(self.cov_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def extract_cov_upper_tri(self, x):
        batch_size, C, T = x.shape
        
        # Center the data over the temporal dimension
        x_mean = x.mean(dim=2, keepdim=True)
        x_centered = x - x_mean
        
        # Compute Covariance: (Batch, C, C)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (T - 1)
        
        # Extract the upper triangular part to avoid redundancy
        triu_indices = torch.triu_indices(C, C)
        cov_upper = cov[:, triu_indices[0], triu_indices[1]] # (Batch, cov_dim)
        return cov_upper

    def forward(self, support, query):
        """
        support, query: (Batch, C, T)
        Returns: Embedding distance
        """
        # 1. Explicitly compute covariance
        cov_s = self.extract_cov_upper_tri(support)
        cov_q = self.extract_cov_upper_tri(query)
        
        # 2. Learn a metric on covariance space
        z_s = self.encoder(cov_s)
        z_q = self.encoder(cov_q)
        
        # 3. Similarity measure (negative distance)
        dist = torch.norm(z_s - z_q, p=2, dim=1) # L2 distance
        return dist
    

class CrossAttentionRelationNet(nn.Module):
    def __init__(self, in_channels=88, seq_len=64, cnn_dims=128):
        super().__init__()
        
        # Independent CNN Encoder to extract temporal feature maps
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, cnn_dims, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # For T=64, pooling twice leaves a sequence length of 16
        self.reduced_seq_len = seq_len // 4 
        
        # Cross-Attention mechanism
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cnn_dims, 
            num_heads=4, 
            batch_first=True
        )
        
        # Relation MLP to compute similarity
        self.relation_mlp = nn.Sequential(
            nn.Linear(cnn_dims * 2 * self.reduced_seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Outputs a scalar similarity score
        )
        
    def forward(self, support, query):
        # 1. CNN feature extraction
        h_s = self.cnn(support) # (Batch, cnn_dims, T')
        h_q = self.cnn(query)   # (Batch, cnn_dims, T')
        
        # MultiheadAttention expects (Batch, Seq_len, Features)
        h_s = h_s.permute(0, 2, 1)
        h_q = h_q.permute(0, 2, 1)
        
        # 2. Cross-Attention: Query attends to Support
        # Q = h_q, K = h_s, V = h_s
        h_fused, _ = self.cross_attn(query=h_q, key=h_s, value=h_s)
        
        # 3. Concatenate query representation and attended representation
        concat_features = torch.cat([h_q, h_fused], dim=2) # (Batch, T', cnn_dims * 2)
        concat_flat = concat_features.view(concat_features.size(0), -1)
        
        # 4. Output similarity score (Higher is more similar)
        similarity = self.relation_mlp(concat_flat)
        return similarity.squeeze(1)