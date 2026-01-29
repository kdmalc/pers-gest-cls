import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict

class FewShotEpisodeDataset(Dataset):
    def __init__(self, base_dataset, n_way, k_shot, q_query, episodes_per_epoch):
        self.base_dataset = base_dataset  # List of (x, y) or a standard torch Dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch

        # Group indices by class for fast sampling
        self.class_to_indices = defaultdict(list)
        for idx, (_, y) in enumerate(self.base_dataset):
            self.class_to_indices[y].append(idx)
        self.classes = list(self.class_to_indices.keys())
    
    def __len__(self):
        return self.episodes_per_epoch
    
    def __getitem__(self, idx):
        classes = random.sample(self.classes, self.n_way)
        support_x, support_y, query_x, query_y = [], [], [], []
        for i, cls in enumerate(classes):
            idxs = random.sample(self.class_to_indices[cls], self.k_shot + self.q_query)
            support_idx = idxs[:self.k_shot]
            query_idx = idxs[self.k_shot:]
            for si in support_idx:
                support_x.append(self.base_dataset[si][0])
                support_y.append(i)
            for qi in query_idx:
                query_x.append(self.base_dataset[qi][0])
                query_y.append(i)
        # Stack or convert to tensor as appropriate
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)
        return support_x, support_y, query_x, query_y


class RelationNetwork(nn.Module):
    def __init__(self, encoder, relation_module):
        super().__init__()
        self.encoder = encoder
        self.relation_module = relation_module  # Typically a small MLP or CNN

    def forward(self, x):
        return self.encoder(x)


class RelationModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


def relationnet_loss(embeddings, targets, n_classes, n_support, n_query, relation_module):
    # Split support and query
    support = embeddings[:n_classes * n_support]
    support_labels = targets[:n_classes * n_support]
    query = embeddings[n_classes * n_support:]
    query_labels = targets[n_classes * n_support:]

    # For each query, compute relation score with each class prototype (mean of support)
    prototypes = []
    for c in range(n_classes):
        idxs = (support_labels == c).nonzero(as_tuple=True)[0]
        proto = support[idxs].mean(0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes)  # [N, D]

    # Make all pairs: [N_query, N, 2*D]
    n_queries = query.shape[0]
    query_expand = query.unsqueeze(1).expand(n_queries, n_classes, -1)
    prototypes_expand = prototypes.unsqueeze(0).expand(n_queries, n_classes, -1)
    pairs = torch.cat([query_expand, prototypes_expand], dim=2)  # [N_query, N, 2D]

    # Flatten for batch processing
    pairs = pairs.view(-1, pairs.shape[-1])
    scores = relation_module(pairs).view(n_queries, n_classes)  # [N_query, N]
    log_p_y = torch.log(scores + 1e-8)
    loss = F.nll_loss(log_p_y, query_labels)
    preds = log_p_y.argmax(dim=1)
    acc = (preds == query_labels).float().mean().item()
    return loss, acc


def train_relationnet(model, relation_module, train_loader, test_loader, optimizer, device, epochs):
    for epoch in range(1, epochs+1):
        model.train()
        relation_module.train()
        train_loss, train_acc = 0., 0.
        for support_x, support_y, query_x, query_y in train_loader:
            support_x = support_x.squeeze(0).to(device)
            support_y = support_y.squeeze(0).to(device)
            query_x = query_x.squeeze(0).to(device)
            query_y = query_y.squeeze(0).to(device)
            loss, acc = relationnet_loss(support_x, support_y, query_x, query_y, model.encoder, model.relation_module, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Evaluate on test episodes
        model.eval()
        relation_module.eval()
        test_loss, test_acc = 0., 0.
        with torch.no_grad():
            for support_x, support_y, query_x, query_y in test_loader:
                support_x = support_x.squeeze(0).to(device)
                support_y = support_y.squeeze(0).to(device)
                query_x = query_x.squeeze(0).to(device)
                query_y = query_y.squeeze(0).to(device)
                loss, acc = relationnet_loss(support_x, support_y, query_x, query_y, model.encoder, model.relation_module, device)
                test_loss += loss.item()
                test_acc += acc
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        print(f"Epoch {epoch}: Train loss={train_loss:.4f} acc={train_acc:.4f} | Test loss={test_loss:.4f} acc={test_acc:.4f}")


def create_episode(dataset, n_way, k_shot, q_query):
    # dataset: list of (x, y)
    # Returns: support_x, support_y, query_x, query_y

    import random
    from collections import defaultdict

    # Group samples by class
    class_to_samples = defaultdict(list)
    for x, y in dataset:
        class_to_samples[y].append(x)

    # Randomly sample n_way classes
    classes = random.sample(list(class_to_samples.keys()), n_way)

    support_x, support_y, query_x, query_y = [], [], [], []
    for i, cls in enumerate(classes):
        samples = class_to_samples[cls]
        selected = random.sample(samples, k_shot + q_query)
        support_samples = selected[:k_shot]
        query_samples = selected[k_shot:]
        support_x.extend(support_samples)
        support_y.extend([i]*k_shot)
        query_x.extend(query_samples)
        query_y.extend([i]*q_query)
    return support_x, support_y, query_x, query_y


class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pool
        self.fc = nn.Linear(64, out_dim)
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # shape: (batch, 64)
        x = self.fc(x)
        return x  # (batch, out_dim)


class FeatureVectorDataset(Dataset):
    def __init__(self, df):
        self.features = df['feature'].tolist()           # List of lists (length 80)
        self.labels = df['Gesture_Encoded'].tolist()     # Or use 'participant_ids', 'Cluster_ID' if needed

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)  # Shape: (80,)
        x = x.view(80, 1)  # Shape: (80, 1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class FeatureMatrixDataset(Dataset):
    def __init__(self, df):
        self.features = df['feature'].tolist()           # Each is a list of 80 elements
        self.labels = df['Gesture_Encoded'].tolist()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)  # Shape: (80,)
        x = x.view(16, 5)  # Shape: (16, 5)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
