def dollar_B_classify(
    sup_emg:    torch.Tensor,   # (N_sup, C, T)  — already normalized
    sup_labels: torch.Tensor,   # (N_sup,)
    qry_emg:    torch.Tensor,   # (N_qry, C, T)
    n_components: int,
    n_classes:  int,
) -> torch.Tensor:
    """
    Exact $B classifier: per-class PCA, query projected into each class's
    own subspace, L1 distance to class template. argmin wins.
    
    For k-shot > 1: average the per-class projected points as the template
    (equivalent to averaging covariances then taking centroid).
    """
    N_qry = qry_emg.shape[0]
    C, T  = qry_emg.shape[1], qry_emg.shape[2]
    n_pc  = min(n_components, C - 1)
    
    # Distance from each query to each class: (N_qry, n_classes)
    all_dists = torch.zeros(N_qry, n_classes, device=qry_emg.device)

    for c in range(n_classes):
        mask    = (sup_labels == c)
        sup_c   = sup_emg[mask]          # (k_shot, C, T)
        
        # --- Fit per-class PCA (exactly as $B Step 3) ---
        # Covariance: average over k_shot trials
        mean_c = sup_c.mean(dim=(0, 2))  # (C,)
        cov    = torch.zeros(C, C, device=sup_emg.device)
        for trial in sup_c:
            x = trial - mean_c.unsqueeze(-1)   # (C, T)
            cov += x @ x.t() / (T - 1)
        cov /= sup_c.shape[0]
        
        _, vecs = torch.linalg.eigh(cov)       # ascending eigenvalues
        U = vecs[:, -n_pc:].flip(dims=[1])     # (C, n_pc) top components
        
        # --- Project support into this class's space, form template ---
        # For each support trial: (C,T) -> project -> (n_pc, T) -> flatten (n_pc*T,)
        sup_projected = []
        for trial in sup_c:
            x = trial - mean_c.unsqueeze(-1)   # (C, T)
            proj = U.t() @ x                   # (n_pc, T)
            sup_projected.append(proj.flatten())
        template = torch.stack(sup_projected).mean(dim=0)  # (n_pc*T,)
        
        # --- Project each query into this class's space ---
        for q_idx in range(N_qry):
            x    = qry_emg[q_idx] - mean_c.unsqueeze(-1)  # (C, T)
            proj = U.t() @ x                               # (n_pc, T)
            qry_vec = proj.flatten()                       # (n_pc*T,)
            all_dists[q_idx, c] = (qry_vec - template).abs().sum()  # L1

    return all_dists.argmin(dim=1)   # (N_qry,)