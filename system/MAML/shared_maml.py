import torch
import numpy as np

def meta_evaluate(model, episodic_loader, config, adapt_and_eval_fn):
    """
    A shared eval function.
    adapt_and_eval_fn: This will be either maml_adapt_and_eval or mamlpp_adapt_and_eval
    """
    model.train() # Requirement for RNN gradients
    total_loss = total_correct = total_count = n_eps = 0
    pre_adapt_accs = []

    step_counter = 0
    ep_counter = 0
    for step_item in episodic_loader:  # step_item is now the batch???
        step_counter += 1
        episodes = [step_item] if isinstance(step_item, dict) else step_item
        for ep in episodes:
            ep_counter += 1
            #print(f"step / ep: {step_counter} / {ep_counter}")
            #print(f"Meta eval user_id: {ep['user_id']}")

            # Inside: for ep in episodes:
            #support_labels = ep["support"]["labels"] 
            #emg_data_shape = ep["support"]["emg"] --> Torch.tensor of size [10, 16, 64]
            # Just print the first 5 labels to see if they change between steps for the same user
            #print(f"User: {ep['user_id']} | Support Labels (first 5): {support_labels[:5]}")

            # Compute norm and mean of the first sample in the batch
            # These are all the same... all have norm 32.0 and mean 0.0...
            #first_sample = ep["support"]["emg"][0].detach().float()
            #sample_norm = torch.norm(first_sample).item()
            #sample_mean = first_sample.mean().item()
            #print(f"First Sample Norm: {sample_norm:.4f} | Mean: {sample_mean:.4f}")
            #print(f"Num unique values in sample: {len(torch.unique(ep['support']['emg'][0]))}") --> 1024
            #print(f"First 5 values in first sample: {ep['support']['emg'][0, 0, :5]}")

            metrics = adapt_and_eval_fn(model, config, ep["support"], ep["query"])
            
            # Aggregate
            q_size = len(ep["query"]["labels"]) if isinstance(ep["query"], dict) else len(ep["query"][1])
            total_loss += metrics["loss"]  #.item() --> Already a float!
            total_correct += (metrics["acc"] * q_size)
            total_count += q_size
            n_eps += 1

            if metrics.get("pre_adapt_acc") is not None:
                pre_adapt_accs.append(metrics["pre_adapt_acc"])

        if pre_adapt_accs:
            arr = np.array(pre_adapt_accs)
            print(f"  [Debug] Pre-adapt acc: {arr.mean():.4f} ± {arr.std():.4f}  "
                f"(min={arr.min():.4f}, max={arr.max():.4f}, n={len(arr)})")

    return {
        "loss": total_loss / max(n_eps, 1),
        "acc": total_correct / max(total_count, 1)
    }

def calculate_gradient_alignment(task_gradients):
    """
    task_gradients: List of tuples/lists containing gradients for each task in the meta-batch.
                    e.g., [task1_grads, task2_grads, ...]
    """
    alignments = []
    num_tasks = len(task_gradients)
    
    if num_tasks < 2:
        return 0.0

    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            grad_i_flat = []
            grad_j_flat = []
            
            # Iterate through paired parameters
            for g_i, g_j in zip(task_gradients[i], task_gradients[j]):
                # If a module is entirely unused (e.g. inactive MoE expert or disabled demo encoder), 
                # BOTH tasks will likely have None. We just skip it entirely.
                if g_i is None and g_j is None:
                    continue
                
                # If MOE is active, one task might use an expert and the other might not.
                # In this case, we MUST treat the unused one as zeros to correctly penalize the alignment.
                t_g_i = g_i.flatten() if g_i is not None else torch.zeros_like(g_j).flatten()
                t_g_j = g_j.flatten() if g_j is not None else torch.zeros_like(g_i).flatten()
                
                grad_i_flat.append(t_g_i)
                grad_j_flat.append(t_g_j)
            
            if len(grad_i_flat) == 0:
                continue

            v_i = torch.cat(grad_i_flat)
            v_j = torch.cat(grad_j_flat)
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(v_i.unsqueeze(0), v_j.unsqueeze(0)).item()
            alignments.append(cos_sim)
            
    return sum(alignments) / len(alignments) if alignments else 0.0