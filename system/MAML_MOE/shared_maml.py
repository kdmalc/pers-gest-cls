import torch
import torch.nn as nn
import numpy as np

def meta_evaluate(model, episodic_loader, config, adapt_and_eval_fn):
    """
    A shared eval function.
    adapt_and_eval_fn: This will be either maml_adapt_and_eval or mamlpp_adapt_and_eval
    """
    model.train() # Requirement for RNN gradients
    total_loss = total_correct = total_count = n_eps = 0

    step_counter = 0
    ep_counter = 0
    for step_item in episodic_loader:
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
            total_loss += metrics["loss"]
            total_correct += (metrics["acc"] * q_size)
            total_count += q_size
            n_eps += 1

    return {
        "loss": total_loss / max(n_eps, 1),
        "acc": total_correct / max(total_count, 1)
    }