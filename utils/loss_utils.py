import torch

def masked_mse_stats(preds, targets, xlens):
    device = preds.device
    B, T = preds.shape

    time_idx = torch.arange(T, device=device)[None, :]
    seq_mask = time_idx < xlens[:, None]
    valid_obs_mask = ~torch.isnan(targets)

    mask = seq_mask & valid_obs_mask

    if not mask.any():
        return torch.tensor(0.0, device=device), 0

    diff = preds[mask] - targets[mask]
    loss_sum = (diff ** 2).sum()
    num_valid = diff.numel()

    return loss_sum, num_valid
