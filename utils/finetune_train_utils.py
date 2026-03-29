import math
import numpy as np
import torch


def masked_cumulative_mse_loss(pred_diff, true_cum, xlens, valid_obs_mask):
    """
    pred_diff: (B,T) predicted daily diff in REAL units (denormalized already)
    true_cum : (B,T) true cumulative TuberDW in REAL units (denormalized already; sparse = NaNs)
    xlens    : (B,)
    valid_obs_mask: (B,T) True where we have an observed ground-truth cum value

    Returns scalar loss.
    """
    device = pred_diff.device
    B, T = pred_diff.shape

    pred_cum = torch.cumsum(pred_diff, dim=1)

    time_idx = torch.arange(T, device=device)[None, :]
    seq_mask = time_idx < xlens[:, None]

    mask = seq_mask & valid_obs_mask

    if not mask.any():
        # return differentiable zero
        return torch.zeros((), device=device, requires_grad=True)

    diff = pred_cum[mask] - true_cum[mask]
    return (diff ** 2).mean()


def fine_tune_with_cum_loss(
    model,
    train_loader,
    val_loader,
    ppsr,
    device,
    epochs=500,
    lr=1e-5,
    patience=10,
    ckpt_path="save/best_finetuned_cumloss.pt",
):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_val = float("inf")
    wait = 0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_losses = []

        for batch in train_loader:
            # your ds may include extra tensors (lb/ub), so unpack safely
            x, y, xlens = batch[0], batch[1], batch[2]

            x = x.to(device)
            y = y.to(device)
            xlens = xlens.to(device)

            optimizer.zero_grad()

            # model outputs normalized diff
            pred_norm = model(x, xlens)

            # denormalize predicted daily diff -> real units
            pred_diff = ppsr.denormalize(pred_norm, "TuberDW_diff")

            # denormalize ground-truth cumulative tuber DW -> real units (sparse NaNs allowed)
            true_cum = ppsr.denormalize(y, "TuberDW")

            valid_obs_mask = ~torch.isnan(true_cum)

            loss = masked_cumulative_mse_loss(pred_diff, true_cum, xlens, valid_obs_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # ---------------- VALID ----------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                x, y, xlens = batch[0], batch[1], batch[2]
                x = x.to(device)
                y = y.to(device)
                xlens = xlens.to(device)

                pred_norm = model(x, xlens)
                pred_diff = ppsr.denormalize(pred_norm, "TuberDW_diff")
                true_cum  = ppsr.denormalize(y, "TuberDW")

                valid_obs_mask = ~torch.isnan(true_cum)

                vloss = masked_cumulative_mse_loss(pred_diff, true_cum, xlens, valid_obs_mask)
                val_losses.append(vloss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss   = float(np.mean(val_losses))   if val_losses   else float("nan")
        val_rmse   = math.sqrt(val_loss) if (val_loss == val_loss) else float("nan")

        print(
            f"Epoch {epoch:03d} | "
            f"Train Cum-MSE {train_loss:.6f} | "
            f"Val Cum-MSE {val_loss:.6f} | "
            f"Val Cum-RMSE {val_rmse:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best @ epoch {epoch}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # load best
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\nBest epoch = {best_epoch}, best Val Cum-MSE = {best_val:.6f}")
    return model
