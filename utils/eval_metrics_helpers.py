from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import torch
 
 
def passing_errors(preds, lb, ub):
    """
    Interval-based error:
      - 0 if pred is inside [lb, ub]
      - pred - ub if pred > ub
      - lb - pred if pred < lb
    """
    preds = np.asarray(preds)
    lb = np.asarray(lb)
    ub = np.asarray(ub)
 
    err = np.zeros_like(preds, dtype=float)
 
    above = preds > ub
    below = preds < lb
 
    err[above] = preds[above] - ub[above]
    err[below] = lb[below] - preds[below]
 
    return err
 
 
def safe_pct_improvement(old_val, new_val):
    """
    Positive means new_val is better (smaller) than old_val.
    """
    if old_val == 0:
        return np.nan
    return 100.0 * (old_val - new_val) / old_val
 
 
def get_mode_config(
    mode,
    ylabel=None,
    dssat_col=None,
    lb_col=None,
    ub_col=None,
    pred_denorm_col=None,
    prepare_kind=None,
    positive_only=None,
):
    """
    Returns config defaults for each mode, unless overridden.
    """
    mode = mode.lower()
 
    if mode == "soiln":
        cfg = {
            "ylabel": "NTotal",
            "dssat_col": "SoilN",
            "lb_col": "SMN_LB",
            "ub_col": "SMN_UB",
            "pred_denorm_col": "NTotal",
            "prepare_kind": "prepare_dataset",
            "positive_only": False,
            "use_cumsum": False,
        }
    elif mode == "tuber":
        cfg = {
            "ylabel": "TuberDW",
            "dssat_col": "TuberDW_cumsum",
            "lb_col": "tuber_diff_lb",
            "ub_col": "tuber_diff_ub",
            "pred_denorm_col": "TuberDW_diff",
            "prepare_kind": "prepare_dataset_gt",
            "positive_only": True,
            "use_cumsum": True,
        }
    else:
        raise ValueError("mode must be either 'soiln' or 'tuber'")
 
    # allow overrides
    if ylabel is not None:
        cfg["ylabel"] = ylabel
    if dssat_col is not None:
        cfg["dssat_col"] = dssat_col
    if lb_col is not None:
        cfg["lb_col"] = lb_col
    if ub_col is not None:
        cfg["ub_col"] = ub_col
    if pred_denorm_col is not None:
        cfg["pred_denorm_col"] = pred_denorm_col
    if prepare_kind is not None:
        cfg["prepare_kind"] = prepare_kind
    if positive_only is not None:
        cfg["positive_only"] = positive_only
 
    return cfg
 
 
def prepare_inputs_for_mode(ppsr, scen, prepare_kind):
    """
    Dispatches to the correct dataset prep function.
    """
    scaled = ppsr.normalize(scen)
 
    if prepare_kind == "prepare_dataset":
        x_s, _, ds = ppsr.prepare_dataset(scaled)
    elif prepare_kind == "prepare_dataset_gt":
        x_s, _, ds = ppsr.prepare_dataset_gt(scaled)
    else:
        raise ValueError(
            "prepare_kind must be 'prepare_dataset' or 'prepare_dataset_gt'"
        )
 
    return x_s, ds
 
 
def build_prediction_from_model_output(
    raw_out,
    ppsr,
    pred_denorm_col,
    use_cumsum=False,
):
    """
    Converts raw model output to final prediction scale.
    """
    pred = ppsr.denormalize(raw_out, pred_denorm_col)
 
    if use_cumsum:
        pred = np.cumsum(pred)
 
    return pred
 
 
def evaluate_farm_year_with_passing_error(
    data_df,
    ppsr,
    model_base,
    model_ft,
    farm,
    year,
    global_range,
    mode="soiln",
    xlabel="DayAfterPlant",
    ylabel=None,
    dssat_col=None,
    lb_col=None,
    ub_col=None,
    pred_denorm_col=None,
    prepare_kind=None,
    positive_only=None,
):
    """
    Generic evaluator for either:
      - mode='soiln'
      - mode='tuber'
 
    Returns exact RMSE/NRMSE, passing RMSE/NRMSE, passing rates,
    and FT-vs-Base / FT-vs-DSSAT comparisons.
    """
    cfg = get_mode_config(
        mode=mode,
        ylabel=ylabel,
        dssat_col=dssat_col,
        lb_col=lb_col,
        ub_col=ub_col,
        pred_denorm_col=pred_denorm_col,
        prepare_kind=prepare_kind,
        positive_only=positive_only,
    )
 
    ylabel = cfg["ylabel"]
    dssat_col = cfg["dssat_col"]
    lb_col = cfg["lb_col"]
    ub_col = cfg["ub_col"]
    pred_denorm_col = cfg["pred_denorm_col"]
    prepare_kind = cfg["prepare_kind"]
    positive_only = cfg["positive_only"]
    use_cumsum = cfg["use_cumsum"]
 
    device = next(model_ft.parameters()).device
 
    model_base.eval()
    model_ft.eval()
 
    df_f = data_df[
        (data_df.Year == year) &
        (data_df.Farm == farm)
    ]
 
    if df_f.empty:
        return None
 
    # treatment-level metric lists
    rmse_base_list = []
    rmse_ft_list = []
    rmse_dssat_list = []
 
    passing_rmse_base_list = []
    passing_rmse_ft_list = []
    passing_rmse_dssat_list = []
 
    passing_rate_base_list = []
    passing_rate_ft_list = []
    passing_rate_dssat_list = []
 
    n_treatments = 0
 
    for tr in sorted(df_f.Treatment.unique()):
        df_tr = df_f[df_f.Treatment == tr]
 
        all_true = []
        all_base = []
        all_ft = []
        all_dssat = []
        all_lb = []
        all_ub = []
 
        for pd_ in sorted(df_tr.PlantingDay.unique()):
            scen = (
                df_tr[df_tr.PlantingDay == pd_]
                .sort_values(xlabel)
                .reset_index(drop=True)
            )
 
            x_s, ds = prepare_inputs_for_mode(ppsr, scen, prepare_kind)
 
            seq_len = int(ds.tensors[2][0].item())
            xlens = ds.tensors[2].to(device)
 
            with torch.no_grad():
                out_base = model_base(x_s.to(device), xlens)
                out_ft = model_ft(x_s.to(device), xlens)
 
            out_base = out_base.cpu().numpy().squeeze()[:seq_len]
            out_ft = out_ft.cpu().numpy().squeeze()[:seq_len]
 
            pred_base = build_prediction_from_model_output(
                raw_out=out_base,
                ppsr=ppsr,
                pred_denorm_col=pred_denorm_col,
                use_cumsum=use_cumsum,
            )
            pred_ft = build_prediction_from_model_output(
                raw_out=out_ft,
                ppsr=ppsr,
                pred_denorm_col=pred_denorm_col,
                use_cumsum=use_cumsum,
            )
 
            true_vals = scen[ylabel].values[:seq_len]
            dssat_vals = scen[dssat_col].values[:seq_len]
            lb_vals = scen[lb_col].values[:seq_len]
            ub_vals = scen[ub_col].values[:seq_len]
 
            valid = (
                ~np.isnan(true_vals) &
                ~np.isnan(dssat_vals) &
                ~np.isnan(lb_vals) &
                ~np.isnan(ub_vals)
            )
 
            if positive_only:
                valid = valid & (true_vals > 0)
 
            if valid.sum() < 2:
                continue
 
            all_true.append(true_vals[valid])
            all_base.append(pred_base[valid])
            all_ft.append(pred_ft[valid])
            all_dssat.append(dssat_vals[valid])
            all_lb.append(lb_vals[valid])
            all_ub.append(ub_vals[valid])
 
        if len(all_true) == 0:
            continue
 
        all_true = np.concatenate(all_true)
        all_base = np.concatenate(all_base)
        all_ft = np.concatenate(all_ft)
        all_dssat = np.concatenate(all_dssat)
        all_lb = np.concatenate(all_lb)
        all_ub = np.concatenate(all_ub)
 
        # exact RMSE vs ground truth
        rmse_base = np.sqrt(mean_squared_error(all_true, all_base))
        rmse_ft = np.sqrt(mean_squared_error(all_true, all_ft))
        rmse_dssat = np.sqrt(mean_squared_error(all_true, all_dssat))
 
        rmse_base_list.append(rmse_base)
        rmse_ft_list.append(rmse_ft)
        rmse_dssat_list.append(rmse_dssat)
 
        # passing RMSE vs bounds
        pass_err_base = passing_errors(all_base, all_lb, all_ub)
        pass_err_ft = passing_errors(all_ft, all_lb, all_ub)
        pass_err_dssat = passing_errors(all_dssat, all_lb, all_ub)
 
        passing_rmse_base = np.sqrt(np.mean(pass_err_base ** 2))
        passing_rmse_ft = np.sqrt(np.mean(pass_err_ft ** 2))
        passing_rmse_dssat = np.sqrt(np.mean(pass_err_dssat ** 2))
 
        passing_rmse_base_list.append(passing_rmse_base)
        passing_rmse_ft_list.append(passing_rmse_ft)
        passing_rmse_dssat_list.append(passing_rmse_dssat)
 
        # passing rate
        passing_rate_base = np.mean(pass_err_base == 0)
        passing_rate_ft = np.mean(pass_err_ft == 0)
        passing_rate_dssat = np.mean(pass_err_dssat == 0)
 
        passing_rate_base_list.append(passing_rate_base)
        passing_rate_ft_list.append(passing_rate_ft)
        passing_rate_dssat_list.append(passing_rate_dssat)
 
        n_treatments += 1
 
    if n_treatments == 0:
        return None
 
    # average across treatments
    avg_rmse_base = np.mean(rmse_base_list)
    avg_rmse_ft = np.mean(rmse_ft_list)
    avg_rmse_dssat = np.mean(rmse_dssat_list)
 
    avg_nrmse_base = avg_rmse_base / global_range
    avg_nrmse_ft = avg_rmse_ft / global_range
    avg_nrmse_dssat = avg_rmse_dssat / global_range
 
    avg_passing_rmse_base = np.mean(passing_rmse_base_list)
    avg_passing_rmse_ft = np.mean(passing_rmse_ft_list)
    avg_passing_rmse_dssat = np.mean(passing_rmse_dssat_list)
 
    avg_passing_nrmse_base = avg_passing_rmse_base / global_range
    avg_passing_nrmse_ft = avg_passing_rmse_ft / global_range
    avg_passing_nrmse_dssat = avg_passing_rmse_dssat / global_range
 
    avg_passing_rate_base = np.mean(passing_rate_base_list)
    avg_passing_rate_ft = np.mean(passing_rate_ft_list)
    avg_passing_rate_dssat = np.mean(passing_rate_dssat_list)
 
    # percent improvements
    ft_vs_base_pct = safe_pct_improvement(avg_rmse_base, avg_rmse_ft)
    ft_vs_dssat_pct = safe_pct_improvement(avg_rmse_dssat, avg_rmse_ft)
 
    ft_vs_base_passing_pct = safe_pct_improvement(
        avg_passing_rmse_base, avg_passing_rmse_ft
    )
    ft_vs_dssat_passing_pct = safe_pct_improvement(
        avg_passing_rmse_dssat, avg_passing_rmse_ft
    )
 
    return {
        "Mode": mode,
        "Farm": farm,
        "Year": year,
 
        # exact metrics
        # "Avg_RMSE_Base": avg_rmse_base,
        # "Avg_RMSE_FT": avg_rmse_ft,
        # "Avg_RMSE_DSSAT": avg_rmse_dssat,
 
        # "Avg_NRMSE_Base": avg_nrmse_base,
        # "Avg_NRMSE_FT": avg_nrmse_ft,
        # "Avg_NRMSE_DSSAT": avg_nrmse_dssat,
 
        # "FT_vs_Base_%": ft_vs_base_pct,
        # "FT_vs_DSSAT_%": ft_vs_dssat_pct,
 
        # passing metrics
        "Avg_Passing_RMSE_Base": avg_passing_rmse_base,
        "Avg_Passing_RMSE_FT": avg_passing_rmse_ft,
        "Avg_Passing_RMSE_DSSAT": avg_passing_rmse_dssat,
 
        "Avg_Passing_NRMSE_Base": avg_passing_nrmse_base,
        "Avg_Passing_NRMSE_FT": avg_passing_nrmse_ft,
        "Avg_Passing_NRMSE_DSSAT": avg_passing_nrmse_dssat,
 
        "FT_vs_Base_Passing_%": ft_vs_base_passing_pct,
        "FT_vs_DSSAT_Passing_%": ft_vs_dssat_passing_pct,
 
        # # pass rate
        # "Passing_Rate_Base": avg_passing_rate_base,
        # "Passing_Rate_FT": avg_passing_rate_ft,
        # "Passing_Rate_DSSAT": avg_passing_rate_dssat,
 
        "N_Treatments": n_treatments,
    }
 
 
def run_eval_for_farm_ids(
    farm_ids,
    data_df,
    ppsr,
    model_base,
    model_ft,
    global_range,
    mode="soiln",
    xlabel="DayAfterPlant",
    ylabel=None,
    dssat_col=None,
    lb_col=None,
    ub_col=None,
    pred_denorm_col=None,
    prepare_kind=None,
    positive_only=None,
):
    """
    Convenience wrapper to evaluate multiple farm-year ids like:
      ['JR2011', 'AS2012', 'AS2014']
    """
    results = []
 
    for fid in farm_ids:
        farm = fid[:2]
        year = int(fid[2:])
 
        res = evaluate_farm_year_with_passing_error(
            data_df=data_df,
            ppsr=ppsr,
            model_base=model_base,
            model_ft=model_ft,
            farm=farm,
            year=year,
            global_range=global_range,
            mode=mode,
            xlabel=xlabel,
            ylabel=ylabel,
            dssat_col=dssat_col,
            lb_col=lb_col,
            ub_col=ub_col,
            pred_denorm_col=pred_denorm_col,
            prepare_kind=prepare_kind,
            positive_only=positive_only,
        )
 
        if res is not None:
            results.append(res)
 
    summary_df = pd.DataFrame(results)
 
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["Farm", "Year"]).reset_index(drop=True)
 
    return summary_df
 
 
# soiln_summary = run_eval_for_farm_ids(
#     farm_ids=["JR2011", "AS2012", "AS2014"],
#     data_df=soiln_data,          # your SoilN dataframe
#     ppsr=ppsr,
#     model_base=model_base,
#     model_ft=fine_tuned_model,
#     global_range=GLOBAL_RANGE,
#     mode="soiln",
# )
 
# tuber_summary = run_eval_for_farm_ids(
#     farm_ids=["JR2011", "AS2012", "AS2014"],
#     data_df=tuber_data,          # your Tuber dataframe
#     ppsr=ppsr,
#     model_base=model_base,
#     model_ft=fine_tuned_model,
#     global_range=GLOBAL_RANGE,
#     mode="tuber",
# )