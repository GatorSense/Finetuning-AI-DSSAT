from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import types
import sys

import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.metrics import r2_score

from models.models import TransformerModel
from models.soiln_finetune import SoilnTransformerWithExtraEncoders
from utils.finetunemodel import TuberTransformerWithExtraEncoders
from utils.model import TuberTransformer

# Keep these imports so pickle can resolve known class paths.
from utils.preprocessing import PreProcessor as _SoilnLegacyPreProcessor  # noqa: F401
from utils.preprocessingtuber import PreProcessor as _TuberPreProcessor  # noqa: F401


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    label: str
    data_file: str
    time_col: str
    target_col: str
    all_scope_col: str
    default_day_cap: int


MODE_SPECS: Dict[str, ModeSpec] = {
    "soiln": ModeSpec(
        mode="soiln",
        label="SoilN",
        data_file="data_soiln.parquet",
        time_col="DayAfterPlant",
        target_col="NTotal",
        all_scope_col="Treatment",
        default_day_cap=159,
    ),
    "tube": ModeSpec(
        mode="tube",
        label="Tube FT",
        data_file="data_tube.parquet",
        time_col="DayAfterPlant",
        target_col="TuberDW",
        all_scope_col="Treatment",
        default_day_cap=109,
    ),
}


_FILTER_KEY_ALIASES = {
    "nfirstapp": "NFirstApp",
    "planting_day": "PlantingDay",
    "irrgdep": "IrrgDep",
    "irrgthresh": "IrrgThresh",
    "farm": "Farm",
    "year": "Year",
    "treatment": "Treatment",
}


@dataclass
class InferenceContext:
    mode: str
    spec: ModeSpec
    root: Path
    device: torch.device
    data: pd.DataFrame
    ppsr: Any
    base_model: torch.nn.Module
    ft_model: torch.nn.Module


def _load_pickle_with_compat(path: Path) -> Any:
    """Load pickles while mapping legacy `utils.preprocessing.PreProcessor` if needed."""
    compat_module_name = "utils.preprocessing"
    previous_module = sys.modules.get(compat_module_name)

    compat_module = types.ModuleType(compat_module_name)
    compat_module.PreProcessor = _SoilnLegacyPreProcessor

    sys.modules[compat_module_name] = compat_module
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        if previous_module is not None:
            sys.modules[compat_module_name] = previous_module
        else:
            sys.modules.pop(compat_module_name, None)


def _normalize_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key not in MODE_SPECS:
        raise ValueError(f"Unsupported mode '{mode}'. Use one of: {list(MODE_SPECS)}")
    return key


def _normalize_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not filters:
        return {}
    out: Dict[str, Any] = {}
    for key, value in filters.items():
        col = _FILTER_KEY_ALIASES.get(str(key).lower(), key)
        out[col] = value
    return out


def _ensure_required_columns(df: pd.DataFrame, required: list[str], mode: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{mode} data missing required columns: {missing}")


def _load_soiln_data(root: Path, spec: ModeSpec) -> pd.DataFrame:
    df = pd.read_parquet(root / spec.data_file)
    _ensure_required_columns(
        df,
        ["NTotal", "GroundTruthN", "Year", "Farm", "Treatment", spec.time_col, "Rain", "NApp"],
        "soiln",
    )

    df.loc[df["NTotal"] == 0, "NTotal"] = np.nan
    df = df[df[spec.time_col] <= spec.default_day_cap].copy()

    if "NTotal" in df.columns:
        df = df.rename(columns={"NTotal": "NTotal_old"})
    df = df.rename(columns={"GroundTruthN": "NTotal"})
    if "NTotal_old" in df.columns:
        df = df.rename(columns={"NTotal_old": "soiln"})

    _ensure_required_columns(df, ["NTotal", "soiln", "Rain", "NApp"], "soiln")
    df = df.astype({"NTotal": "float32", "soiln": "float32"})
    df.loc[df["NTotal"] == 0, "NTotal"] = np.nan
    return df


def _load_tube_data(root: Path, spec: ModeSpec) -> pd.DataFrame:
    df = pd.read_parquet(root / spec.data_file)
    _ensure_required_columns(
        df,
        ["TuberDW", "TuberDW_cumsum", "Year", "Farm", "Treatment", spec.time_col, "Rain", "NApp"],
        "tube",
    )

    df.loc[df["TuberDW"] == 0, "TuberDW"] = np.nan
    df = df[df["Farm"].isin(["AS", "JR", "PP"])].copy()
    df = df[df[spec.time_col] <= spec.default_day_cap].copy()
    return df


def _load_soiln_models(root: Path, device: torch.device) -> tuple[Any, torch.nn.Module, torch.nn.Module]:
    ppsr = _load_pickle_with_compat(root / "save" / "preprocessor.pkl")

    base_model = TransformerModel(
        input_dim=6,
        d_model=128,
        nhead=4,
        num_layers=6,
        dropout=0.2,
        seq_len=160,
    )

    checkpoint = torch.load(root / "save" / "model_snapshot.pt", map_location=device)
    base_state = checkpoint["MODEL_STATE"]
    base_model.load_state_dict(base_state)
    base_model.to(device).eval()

    ft_model = SoilnTransformerWithExtraEncoders(
        pretrained_state_dict=base_state,
        input_dim=6,
        d_model=128,
        nhead=4,
        base_num_layers=6,
        num_new_layers=2,
        dropout=0.2,
        seq_len=160,
    ).to(device)
    ft_model.load_state_dict(torch.load(root / "save" / "soiln_finetuned.pt", map_location=device))
    ft_model.eval()
    return ppsr, base_model, ft_model


def _load_tube_models(root: Path, device: torch.device) -> tuple[Any, torch.nn.Module, torch.nn.Module]:
    with open(root / "save" / "ppsr_tuber.pkl", "rb") as f:
        ppsr = pickle.load(f)

    base_model = TuberTransformer(
        input_dim=7,
        d_model=64,
        nhead=4,
        num_layers=4,
        dropout=0.2,
        seq_len=160,
    )
    base_ckpt = torch.load(root / "save" / "tuber_model.pt", map_location=device)
    base_model.load_state_dict(base_ckpt)
    base_model.to(device).eval()

    ft_model = TuberTransformerWithExtraEncoders(
        pretrained_state_dict=base_ckpt,
        input_dim=7,
        d_model=64,
        nhead=4,
        base_num_layers=4,
        num_new_layers=1,
        dropout=0.2,
        seq_len=160,
    )
    ft_model.load_state_dict(torch.load(root / "save" / "tuber_model_finetuned.pt", map_location=device))
    ft_model.to(device).eval()
    return ppsr, base_model, ft_model


def build_inference_context(
    mode: str,
    root: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> InferenceContext:
    mode_key = _normalize_mode(mode)
    spec = MODE_SPECS[mode_key]

    root_path = Path(root) if root is not None else Path.cwd()
    run_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode_key == "soiln":
        data = _load_soiln_data(root_path, spec)
        ppsr, base_model, ft_model = _load_soiln_models(root_path, run_device)
    else:
        data = _load_tube_data(root_path, spec)
        ppsr, base_model, ft_model = _load_tube_models(root_path, run_device)

    return InferenceContext(
        mode=mode_key,
        spec=spec,
        root=root_path,
        device=run_device,
        data=data,
        ppsr=ppsr,
        base_model=base_model,
        ft_model=ft_model,
    )


def _apply_base_filters(data: pd.DataFrame, year: int, farm: str) -> pd.DataFrame:
    return data[(data["Year"] == year) & (data["Farm"] == farm)].copy()


def _apply_optional_filters(data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    filtered = data
    for col, value in filters.items():
        if col not in filtered.columns:
            raise ValueError(f"Filter column '{col}' does not exist in data.")
        filtered = filtered[filtered[col] == value]
    return filtered


def get_available_treatments(
    ctx: InferenceContext,
    year: int,
    farm: str,
    scenario_filters: Optional[Dict[str, Any]] = None,
) -> list[Any]:
    filters = _normalize_filters(scenario_filters)
    filters.pop("Treatment", None)

    scoped = _apply_base_filters(ctx.data, year=year, farm=farm)
    scoped = _apply_optional_filters(scoped, filters)

    treatments = sorted(scoped[ctx.spec.all_scope_col].dropna().unique().tolist())
    return treatments


def _pick_single_scenario_rows(
    data: pd.DataFrame,
    mode: str,
    filters: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if data.empty:
        raise ValueError("No rows found for selected scenario filters.")

    candidate_keys = ["PlantingDay", "NFirstApp", "IrrgDep", "IrrgThresh"]
    available = [k for k in candidate_keys if k in data.columns]

    explicit_keys = {k for k in filters.keys() if k in available}
    remaining = [k for k in available if k not in explicit_keys]

    scenario_meta: Dict[str, Any] = {}

    if remaining:
        first = data.sort_values(remaining).iloc[0]
        for col in remaining:
            scenario_meta[col] = first[col]
            data = data[data[col] == first[col]]

    if data.empty:
        raise ValueError(f"{mode} scenario resolution resulted in empty data.")

    return data.sort_values("DayAfterPlant").reset_index(drop=True), scenario_meta


def _slice_or_default(df: pd.DataFrame, col: str, seq_len: int) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(seq_len, dtype=float)
    return df[col].values[:seq_len].astype(float)


def run_inference_for_scenario(
    ctx: InferenceContext,
    year: int,
    farm: str,
    treatment: Optional[Any] = None,
    scenario_filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    filters = _normalize_filters(scenario_filters)

    scoped = _apply_base_filters(ctx.data, year=year, farm=farm)

    chosen_treatment = treatment if treatment is not None else filters.pop("Treatment", None)
    if chosen_treatment is None:
        available = sorted(scoped[ctx.spec.all_scope_col].dropna().unique().tolist())
        if not available:
            raise ValueError(f"No treatments found for Year={year}, Farm={farm}.")
        chosen_treatment = available[0]

    # Calculate the exact application sequence for SoilN by finding all days N was applied across ANY treatment
    napp_sequence_str = str(chosen_treatment)
    if ctx.mode == "soiln" and "NApp" in scoped.columns:
        app_days = scoped[scoped["NApp"] > 0][ctx.spec.time_col].unique()
        app_days = np.sort(app_days)
        
        # Now find the specific amount applied on these target days for the chosen treatment
        tr_scoped = scoped[scoped[ctx.spec.all_scope_col] == chosen_treatment]
        seq = []
        for d in app_days:
            match = tr_scoped[tr_scoped[ctx.spec.time_col] == d]
            if not match.empty:
                val = match["NApp"].iloc[0]
                seq.append(f"{val:.0f}")
            else:
                seq.append("0")
        if seq:
            napp_sequence_str = "-".join(seq)
        else:
            napp_sequence_str = "0"

    scoped = scoped[scoped[ctx.spec.all_scope_col] == chosen_treatment]
    scoped = _apply_optional_filters(scoped, filters)

    scenario_df, auto_meta = _pick_single_scenario_rows(scoped, mode=ctx.mode, filters=filters)
    if scenario_df.empty:
        raise ValueError("Resolved scenario is empty after filtering.")

    if ctx.mode == "soiln":
        scaled = ctx.ppsr.normalize(scenario_df.copy())
        _, _, ds = ctx.ppsr.prepare_dataset(scaled)
        x, _, xlens = ds[0]

        with torch.no_grad():
            pred_base = (
                ctx.base_model(x.unsqueeze(0).to(ctx.device), xlens=xlens.unsqueeze(0).to(ctx.device))
                .cpu()
                .numpy()
                .squeeze()
            )
            pred_ft = (
                ctx.ft_model(x.unsqueeze(0).to(ctx.device), xlens=xlens.unsqueeze(0).to(ctx.device))
                .cpu()
                .numpy()
                .squeeze()
            )

        seq_len = int(xlens.item())
        pred_base_final = np.asarray(ctx.ppsr.denormalize(pred_base[:seq_len], cols="NTotal")).reshape(-1)
        pred_ft_final = np.asarray(ctx.ppsr.denormalize(pred_ft[:seq_len], cols="NTotal")).reshape(-1)
    else:
        scaled = ctx.ppsr.normalize(scenario_df.copy())
        x_s, _, ds = ctx.ppsr.prepare_dataset_gt(scaled)
        xlens = ds.tensors[2].to(ctx.device)

        with torch.no_grad():
            out_base = ctx.base_model(x_s.to(ctx.device), xlens).cpu().numpy().squeeze()
            out_ft = ctx.ft_model(x_s.to(ctx.device), xlens).cpu().numpy().squeeze()

        seq_len = int(ds.tensors[2][0].item())
        pred_base_diff = np.asarray(ctx.ppsr.denormalize(out_base[:seq_len], cols="TuberDW_diff")).reshape(-1)
        pred_ft_diff = np.asarray(ctx.ppsr.denormalize(out_ft[:seq_len], cols="TuberDW_diff")).reshape(-1)
        pred_base_final = np.cumsum(pred_base_diff)
        pred_ft_final = np.cumsum(pred_ft_diff)

    dap = scenario_df[ctx.spec.time_col].values[:seq_len].astype(float)
    true_vals = scenario_df[ctx.spec.target_col].values[:seq_len].astype(float)
    rain_vals = _slice_or_default(scenario_df, "Rain", seq_len)
    napp_vals = _slice_or_default(scenario_df, "NApp", seq_len)

    if ctx.mode == "soiln":
        dssat_target = scenario_df.get("soiln", pd.Series(np.nan, index=scenario_df.index)).values[:seq_len].astype(float)
        smn_lb = scenario_df.get("SMN_LB", pd.Series(np.nan, index=scenario_df.index)).values[:seq_len].astype(float)
        smn_ub = scenario_df.get("SMN_UB", pd.Series(np.nan, index=scenario_df.index)).values[:seq_len].astype(float)
    else:
        dssat_target = scenario_df.get("TuberDW_cumsum", pd.Series(np.nan, index=scenario_df.index)).values[:seq_len].astype(float)
        smn_lb = scenario_df.get("tuber_diff_lb", pd.Series(np.nan, index=scenario_df.index)).values[:seq_len].astype(float)
        smn_ub = scenario_df.get("tuber_diff_ub", pd.Series(np.nan, index=scenario_df.index)).values[:seq_len].astype(float)
        
        # For tuber, since target is cumulative, bounds of diff must be cumsummed
        if np.any(np.isfinite(smn_lb)):
            smn_lb = np.cumsum(np.nan_to_num(smn_lb))
        if np.any(np.isfinite(smn_ub)):
            smn_ub = np.cumsum(np.nan_to_num(smn_ub))

    # For soiln, offset the DAP by the PlantingDay to show pre-plant negative days
    if ctx.mode == "soiln" and "PlantingDay" in scenario_df.columns:
        # PlantingDay is usually constant per scenario, but we can safely vector-subtract
        dap = dap - scenario_df["PlantingDay"].values[:seq_len]

    valid = np.isfinite(true_vals) & (true_vals > 0)
    r2_base = r2_score(true_vals[valid], pred_base_final[valid]) if valid.sum() > 1 else float("nan")
    r2_ft = r2_score(true_vals[valid], pred_ft_final[valid]) if valid.sum() > 1 else float("nan")

    scenario_meta: Dict[str, Any] = {
        "Year": year,
        "Farm": farm,
        "Treatment": chosen_treatment,
    }
    scenario_meta.update({k: v for k, v in filters.items() if k in scenario_df.columns})
    scenario_meta.update(auto_meta)

    res_dict = {
        "mode": ctx.mode,
        "label": ctx.spec.label,
        "target_col": ctx.spec.target_col,
        "scenario": scenario_meta,
        "time": dap,
        "true": true_vals,
        "dssat_target": dssat_target,
        "pred_base": pred_base_final,
        "pred_finetuned": pred_ft_final,
        "rain": rain_vals,
        "napp": napp_vals,
        "napp_sequence": napp_sequence_str,
        "valid_mask": valid,
        "r2_base": r2_base,
        "r2_finetuned": r2_ft,
        "seq_len": seq_len,
        "day_cap": ctx.spec.default_day_cap,
    }
    if smn_lb is not None:
        res_dict["smn_lb"] = smn_lb
    if smn_ub is not None:
        res_dict["smn_ub"] = smn_ub
        
    return res_dict


def summarize_results(results: list[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for res in results:
        row = {
            "mode": res["mode"],
            "r2_base": res["r2_base"],
            "r2_finetuned": res["r2_finetuned"],
            "seq_len": res["seq_len"],
        }
        row.update(res["scenario"])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["mode", "Year", "Farm", "Treatment", "r2_base", "r2_finetuned", "seq_len"])
def _calc_nmrse(y_t: np.ndarray, y_p: np.ndarray) -> float:
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(mask):
        return np.nan
    y_t_valid = y_t[mask]
    y_p_valid = y_p[mask]
    mean_t = np.mean(y_t_valid)
    if mean_t == 0:
        return np.nan
    rmse = np.sqrt(np.mean((y_t_valid - y_p_valid)**2))
    return (rmse / mean_t) * 100.0


def _build_nmrse_row(res: Dict[str, Any], use_mean_gt: bool) -> dict:
    valid = np.asarray(res["valid_mask"])
    y_base = np.asarray(res["pred_base"])
    y_ft = np.asarray(res.get("pred_finetuned", np.zeros_like(y_base)))
    y_dssat = np.asarray(res.get("dssat_target", np.zeros_like(y_base)))
    
    if use_mean_gt and "smn_lb" in res and "smn_ub" in res:
        lb = np.asarray(res["smn_lb"])
        ub = np.asarray(res["smn_ub"])
        y_true = (lb + ub) / 2.0
    else:
        y_true = np.asarray(res["true"])
        
    # Calculate metrics only on valid ground truth points
    nmrse_base = _calc_nmrse(y_true[valid], y_base[valid])
    nmrse_ft = _calc_nmrse(y_true[valid], y_ft[valid]) if "pred_finetuned" in res else np.nan
    nmrse_dssat = _calc_nmrse(y_true[valid], y_dssat[valid]) if "dssat_target" in res else np.nan
    
    row = {
        "Year": res["scenario"].get("Year"),
        "Farm": res["scenario"].get("Farm"),
        "Treatment": res.get("napp_sequence", str(res["scenario"].get("Treatment"))),
        "Base_NMRSE_%": nmrse_base,
        "FT_NMRSE_%": nmrse_ft,
        "DSSAT_NMRSE_%": nmrse_dssat,
        "N_Points": np.sum(valid),
    }
    return row


def evaluate_nmrse_table(
    mode: str,
    year: Optional[int] = None,
    farm: Optional[str] = None,
    use_mean_gt: bool = False,
    scenario_filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Evaluate NMRSE metrics for Base, Fine-tuned, and DSSAT vs. Ground Truth.
    If year and farm are None, computes the average across all treatments
    for every available ground-truth scenario.
    """
    ctx = build_inference_context(mode=mode)
    
    if year is not None and farm is not None:
        treatments = get_available_treatments(ctx, year=year, farm=farm, scenario_filters=scenario_filters)
        if not treatments:
            raise ValueError(f"No treatments found for mode={mode}, year={year}, farm={farm}.")
            
        rows = []
        for tr in treatments:
            res = run_inference_for_scenario(
                ctx, year=year, farm=farm, treatment=tr, scenario_filters=scenario_filters,
            )
            rows.append(_build_nmrse_row(res, use_mean_gt))
            
        return pd.DataFrame(rows)
        
    else:
        # Evaluate global scenarios (all Farm/Year that have Ground Truth > 0)
        df = ctx.data
        gt_mask = pd.notnull(df[ctx.spec.target_col]) & (df[ctx.spec.target_col] > 0)
        gt_scenarios = df[gt_mask][["Year", "Farm"]].drop_duplicates().values.tolist()
        
        agg_rows = []
        for y, f in gt_scenarios:
            try:
                treatments = get_available_treatments(ctx, year=int(y), farm=f, scenario_filters=scenario_filters)
                scenario_rows = []
                for tr in treatments:
                    try:
                        res = run_inference_for_scenario(
                            ctx, year=int(y), farm=f, treatment=tr, scenario_filters=scenario_filters,
                        )
                        scenario_rows.append(_build_nmrse_row(res, use_mean_gt))
                    except Exception:
                        pass
                
                if scenario_rows:
                    scen_df = pd.DataFrame(scenario_rows)
                    agg_row = {
                        "Year": int(y),
                        "Farm": f,
                        "Base_NMRSE_%": scen_df["Base_NMRSE_%"].mean(),
                        "FT_NMRSE_%": scen_df["FT_NMRSE_%"].mean(),
                        "DSSAT_NMRSE_%": scen_df["DSSAT_NMRSE_%"].mean(),
                        "N_Treatments": len(scen_df),
                    }
                    agg_rows.append(agg_row)
            except Exception:
                pass
                
        return pd.DataFrame(agg_rows)
