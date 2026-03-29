import numpy as np
import pandas as pd

def clean_soiln_dataframe(df):
    df = df.copy()

    df.loc[df["NTotal"] == 0, "NTotal"] = np.nan
    df = df[df["DayAfterPlant"] <= 159]

    if "NTotal" in df.columns:
        df = df.rename(columns={"NTotal": "soiln"})

    df = df.rename(columns={"GroundTruthN": "NTotal"})
    df = df.astype({"NTotal": "float32", "soiln": "float32"})
    df.loc[df["NTotal"] == 0, "NTotal"] = np.nan

    return df


def split_farms(df, train_farms, val_farms):
    train_df = df[df["FarmId"].isin(train_farms)].copy()
    val_df   = df[df["FarmId"].isin(val_farms)].copy()
    return train_df, val_df
