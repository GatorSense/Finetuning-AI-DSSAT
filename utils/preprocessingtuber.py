import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import TensorDataset
import time
from pathlib import Path

class PreProcessor:
    def __init__(self, max_seq_len=160):
        """
        Args:
            max_seq_len (int): Maximum length of crop cycle to be considered (default=160)
        """

        self.seq_len = max_seq_len
        
        # Columns to scale (EXCLUDING tuber_diff_lb and tuber_diff_ub).
        # These remain unscaled so we can do the passing-error (pE) math in real units.
        self.scale_cols = [
            'IrrgDep', 'IrrgThresh', 'NApp',
            'Rain', 'AirTempCMax', 'AirTempCMin', 'SolarRad',
            'NTotal', 'GroStage', 'TuberDW_diff', 'TuberDW'
        ]
        #
        
        self.min_vals = None
        self.max_vals = None

    ###########################################################################
    def fit(self, data):
        """
        Compute and fit the min and max values for scaling, using self.scale_cols.
        """
        self.min_vals = data[self.scale_cols].min()
        self.max_vals = data[self.scale_cols].max()

    ###########################################################################
    def validate_scaler_fit(self, _cols=None):
        """
        Validate that the scaler is fitted and that _cols are in self.scale_cols.
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Normalizer not fitted! Call .fit() first.")
        
        if not all(c in self.scale_cols for c in _cols):
            missing = [c for c in _cols if c not in self.scale_cols]
            raise ValueError(f"Some columns not fitted for data: {missing}")

    ###########################################################################
    def normalize(self, data, cols=None):
        """
        Min-Max scale the specified columns. If cols=None, use self.scale_cols.
        
        Returns a new DataFrame with scaled columns (float32).
        """
        if cols is None:
            cols = [col for col in self.scale_cols if col in data.columns]
        self.validate_scaler_fit(cols)

        _min_max_range = self.max_vals[cols] - self.min_vals[cols]
        
        _min_max_range[_min_max_range == 0] = 1.0
        
        _scaled_data = ((data[cols] - self.min_vals[cols]) / _min_max_range).astype(np.float32)
        
        # Rejoin scaled columns back with original DataFrame
        result = data.drop(columns=cols).join(_scaled_data)
        return result[data.columns]

    ###########################################################################
    def denormalize(self, data, cols:str=None):
        """
        Denormalize a single column (cols must be a single string).
        
        data: can be float, np.array, torch.Tensor, etc.
        """
        if cols is None:
            raise ValueError("Please provide one feature name to denormalize.")
        if cols not in self.scale_cols:
            raise ValueError(f"{cols} is not in the fitted scale columns.")
        self.validate_scaler_fit(_cols=[cols])

        _min_max_range = self.max_vals[cols] - self.min_vals[cols]
        _denormalized_data = (data * _min_max_range) + self.min_vals[cols]
        return _denormalized_data

    ###########################################################################
    def train_test_divide(self, data):
        """
        Splits the DataFrame by self.test_years in the "Year" column.
        """
        # 0) Compute TuberDW_diff (Added here to ensure it's present before split)
        data = self.compute_tuber_diff(data)
        
        test_data = data[data["Year"].isin(self.test_years)]
        train_data = data[~data["Year"].isin(self.test_years)]
        return train_data, test_data

    ###########################################################################
    def group_and_pad_scenarios(self, data, target: str = 'TuberDW_diff'):
        """
        Group by [Year, Treatment, ...] and pad sequences to self.seq_len.
        
        Returns:
            pd_features: shape (batch_size, seq_len, #feature_columns)
            pd_targets: shape (batch_size, seq_len, #target_columns)
            pd_seq_lens: shape (batch_size, 1)
            pd_smnLB: shape (batch_size, seq_len)    or None if not found
            pd_smnUB: shape (batch_size, seq_len)    or None if not found
        """
        _data = pl.from_pandas(data)
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        # Grouping columns for a single scenario
        gpl = ["Year", "Treatment", "PlantingDay", "IrrgDep", "IrrgThresh"]
        

        # We consider these always in features
        #features = ['TuberDW_cumsum', 'Rain', 'AirTempC','SolarRad']
        features = ['IrrgDep', 'IrrgThresh', 'AirTempCMax', 'AirTempCMin', 'Rain', 'SolarRad', 'NApp']
        #target = ['TuberDW_diff']

        # We always aggregate these columns so we can pad them:
        #agg_cols = ['TuberDW_cumsum','TuberDW', 'Rain', 'AirTempC','SolarRad']
        agg_cols = ['AirTempCMax', 'AirTempCMin', 'Rain', 'SolarRad', 'NApp', target]

        # If tuber_diff_lb or tuber_diff_ub exist, add them to the aggregator:
        has_smnLB = "tuber_diff_lb" in data.columns
        has_smnUB = "tuber_diff_ub" in data.columns
        if has_smnLB:
            agg_cols.append("tuber_diff_lb")
        if has_smnUB:
            agg_cols.append("tuber_diff_ub")

        # Sort and group
        grouped = (
            _data.sort(by=gpl + ["DayAfterPlant"], maintain_order=True)
            .group_by(gpl, maintain_order=True)
        )

        print("Padding....")
        # Pad aggregator columns to seq_len with 0.0 beyond actual count
        padded_scenarios = grouped.agg(
            pl.col(agg_cols).extend_constant(0.0, self.seq_len - pl.count()),
            pl.count().alias("actual_seq_len")
        )

        # Also replicate the "IrrgDep", "IrrgThresh" as lists of length seq_len
        padded_scenarios = padded_scenarios.with_columns([
            pl.col(col)
            .repeat_by("actual_seq_len")  # list repeated by actual_seq_len
            .list.eval(pl.element().extend_constant(0.0, self.seq_len - pl.element().len()))
            .alias(col)
            for col in ["IrrgDep", "IrrgThresh"]
        ])
        #df_padded = padded_scenarios.to_pandas()
        #return df_padded
        
        # Convert to numpy
        pd_features = padded_scenarios.select(features).to_numpy()   # shape (batch_size, )

        # pd_targets is the numeric array for NTotal
        pd_targets = padded_scenarios.select(target).to_numpy()      # shape (batch_size, )

        # Save the actual sequence lengths
        pd_seq_lens = padded_scenarios.select("actual_seq_len").to_numpy()

        # Now handle tuber_diff_lb / tuber_diff_ub if present
        pd_smnLB = None
        pd_smnUB = None
        if has_smnLB:
            pd_smnLB = padded_scenarios.select("tuber_diff_lb").to_numpy()  # shape (batch_size,)
        if has_smnUB:
            pd_smnUB = padded_scenarios.select("tuber_diff_ub").to_numpy()  # shape (batch_size,)

        return (pd_features, pd_targets, pd_seq_lens, pd_smnLB, pd_smnUB)
        

    ###########################################################################
    def create_tensors(self, x, y, xlens, lb=None, ub=None):
        """
        Convert polars-lists columns to actual torch Tensors with shape:
          x_t   : (batch_size, seq_len, #features)
          y_t   : (batch_size, seq_len)
          xlens_t:(batch_size,)
          lb_t  : (batch_size, seq_len) or None
          ub_t  : (batch_size, seq_len) or None
        """
        print("Creating Tensors....")

        def row_to_2d(features_row):
            """
            features_row is a tuple (or list) of length = #features,
            each item is a list of length seq_len
            e.g. features_row = ([...seq_len...], [...seq_len...], etc.)
            We want shape (seq_len, #features).
            """
            arrays = [np.array(item, dtype=np.float32) for item in features_row]
            return np.column_stack(arrays)  # shape (seq_len, #features)

        # x -> shape (batch_size,) with each entry a tuple of lists. Convert each row to 2D, then stack.
        x_list = [row_to_2d(row) for row in x]
        x_t = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float32)

        # y -> shape (batch_size,) with each entry a single tuple of length 1, containing a list of seq_len
        y_list = [np.array(row[0], dtype=np.float32) for row in y]
        y_t = torch.tensor(np.stack(y_list, axis=0), dtype=torch.float32)

        # xlens -> shape (batch_size,) with each entry = [int_value]
        xlens_list = [int(elem[0]) for elem in xlens]
        xlens_t = torch.tensor(xlens_list, dtype=torch.int32)

        # lb, ub -> shape (batch_size,) with each entry a single list of length seq_len
        lb_t = None
        ub_t = None
        if lb is not None:
            lb_list = [np.array(row[0], dtype=np.float32) for row in lb]
            lb_t = torch.tensor(np.stack(lb_list, axis=0), dtype=torch.float32)
        if ub is not None:
            ub_list = [np.array(row[0], dtype=np.float32) for row in ub]
            ub_t = torch.tensor(np.stack(ub_list, axis=0), dtype=torch.float32)

        return x_t, y_t, xlens_t, lb_t, ub_t

    ###########################################################################
    def prepare_dataset(self, data):
        """
        Build final Tensors & TensorDataset from a DataFrame.

        Returns:
          (x_t, y_t, ds)
        or if tuber_diff_lb / tuber_diff_ub exist:
          (x_t, y_t, ds) but ds will have up to 5 items: (x, y, xlens, lb, ub).
        """
        # group_and_pad to get raw arrays
        pd_features, pd_targets, pd_seq_lens, pd_smnLB, pd_smnUB = self.group_and_pad_scenarios(data, target='TuberDW_diff')

        # create the actual torch Tensors
        x_t, y_t, xlens_t, lb_t, ub_t = self.create_tensors(
            pd_features, pd_targets, pd_seq_lens,
            lb=pd_smnLB,
            ub=pd_smnUB
        )

        # Now we build a TensorDataset that can contain up to 5 Tensors:
        # (x, y, xlens, lb, ub).
        if lb_t is not None and ub_t is not None:
            tensor_dataset = TensorDataset(x_t, y_t, xlens_t, lb_t, ub_t)
        elif lb_t is not None:
            tensor_dataset = TensorDataset(x_t, y_t, xlens_t, lb_t)
        elif ub_t is not None:
            tensor_dataset = TensorDataset(x_t, y_t, xlens_t, ub_t)
        else:
            tensor_dataset = TensorDataset(x_t, y_t, xlens_t)

        return x_t, y_t, tensor_dataset
    
        ###########################################################################
    def prepare_dataset_gt(self, data):
        """
        Build final Tensors & TensorDataset from a DataFrame.

        Returns:
          (x_t, y_t, ds)
        or if tuber_diff_lb / tuber_diff_ub exist:
          (x_t, y_t, ds) but ds will have up to 5 items: (x, y, xlens, lb, ub).
        """
        # group_and_pad to get raw arrays
        pd_features, pd_targets, pd_seq_lens, pd_smnLB, pd_smnUB = self.group_and_pad_scenarios(data, target='TuberDW')

        # create the actual torch Tensors
        x_t, y_t, xlens_t, lb_t, ub_t = self.create_tensors(
            pd_features, pd_targets, pd_seq_lens,
            lb=pd_smnLB,
            ub=pd_smnUB
        )

        # Now we build a TensorDataset that can contain up to 5 Tensors:
        # (x, y, xlens, lb, ub).
        if lb_t is not None and ub_t is not None:
            tensor_dataset = TensorDataset(x_t, y_t, xlens_t, lb_t, ub_t)
        elif lb_t is not None:
            tensor_dataset = TensorDataset(x_t, y_t, xlens_t, lb_t)
        elif ub_t is not None:
            tensor_dataset = TensorDataset(x_t, y_t, xlens_t, ub_t)
        else:
            tensor_dataset = TensorDataset(x_t, y_t, xlens_t)

        return x_t, y_t, tensor_dataset

    ###########################################################################
    def compute_tuber_diff(self, data):
        """
        Computes TuberDW_diff within each group.
        """
        # Ensure data is Polars DataFrame
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)

        group_cols = ["Year", "Treatment", "PlantingDay", "IrrgDep", "IrrgThresh"]

        # 1) Compute TuberDW_diff within each group
        data = data.with_columns([
            pl.col("TuberDW").shift(1).over(group_cols).alias("prev_TuberDW")
        ]).with_columns([
            (pl.col("TuberDW") - pl.col("prev_TuberDW")).fill_null(0).alias("TuberDW_diff")
        ]).drop("prev_TuberDW")
        
        return data.to_pandas()

    ###########################################################################
    def get_train_test_data(self, data):
        """
        1) Splits into train/test
        2) Fits min/max on train
        3) Normalizes train/test
        4) Calls prepare_dataset(...) on each
        """
        # 0) Compute TuberDW_diff
        data = self.compute_tuber_diff(data)

        # 1) Train/Test split
        start_time = time.time()
        train_data, test_data = self.train_test_divide(data)
        print(f"Dividing train/test: {time.time() - start_time:.4f} secs")
        print(f"Train Shape: {train_data.shape} | Test Shape: {test_data.shape}")

        # 2) Fit normalizer on train_data; then normalize
        start_time = time.time()
        self.fit(train_data)

        scaled_train_data = self.normalize(train_data)
        scaled_test_data = self.normalize(test_data)

        print(f"Normalizing data: {time.time() - start_time:.4f} secs")

        # 3) Build Tensors/Datasets for train
        start_time = time.time()
        tr_data, tr_target, tr_dataset = self.prepare_dataset(scaled_train_data)
        print(f"Train Data prepared in {time.time() - start_time:.4f} secs")

        # 4) Build Tensors/Datasets for test
        start_time = time.time()
        ts_data, ts_target, ts_dataset = self.prepare_dataset(scaled_test_data)
        print(f"Test  Data prepared in {time.time() - start_time:.4f} secs")

        return (
            scaled_train_data, scaled_test_data,
            tr_data, tr_target, tr_dataset,
            ts_data, ts_target, ts_dataset
        )
