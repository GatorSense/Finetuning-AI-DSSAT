import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import TensorDataset
import time
from pathlib import Path


class PreProcessor:
    def __init__(self, test_years=None, max_seq_len=160):
        # HyperParameters
        """
        Args:
        test_years (list[int]): Years for testing/validation set
        max_seq_len (int): Maximum length of crop cycle to be considered (default=160)
        """
        
        self.seq_len = max_seq_len
        if test_years is None:
            raise ValueError("test_years cannot be None")
        self.test_years = test_years
        self.min_vals = None
        self.max_vals = None
        self.scale_cols = ['IrrgDep', 'IrrgThresh', 'NApp', 'Rain', 'AirTempC', 'SolarRad', 'NTotal']
        
########################################################################################################
        
    def fit(self, data):
        """
        Compute and fit the min and max values for scaling.
        
        Args:
        data (pd.DataFrame): Input data to fit the min-max normal scaler
        cols (list[str]): List of columns to be fitted
        skipcols (list[str]): List of columns to be ignored (or categorical) -> required if cols=None
        """
        self.min_vals = data[self.scale_cols].min()
        self.max_vals = data[self.scale_cols].max()

########################################################################################################
        
    def validate_scaler_fit(self, _cols=None):
        """
        Validates if the scaler is fitted properly and cols to be normalized/denormalized exists
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError(f"Normalizer not fitted! Please call __class__.fit() function first.")

        if _cols is None:
            return

        missing = [e for e in _cols if e not in self.scale_cols]
        if missing:
            raise ValueError(f"Given cols: {missing} not fitted for data.")

########################################################################################################

    def normalize(self, data, cols=None):
        """
        Args:
        data (pd.DataFrame): Pandas DataFrame with Columns
        cols (lst[str] | None): List of columns to be normalized
        
        Returns:
        pd.DataFrame: Scaled data with same columns as Input Data
        
        Descriptions:
        If cols is None, use __class__.scale_cols for fitted cols list
        """
        if cols is None:
            cols = [col for col in self.scale_cols if col in data.columns]
        self.validate_scaler_fit(cols)
        
        _min_max_range = self.max_vals[cols] - self.min_vals[cols]
        _scaled_data = ((data[cols] - self.min_vals[cols]) / _min_max_range).astype(np.float32)
        result = data.drop(columns=cols).join(_scaled_data)
        return result[data.columns]

########################################################################################################

    def denormalize(self, data, cols:str=None):
        """
        Args:
        data (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Pandas DataFrame with Columns
        cols (str | None): Column Name to be descaled/denormalized
        
        Returns:
        (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Denormalized data for the given feature
        
        Descriptions:
        cols (args) : Use __class__.scale_cols for fitted cols list
        """
        if cols is None or len([cols]) > 1:
            raise ValueError("Please provide one single feature to denormalize at one time.")
        self.validate_scaler_fit(_cols=[cols])
            
        _min_max_range = self.max_vals[cols] - self.min_vals[cols]
        _denormalized_data = (data * _min_max_range) + self.min_vals[cols]
        return _denormalized_data

########################################################################################################

    def train_test_divide(self, data):
        # Divide Total data into Train and test based on "Year"
        test_data = data[data["Year"].isin(self.test_years)]
        train_data = data[~data["Year"].isin(self.test_years)]
        return train_data, test_data

########################################################################################################

    def group_and_pad_scenarios(self, data):
        """
        Args:
        data (pd.DataFrame): Pandas DataFrame with cols
        
        Returns:
        (x, y, xlens)
        
        Description:
        cols (args | return) : ["IrrgDep", "IrrgThresh", "NApp", "Rain", "AirTempC", "SolarRad", "NTotal"]
        x | pd_features (np.ndarray): Input Data - shape (batch_size, num_input_features)
        y | pd_targets (np.ndarray): Target Data - shape (batch_size, 1)
        xlens | pd_seq_lens (np.ndarray): Actual length scalers of the Data Sequence - shape (batch_size, 1)
        """
        
        _data = pl.from_pandas(data)
        gpl = ["Year", "Treatment", "NFirstApp", "PlantingDay", "IrrgDep", "IrrgThresh"]

        features = ["IrrgDep", "IrrgThresh", "NApp", "Rain", "AirTempC", "SolarRad"]
        target = ["NTotal"]
        agg_cols = ["NApp", "Rain", "AirTempC", "SolarRad", "NTotal"]

        grouped = (
            _data.sort(by=gpl + ["DayAfterPlant"], maintain_order=True)
            .group_by(gpl, maintain_order=True)
        )
        
        print("Padding....")
        padded_scenarios = grouped.agg(
            pl.col(agg_cols).extend_constant(0.0, self.seq_len - pl.count()),
            pl.count().alias("actual_seq_len")
        )
        
        padded_scenarios = padded_scenarios.with_columns([
            pl.col(col)
            .repeat_by("actual_seq_len")
            .list.eval(pl.element().extend_constant(0.0, self.seq_len - pl.element().len()))
            .alias(col)
            for col in ["IrrgDep", "IrrgThresh"]
        ])
        
        pd_features = padded_scenarios.select(features).to_numpy()
        pd_targets = padded_scenarios.select(target).to_numpy()
        pd_seq_lens = padded_scenarios.select("actual_seq_len").to_numpy()

        return (pd_features, pd_targets, pd_seq_lens)

########################################################################################################

    def create_tensors(self, x, y, xlens):
        """
        Input:
        x (np.ndarray): Input sequence - shape (batch_size, num_input_features)
        y (np.ndarray): Target sequence - shape (batch_size, 1)
        xlens (np.ndarray): Actual length scalers of the sequence - shape (batch_size, 1)
        
        Returns:
        x_t (torch.Tensor): Input sequence - shape (batch_size, seq_len, num_input_features)
        y_t (torch.Tensor): Target sequence - shape (batch_size, seq_len)
        xlens_t (torch.Tensor): Actual length scalers of the sequence - shape (batch_size,)
        """
        
        print("Creating Tensors....")
        x_t = torch.tensor(np.stack([np.stack(feat, axis=1) for feat in x], axis=0), dtype=torch.float32)
        y_t = torch.tensor(np.stack(np.squeeze(y, axis=1), axis=0), dtype=torch.float32)
        xlens_t = torch.tensor(np.stack(np.squeeze(xlens, axis=1).astype(int), axis=0), dtype=torch.int32)
        
        return (x_t, y_t, xlens_t)        

########################################################################################################

    def prepare_dataset(self, data):
        """
        Args:
        data (pd.DataFrame): Pandas DataFrame with cols
        
        Returns:
        (input_tensors, target_tensors, tensor_dataset)
        
        Description:
        cols (args) :  ["IrrgDep", "IrrgThresh", "NApp", "Rain", "AirTempC", "SolarRad", "NTotal"]
        x_t | input_tensors (torch.Tensor): Input sequence - shape (batch_size, seq_len, num_input_features)
        y_t | target_tensors (torch.Tensor): Target sequence - shape (batch_size, seq_len)
        tensor_dataset (torch.TensorDataset): TensorDataset of tuples, where each tuple is (input_tensors, target_tensors, seq_len_tensors)
        """
        
        pd_features, pd_targets, pd_seq_lens = self.group_and_pad_scenarios(data)
        input_tensors, target_tensors, seq_len_tensors = self.create_tensors(pd_features, pd_targets, pd_seq_lens)
        
        tensor_dataset = TensorDataset(input_tensors, target_tensors, seq_len_tensors)
        return input_tensors, target_tensors, tensor_dataset

########################################################################################################

    def get_train_test_data(self, data):
        """
        This puts together everything right from Loading data to creating train/test data

        """
        # Train/Test divide
        start_time = time.time()
        train_data, test_data = self.train_test_divide(data)
        print(f"Dividing between training and testing....{time.time() - start_time:.4f} secs")
        print(f"Train Shape: {train_data.shape} | Test Shape: {test_data.shape}")
        
        # Normalisation : Using Min-Max Scalaing to normalise the data
        start_time = time.time()
        self.fit(train_data)
        scaled_train_data = self.normalize(train_data)
        scaled_test_data = self.normalize(test_data)
        print(f"Normalizing data....{time.time() - start_time:.4f} secs")
        
        # Sequencing data and converting into Tensors
        start_time = time.time()
        tr_data, tr_target, tr_dataset = self.prepare_dataset(scaled_train_data)
        print(f"Train: Ensuring seq length.... Creating Dataset.... - {time.time() - start_time:.4f} secs")
        start_time = time.time()
        ts_data, ts_target, ts_dataset = self.prepare_dataset(scaled_test_data)
        print(f"Test: Ensuring seq length.... Creating Dataset.... - {time.time() - start_time:.4f} secs")
        
        return (
            scaled_train_data, scaled_test_data,
            tr_data, tr_target, tr_dataset,
            ts_data, ts_target,ts_dataset,
        )