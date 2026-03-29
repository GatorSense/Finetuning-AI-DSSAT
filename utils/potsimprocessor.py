import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import psutil
import warnings
import json
pl.enable_string_cache()

DataFrameType = Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]
PathType = Union[str, Path]

# Global cache variables
_POTSIM_PATH: Optional[Path] = None
_POTSIM_FRAME: Optional[pl.LazyFrame] = None




def _get_lazyframe(
    data: DataFrameType
):
    """Converts input data to a Polars LazyFrame.

    Args:
        data (DataFrameType): Input data to convert.

    Returns:
        pl.LazyFrame: Converted lazy frame.
    """
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    if isinstance(data, pl.DataFrame):
        data = data.lazy()
    return data




def get_schema(data: DataFrameType):
    """Gets the schema of the input data.

    Args:
        data (DataFrameType): Input data to get schema from.

    Returns:
        Dict: Mapping of column names to their data types.
    """
    return _get_lazyframe(data).collect_schema()




def get_metadata(data: DataFrameType, all: bool = False):
    """Retrieves metadata information for columns in the dataset.

    Args:
        data (DataFrameType): Input data to get metadata from.
        all (bool, optional): Whether to return metadata for all columns. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing column metadata (name, type, description).
    """
    # Read metadata json file
    with open("metadata.json", 'r') as f:
        metadata = json.load(f)
        
    data = _get_lazyframe(data)
    schema_cols = _get_cols(data)
    filtered_metadata = []
    for col_name in schema_cols:
        filtered_metadata.append({
            "column_name": col_name,
            "type": metadata[col_name]["type"],
            "description": metadata[col_name]["description"]
        })
    return pd.DataFrame(filtered_metadata)




def _get_cols(data: DataFrameType):
    """Gets list of column names from input data.

    Args:
        data (DataFrameType): Input data to get columns from.

    Returns:
        List[str]: List of column names.
    """
    return get_schema(data).names()




def _scan_data(filepath: PathType) -> pl.LazyFrame:
    """Scans data file into a Polars LazyFrame.

    Args:
        filepath (PathType): Path to data file (.csv or .parquet).

    Returns:
        pl.LazyFrame: Scanned lazy frame.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file extension is not supported.
    """
    
    # Check filepath for valid file
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    readers = {
        '.csv': pl.scan_csv,
        '.parquet': pl.scan_parquet
    }

    reader = readers.get(filepath.suffix)
    if not reader:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}. Use .csv or .parquet")

    return reader(filepath)




def _check_memory_availability(
    data = pl.LazyFrame,
    buffer: float = 1.50
) -> bool:
    """Checks if sufficient memory is available to load the data.

    Args:
        data (pl.LazyFrame): Input LazyFrame to check.
        buffer (float, optional): Memory buffer multiplier. Defaults to 1.50.

    Returns:
        bool: True if sufficient memory is available.

    Raises:
        MemoryError: If insufficient memory is available.
    """
    available_memory = psutil.virtual_memory().available
    data_size = data.select(pl.len()).collect().item()
    if data_size < 10000:
        return True
    estimated_memory = data.head(10000).collect().estimated_size()
    required_memory = estimated_memory * buffer

    if required_memory > available_memory:
        raise MemoryError(
            f"Insufficient memory: Required {required_memory/(1024**2):.2f} MB"
            f"(including {(buffer-1)*100}% buffer), "
            f"Available {available_memory/(1024**2):.2f} MB"
            "Consider using lazy=True or reducing the dataset size."
        )
    return True




def _process_output(
    data: Union[pl.LazyFrame, pl.DataFrame], 
    lazy: bool = False,
    as_pandas: bool = True
) -> DataFrameType:
    """Processes the output data according to specified format.

    Args:
        data (Union[pl.LazyFrame, pl.DataFrame]): Input data to process.
        lazy (bool, optional): Whether to return LazyFrame. Defaults to False.
        to_pandas (bool, optional): Whether to convert to Pandas DataFrame. Defaults to True.

    Returns:
        DataFrameType: Processed data in specified format.
    """
    if lazy:
        return data
    
    if isinstance(data, pl.LazyFrame):
        _check_memory_availability(data)
        data = data.collect()
    return data.to_pandas() if as_pandas else data




def _check_datacache(
    data: Optional[Union[DataFrameType, PathType]] = None
) -> pl.LazyFrame:
    """Validates and retrieves data from cache if needed.

    Args:
        data (Optional[Union[DataFrameType, PathType]], optional): Input data or path. Defaults to None.

    Returns:
        pl.LazyFrame: Validated lazy frame.

    Raises:
        ValueError: If no source data is available.
    """
    # Ensure the data on which filters need to to be applied is present
    if data is None:
        if _POTSIM_FRAME is None and _POTSIM_PATH is None:
            raise ValueError("No source data available. Either provide from_data or run read_data first")
        data = _POTSIM_FRAME if _POTSIM_FRAME is not None else _scan_data(_POTSIM_PATH)
    elif isinstance(data, (str, Path)):
        data = read_data(data, lazy=True, as_pandas=False)
    else:
        data = _get_lazyframe(data)
    return data




def get_current_scenarios(data: DataFrameType) -> dict:
    """Gets unique values for each scenario column in the dataset.

    Args:
        data (DataFrameType): Input data to get scenarios from.

    Returns:
        dict: Mapping of scenario column names to their unique values.
    """
    data = _get_lazyframe(data)
    all_scenario_cols = ["Year", "PlantingDay", "Treatment", "NFirstApp", "IrrgDep", "IrrgThresh"]
    current_scenario_cols = list(set(all_scenarios_cols) & set(_get_cols(data)))
    
    result = {}
    for col in current_scenario_cols:
        result[col] = (
            data.select(col)
            .unique()
            .collect()
            .get_column(col)
            .to_list()
        )
    return result



    
def read_data(
    dataset_path : PathType,
    weather_path: Optional[PathType] = None,
    usecols: Optional[List] = None,
    scenarios: Optional[Dict[str, Any]] = None,
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType:
    """Reads and processes data from files with optional filtering and column selection.

    Args:
        dataset_path (PathType): Path to main dataset file.
        weather_path (Optional[PathType], optional): Path to weather data file. Defaults to None.
        usecols (Optional[List], optional): List of columns to select. Defaults to None.
        scenarios (Optional[Dict[str, Any]], optional): Dictionary of scenario filters. Defaults to None.
        lazy (bool, optional): Whether to return LazyFrame. Defaults to False.
        to_pandas (bool, optional): Whether to convert to Pandas DataFrame. Defaults to False.

    Returns:
        DataFrameType: Processed data in specified format.

    Raises:
        ValueError: If specified columns are not found.
    """
    
    global _POTSIM_PATH, _POTSIM_FRAME

    data = _scan_data(dataset_path)
    _POTSIM_PATH = Path(dataset_path)
    _POTSIM_FRAME = data

    if weather_path is not None:
        weather_data = _scan_data(weather_path)
        if "Date" not in _get_cols(weather_data) or "Date" not in _get_cols(data):
            raise ValueError("Weather dataset must contain 'Date' column for joining")
        data = data.join(weather_data, on="Date", how="inner")
        
    all_scenario_cols = ["Year", "IrrgDep", "IrrgThresh"]
    if usecols is not None:
        missing_cols = set(usecols) - set(_get_cols(data))
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
            
        # Keep order for usecols argument with scenarios of the left most
        new_cols = [col for col in usecols if col not in all_scenario_cols]
        final_cols = all_scenario_cols + new_cols
        data = data.select(final_cols)
        
    # Check if NTotL1 and NTotL2 are available and create NTotal column
    if "NTotL1" in final_cols and "NTotL2" in final_cols:
        data = data.with_columns(
            ((pl.col("NTotL1") + pl.col("NTotL2")) / 2).alias("NTotal")
        ).drop(["NTotL1", "NTotL2"])


    if scenarios:
        data = apply_filter(data, scenarios, lazy=True, as_pandas=False)
        
    return _process_output(data, lazy, as_pandas)




def apply_filter(
    data: DataFrameType,
    filters: Dict[str, Any],
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType:
    """Applies filters to the dataset.

    Args:
        data (DataFrameType): Input data to filter.
        filters (Dict[str, Any]): Dictionary of column-value pairs for filtering.
        lazy (bool, optional): Whether to return LazyFrame. Defaults to False.
        to_pandas (bool, optional): Whether to convert to Pandas DataFrame. Defaults to False.

    Returns:
        DataFrameType: Filtered data in specified format.

    Raises:
        ValueError: If filter columns are not found in data.
    """
    
    data = _get_lazyframe(data)
    
    invalid_cols = set(filters.keys()) - set(_get_cols(data))
    if invalid_cols:
        raise ValueError(f"Filter columns not found in data: {invalid_cols}")
    
    filter_exprs = []
    for col, values in filters.items():
        if not isinstance(values, (list, tuple)):
            values = [values]
        # expr = pl.col(col).is_in(values)
        # filter_expr = expr if filter_expr is None else filter_expr & expr
        filter_exprs.append(pl.col(col).is_in(values))
        
    if filter_exprs:
        data = data.filter(pl.all_horizontal(filter_exprs))
    
    return _process_output(data, lazy, as_pandas)




def get_memory_usage(
    data: Union[pl.DataFrame, pd.DataFrame],
    unit: str = 'b'
) -> Union[int, float]:
    """Calculates memory usage of the dataset.

    Args:
        data (Union[pl.DataFrame, pd.DataFrame]): Input DataFrame.
        unit (str, optional): Memory unit ('b', 'kb', 'mb', 'gb', 'tb'). Defaults to 'b'.

    Returns:
        Union[int, float]: Memory usage in specified unit.

    Raises:
        ValueError: If invalid unit is specified.
        TypeError: If input is not a DataFrame.
    """
    
    if unit not in {'b', 'kb', 'mb', 'gb', 'tb'}:
        raise ValueError("Unit must be one of 'b', 'kb', 'mb', 'gb', 'tb'")
        
    memory_bytes = 0.0
    if isinstance(data, pd.DataFrame):
        memory_bytes = data.memory_usage(deep=True).sum()
    elif isinstance(data, pl.DataFrame):
        memory_bytes = data.estimated_size()
    else:
        raise TypeError("Data must be either a Pandas or Polars DataFrame")
    
    # Convert to requested unit
    units = {
        'b': 1,
        'kb': 1024,
        'mb': 1024**2,
        'gb': 1024**3,
        'tb': 1024**4
    }
    return memory_bytes / units[unit]




def add_scenarios(
    to_data: DataFrameType,
    scenarios: Dict[str,Any],
    from_data: Optional[Union[DataFrameType, PathType]] = None,
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType:
    """Adds new scenarios to existing dataset.

    Args:
        to_data (DataFrameType): Target dataset.
        scenarios (Dict[str, Any]): Dictionary of new scenarios to add.
        from_data (Optional[Union[DataFrameType, PathType]], optional): Source data. Defaults to None.
        lazy (bool, optional): Whether to return LazyFrame. Defaults to False.
        to_pandas (bool, optional): Whether to convert to Pandas DataFrame. Defaults to False.

    Returns:
        DataFrameType: Updated dataset with new scenarios.

    Raises:
        ValueError: If scenario columns are invalid.
    """
    
    # Convert the main dataframe in lazyframe
    to_data = _get_lazyframe(to_data)
    
    # Validate from_data
    from_data = _check_datacache(from_data)
    
    all_scenario_cols = ["Year", "PlantingDay", "Treatment", "NFirstApp", "IrrgDep", "IrrgThresh"]
    invalid_scenarios = set(scenarios.keys()) - set(all_scenario_cols)
    if invalid_scenarios:
        raise ValueError(f"Scenario columns missing in to_data: {invalid_scenarios}")
    
    # Get current unique scenarios and Update  with new values
    current_scenarios = get_current_scenarios(to_data)
    for col, values in scenarios.items():
        if not isinstance(values, (list, tuple)):
            values = [values]
        if col in current_scenarios:
            current_scenarios[col].extend(values)
            current_scenarios[col] = list(set(current_scenarios[col]))
        else:
            current_scenarios[col] = values 
    
    # Get current columns and any new scenario column if needed
    available_cols = list(set(scenarios.keys()) | set(_get_cols(to_data)))
    from_data = from_data.select(available_cols)
    result = apply_filter(from_data, current_scenarios, lazy=True, as_pandas=False)
    return _process_output(result, lazy, as_pandas)




def add_features(
    to_data: DataFrameType,
    features: List[str],
    from_data: Optional[Union[DataFrameType, PathType]] = None,
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType:
    """Adds new features to existing dataset.

    Args:
        to_data (DataFrameType): Target dataset.
        features (List[str]): List of feature names to add.
        from_data (Optional[Union[DataFrameType, PathType]], optional): Source data. Defaults to None.
        lazy (bool, optional): Whether to return LazyFrame. Defaults to False.
        to_pandas (bool, optional): Whether to convert to Pandas DataFrame. Defaults to False.

    Returns:
        DataFrameType: Updated dataset with new features.

    Raises:
        ValueError: If requested features are not found.
    """
    # Convert the main dataframe to LazyFrame
    to_data = _get_lazyframe(to_data)
    
    # Validate from_data
    from_data = _check_datacache(from_data)
    
    # Validate requested features exist in from_data
    missing_features = set(features) - set(_get_cols(from_data))
    if missing_features:
        raise ValueError(f"Requested features not found in from_data: {missing_features}")
        
    # Get current scenarios and available columns
    current_scenarios = get_current_scenarios(to_data)
    final_features = list(set(features) | set(_get_cols(to_data)))
    
    # Filter from_data with current scenarios and select all needed columns
    from_data = from_data.select(final_features)
    result = apply_filter(from_data, current_scenarios, lazy=True, as_pandas=False)
    

    return _process_output(result, lazy, as_pandas)




def add_data(
    to_data: DataFrameType,
    from_data: Optional[Union[DataFrameType, PathType]] = None,
    features: Optional[List[str]] = None,
    scenarios: Optional[Dict[str, Any]] = None,
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType:
    """Adds both features and scenarios to existing dataset.

    Args:
        to_data (DataFrameType): Target dataset.
        from_data (Optional[Union[DataFrameType, PathType]], optional): Source data. Defaults to None.
        features (Optional[List[str]], optional): List of features to add. Defaults to None.
        scenarios (Optional[Dict[str, Any]], optional): Dictionary of scenarios to add. Defaults to None.
        lazy (bool, optional): Whether to return LazyFrame. Defaults to False.
        to_pandas (bool, optional): Whether to convert to Pandas DataFrame. Defaults to False.

    Returns:
        DataFrameType: Updated dataset with new features and scenarios.

    Raises:
        ValueError: If neither features nor scenarios are provided.
    """
    
    if features is None and scenarios is None:
        raise ValueError("At least one of 'features' or 'scenarios' must be provided")
    
    result = _get_lazyframe(to_data)
    if features is not None:
        result = add_features(
            to_data=result,
            from_data=from_data,
            features=features,
            lazy=True,
            as_pandas=False
        )
    if scenarios is not None:
        result = add_scenarios(
            to_data=result,
            from_data=from_data,
            scenarios=scenarios,
            lazy=True,
            as_pandas=False
        )
    return _process_output(result, lazy, as_pandas)
