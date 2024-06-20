import pandas as pd
import requests
import pickle
import io
from typing import Any
import numpy as np

def load_data_from_url(url, is_pickle=False):
    response = requests.get(url)
    response.raise_for_status()
    if is_pickle:
        return pickle.load(io.BytesIO(response.content))
    return io.BytesIO(response.content)

def load_model_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def load_technicians_df(url):
    try:
        technicians_df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(f"Error loading technicians.csv: {e}")
    return technicians_df

def replace_nan_with_null(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: replace_nan_with_null(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_null(item) for item in data]
    elif isinstance(data, float) and np.isnan(data):
        return None
    else:
        return data