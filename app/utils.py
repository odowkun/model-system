import pandas as pd
import requests
import pickle
import io

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
