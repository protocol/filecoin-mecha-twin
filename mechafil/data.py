import requests
import pandas as pd
import numpy as np
import datetime
from typing import Tuple

from . import data_spacescope, data_starboard
DEFAULT_DATA_BACKEND = 'spacescope'

def get_historical_network_stats(
    start_date: datetime.date, 
    current_date: datetime.date, 
    end_date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> pd.DataFrame:
    if data_backend.lower() == 'spacescope':
        return data_spacescope.get_historical_network_stats(start_date, current_date, end_date)
    elif data_backend.lower() == 'starboard':
        return data_starboard.get_historical_network_stats(start_date, current_date, end_date)
    else:
        raise ValueError("Data Backend: %s not supported!" % (data_backend,))


def get_sector_expiration_stats(
    start_date: datetime.date,
    current_date: datetime.date,
    end_date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> pd.DataFrame:
    if data_backend.lower() == 'spacescope':
        return data_spacescope.get_sector_expiration_stats(start_date, current_date, end_date)
    elif data_backend.lower() == 'starboard':
        return data_starboard.get_sector_expiration_stats(start_date, current_date, end_date)
    else:
        raise ValueError("Data Backend: %s not supported!" % (data_backend,))

def get_day_renewed_power_stats(
    start_date: datetime.date,
    current_date: datetime.date,
    end_date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> Tuple[np.array, np.array]:
    if data_backend.lower() == 'spacescope':
        return data_spacescope.get_day_renewed_power_stats(start_date, current_date, end_date)
    elif data_backend.lower() == 'starboard':
        return data_starboard.get_day_renewed_power_stats(start_date, current_date, end_date)
    else:
        raise ValueError("Data Backend: %s not supported!" % (data_backend,))

def query_sector_expirations(
    start_date: datetime.date, 
    end_date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> pd.DataFrame:
    if data_backend.lower() == 'spacescope':
        return data_spacescope.query_spacescope_sector_expirations(start_date, end_date)
    elif data_backend.lower() == 'starboard':
        return data_starboard.query_starboard_sector_expirations(start_date, end_date)
    else:
        raise ValueError("Data Backend: %s not supported!" % (data_backend,))


def query_supply_stats(
    start_date: datetime.date,
    end_date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> pd.DataFrame:
    if data_backend.lower() == 'spacescope':
        return data_spacescope.query_spacescope_supply_stats(start_date, end_date)
    elif data_backend.lower() == 'starboard':
        return data_starboard.query_starboard_supply_stats(start_date, end_date)
    else:
        raise ValueError("Data Backend: %s not supported!" % (data_backend,))


def query_power_stats(
    start_date: datetime.date, 
    end_date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> pd.DataFrame:
    if data_backend.lower() == 'spacescope':
        return data_spacescope.query_spacescope_power_stats(start_date, end_date)
    elif data_backend.lower() == 'starboard':
        return data_starboard.query_starboard_power_stats(start_date, end_date)
    else:
        raise ValueError("Data Backend: %s not supported!" % (data_backend,))
    

def get_storage_baseline_value(
    date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> float:
    # Get baseline values from Starboard API
    bp_df = query_historical_baseline_power(data_backend)
    # Extract baseline value at date
    init_baseline_bytes = bp_df[bp_df["date"] >= pd.to_datetime(date, utc="UTC")].iloc[
        0, 1
    ]
    return init_baseline_bytes
    

def get_cum_capped_rb_power(
    date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> float:
    # Query data sources and join
    rbp_df = query_historical_rb_power(data_backend)
    bp_df = query_historical_baseline_power(data_backend)
    df = pd.merge(rbp_df, bp_df, on="date", how="inner")
    # Compute cumulative capped RB power
    df["capped_power"] = np.min(df[["baseline", "rb_power"]].values, axis=1)
    df["cum_capped_power"] = df["capped_power"].cumsum()
    date_df = df[df["date"] >= pd.to_datetime(date, utc="UTC")]
    init_cum_capped_power = date_df["cum_capped_power"].iloc[0]
    return init_cum_capped_power


def get_cum_capped_qa_power(
    date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> float:
    # Query data sources and join
    qap_df = query_historical_qa_power(data_backend)
    bp_df = query_historical_baseline_power(data_backend)
    df = pd.merge(qap_df, bp_df, on="date", how="inner")
    # Compute cumulative capped RB power
    df["capped_power"] = np.min(df[["baseline", "qa_power"]].values, axis=1)
    df["cum_capped_power"] = df["capped_power"].cumsum()
    date_df = df[df["date"] >= pd.to_datetime(date, utc="UTC")]
    init_cum_capped_power = date_df["cum_capped_power"].iloc[0]
    return init_cum_capped_power


def get_vested_amount(
    date: datetime.date,
    data_backend: str = DEFAULT_DATA_BACKEND
) -> float:
    start_date = date - datetime.timedelta(days=1)
    end_date = date + datetime.timedelta(days=1)
    stats_df = query_supply_stats(start_date, end_date, data_backend)
    date_stats = stats_df[stats_df["date"] == date]
    return date_stats["vested_fil"].iloc[0]


def query_historical_baseline_power(
    data_backend: str = DEFAULT_DATA_BACKEND
) -> pd.DataFrame:
    if data_backend.lower() == 'spacescope':
        return data_spacescope.query_historical_baseline_power()
    elif data_backend.lower() == 'starboard':
        return data_starboard.query_historical_baseline_power()
    else:
        raise ValueError("Data Backend: %s not supported!" % (data_backend,))


def query_historical_rb_power(
    data_backend: str = DEFAULT_DATA_BACKEND
) -> pd.DataFrame:
    if data_backend.lower() == 'spacescope':
        return data_spacescope.query_historical_rb_power()
    elif data_backend.lower() == 'starboard':
        return data_starboard.query_historical_rb_power()
    else:
        raise ValueError("Data Backend: %s not supported!" % (data_backend,))


def query_historical_qa_power(
    data_backend: str = DEFAULT_DATA_BACKEND
) -> pd.DataFrame:
    if data_backend.lower() == 'spacescope':
        return data_spacescope.query_historical_qa_power()
    elif data_backend.lower() == 'starboard':
        return data_starboard.query_historical_qa_power()
    else:
        raise ValueError("Data Backend: %s not supported!" % (data_backend,))
