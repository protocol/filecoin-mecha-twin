import requests
import pandas as pd
import numpy as np
import datetime

EXBI = 2**60
PIB = 2**50


def get_historical_network_stats(
    start_date: datetime.date, current_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    power_df = query_starboard_power_stats(start_date, current_date)
    onboards_df = query_starboard_daily_power_onboarded(start_date, current_date)
    stats_df = query_starboard_supply_stats(start_date, current_date)
    stats_df = stats_df.merge(power_df, on="date", how="inner").merge(
        onboards_df, on="date", how="inner"
    )
    stats_df["day_renewed_qa_power_pib"] = get_day_renewed_qa_power_stats(
        start_date, current_date, end_date
    )
    return stats_df


def get_sector_expiration_stats(
    start_date: datetime.date,
    current_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    scheduled_df = query_starboard_sector_expirations(start_date, end_date)
    filter_scheduled_df = scheduled_df[
        scheduled_df["date"] >= pd.to_datetime(current_date, utc="UTC")
    ]
    rbp_expire_vec = filter_scheduled_df["total_rb"].values
    qap_expire_vec = filter_scheduled_df["total_qa"].values
    # we need the entire history of known_scheduled_pledge_release, so get the
    # data from the entire time-window, not just from current-date onwards
    pledge_release_vec = scheduled_df["total_pledge"].values
    return rbp_expire_vec, qap_expire_vec, pledge_release_vec


def get_day_renewed_qa_power_stats(
    start_date: datetime.date,
    current_date: datetime.date,
    end_date: datetime.date,
) -> np.array:
    scheduled_df = query_starboard_sector_expirations(start_date, end_date)
    filter_scheduled_df = scheduled_df[
        scheduled_df["date"] < pd.to_datetime(current_date, utc="UTC")
    ]
    day_renewed_qa_power = filter_scheduled_df["extended_qa"].values
    return day_renewed_qa_power


def query_starboard_sector_expirations(
    start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    url = f"https://observable-api.starboard.ventures/getdata/sectors_schedule_expiration_full?start={str(start_date)}&end={str(end_date)}"
    r = requests.get(url)
    # Put data in dataframe
    scheduled_df = pd.DataFrame(r.json()["data"])
    # Convert bytes to pebibytes
    scheduled_df["extended_rb"] = scheduled_df["extended_bytes"].astype(float) / PIB
    scheduled_df["expired_rb"] = scheduled_df["expired_bytes"].astype(float) / PIB
    scheduled_df["open_rb"] = scheduled_df["potential_expire_bytes"].astype(float) / PIB
    scheduled_df["extended_qa"] = scheduled_df["extended_bytes_qap"].astype(float) / PIB
    scheduled_df["expired_qa"] = scheduled_df["expired_bytes_qap"].astype(float) / PIB
    scheduled_df["open_qa"] = (
        scheduled_df["potential_expire_bytes_qap"].astype(float) / PIB
    )
    # Total scheduled to expire, excluding terminated
    scheduled_df["total_rb"] = (
        scheduled_df["extended_rb"]
        + scheduled_df["expired_rb"]
        + scheduled_df["open_rb"]
    )
    scheduled_df["total_qa"] = (
        scheduled_df["extended_qa"]
        + scheduled_df["expired_qa"]
        + scheduled_df["open_qa"]
    )
    scheduled_df["total_pledge"] = (
        scheduled_df["extended_pledge"].astype(float)
        + scheduled_df["expired_pledge"].astype(float)
        + scheduled_df["potential_expire_pledge"].astype(float)
    )
    # Convert interest date to datetime
    scheduled_df["date"] = pd.to_datetime(scheduled_df["interest_date"])
    # Filter dates
    scheduled_df = scheduled_df[
        scheduled_df["date"] >= pd.to_datetime(start_date, utc="UTC")
    ]
    scheduled_df = scheduled_df[
        scheduled_df["date"] < pd.to_datetime(end_date, utc="UTC")
    ]
    return scheduled_df


def query_starboard_daily_power_onboarded(
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    # Get data from prove-commit-split-d API
    url = f"https://observable-api.starboard.ventures/api/v1/observable/prove-commit-split-d-v2?start={str(start_date)}&end={str(end_date)}"
    r = requests.get(url)
    onboards_df = pd.DataFrame(r.json()["data"])
    # Compute total onboardings
    onboards_df["day_onboarded_rb_power_pib"] = (
        onboards_df["half_size_byte"].astype(float)
        + onboards_df["size_byte"].astype(float)
    ) / PIB
    onboards_df["day_onboarded_qa_power_pib"] = (
        onboards_df["half_size_byte_qap"].astype(float)
        + onboards_df["size_byte_qap"].astype(float)
    ) / PIB
    # Convert dates to datetime
    onboards_df["date"] = pd.to_datetime(onboards_df["stat_date"]).dt.date
    # Filter dates
    onboards_df = onboards_df[
        (onboards_df["date"] >= start_date) & (onboards_df["date"] <= end_date)
    ]
    # Filter columns
    onboards_df = onboards_df[
        ["date", "day_onboarded_rb_power_pib", "day_onboarded_qa_power_pib"]
    ]
    return onboards_df


def query_starboard_supply_stats(
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    url = f"https://observable-api.starboard.ventures/api/v1/observable/circulating-supply?start={str(start_date)}&end={str(end_date)}"
    r = requests.get(url)
    raw_stats_df = pd.DataFrame(r.json()["data"])
    # Convert metrics to float
    stats_df = raw_stats_df[
        [
            "circulating_fil",
            "mined_fil",
            "vested_fil",
            "locked_fil",
            "burnt_fil",
        ]
    ].astype(float)
    # Convert dates to datetime dates
    stats_df["date"] = pd.to_datetime(raw_stats_df["stat_date"]).dt.date
    # Filter dates
    stats_df = stats_df[
        (stats_df["date"] >= start_date) & (stats_df["date"] <= end_date)
    ]
    return stats_df


def query_starboard_power_stats(
    start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    url = f"https://observable-api.starboard.ventures/network_storage_capacity?start={str(start_date)}&end={str(end_date)}"
    r = requests.get(url)
    power_df = pd.DataFrame(r.json()["data"])
    # Convert dates to datetime dates
    power_df["date"] = pd.to_datetime(power_df["stat_date"]).dt.date
    # Filter dates
    power_df = power_df[
        (power_df["date"] >= start_date) & (power_df["date"] <= end_date)
    ]
    # Convert power stats to exibytes
    power_df["total_raw_power_eib"] = (
        power_df["total_raw_bytes_power"].astype(float) / EXBI
    )
    power_df["total_qa_power_eib"] = (
        power_df["total_qa_bytes_power"].astype(float) / EXBI
    )
    # Select final columns
    power_df = power_df[["date", "total_raw_power_eib", "total_qa_power_eib"]]
    return power_df


def get_storage_baseline_value(date: datetime.date) -> float:
    # Get baseline values from Starboard API
    bp_df = query_historical_baseline_power()
    # Extract baseline value at date
    init_baseline_bytes = bp_df[bp_df["date"] >= pd.to_datetime(date, utc="UTC")].iloc[
        0, 1
    ]
    return init_baseline_bytes


def get_cum_capped_rb_power(date: datetime.date) -> float:
    # Query data sources and join
    rbp_df = query_historical_rb_power()
    bp_df = query_historical_baseline_power()
    df = pd.merge(rbp_df, bp_df, on="date", how="inner")
    # Compute cumulative capped RB power
    df["capped_power"] = np.min(df[["baseline", "rb_power"]].values, axis=1)
    df["cum_capped_power"] = df["capped_power"].cumsum()
    date_df = df[df["date"] >= pd.to_datetime(date, utc="UTC")]
    init_cum_capped_power = date_df["cum_capped_power"].iloc[0]
    return init_cum_capped_power


def get_cum_capped_qa_power(date: datetime.date) -> float:
    # Query data sources and join
    qap_df = query_historical_qa_power()
    bp_df = query_historical_baseline_power()
    df = pd.merge(qap_df, bp_df, on="date", how="inner")
    # Compute cumulative capped RB power
    df["capped_power"] = np.min(df[["baseline", "qa_power"]].values, axis=1)
    df["cum_capped_power"] = df["capped_power"].cumsum()
    date_df = df[df["date"] >= pd.to_datetime(date, utc="UTC")]
    init_cum_capped_power = date_df["cum_capped_power"].iloc[0]
    return init_cum_capped_power


def get_vested_amount(date: datetime.date) -> float:
    start_date = date - datetime.timedelta(days=1)
    end_date = date + datetime.timedelta(days=1)
    stats_df = query_starboard_supply_stats(start_date, end_date)
    date_stats = stats_df[stats_df["date"] == date]
    return date_stats["vested_fil"].iloc[0]


def query_historical_baseline_power() -> pd.DataFrame:
    url = f"https://observable-api.starboard.ventures/api/v1/observable/network-storage-capacity/new_baseline_power"
    r = requests.get(url)
    bp_df = pd.DataFrame(r.json()["data"])
    bp_df["date"] = pd.to_datetime(bp_df["stat_date"])
    bp_df["baseline"] = bp_df["new_baseline_power"].astype(float)
    bp_df = bp_df[["date", "baseline"]]
    return bp_df


def query_historical_rb_power() -> pd.DataFrame:
    url = f"https://observable-api.starboard.ventures/network_storage_capacity/total_raw_bytes_power"
    r = requests.get(url)
    rbp_df = pd.DataFrame(r.json()["data"])
    rbp_df["date"] = pd.to_datetime(rbp_df["stat_date"])
    rbp_df["rb_power"] = rbp_df["total_raw_bytes_power"].astype(float)
    rbp_df = rbp_df[["date", "rb_power"]]
    return rbp_df


def query_historical_qa_power() -> pd.DataFrame:
    url = f"https://observable-api.starboard.ventures/network_storage_capacity/total_qa_bytes_power"
    r = requests.get(url)
    qap_df = pd.DataFrame(r.json()["data"])
    qap_df["date"] = pd.to_datetime(qap_df["stat_date"])
    qap_df["qa_power"] = qap_df["total_qa_bytes_power"].astype(float)
    qap_df = qap_df[["date", "qa_power"]]
    return qap_df
