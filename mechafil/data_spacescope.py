import requests
import pandas as pd
import numpy as np
import datetime
from typing import Tuple, List

EXBI = 2**60
PIB = 2**50

NETWORK_START = datetime.datetime(2020, 10, 15)
DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS = 90
SPACESCOPE_AUTH_KEY = 'Bearer ghp_xJtTSVcNRJINLWMmfDangcIFCjqPUNZenoVe'

def get_historical_network_stats(
    start_date: datetime.date, current_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    power_df = query_spacescope_power_stats(start_date, current_date)
    onboards_df = query_spacescope_daily_power_onboarded(start_date, current_date)
    stats_df = query_spacescope_supply_stats(start_date, current_date)
    stats_df = stats_df.merge(power_df, on="date", how="inner").merge(
        onboards_df, on="date", how="inner"
    )
    rb_renewal_rate, day_renewed_qa_power = get_day_renewed_power_stats(
        start_date, current_date, end_date
    )
    stats_df["rb_renewal_rate"] = rb_renewal_rate
    stats_df["day_renewed_qa_power_pib"] = day_renewed_qa_power
    return stats_df


def get_sector_expiration_stats(
    start_date: datetime.date,
    current_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    scheduled_df = query_spacescope_sector_expirations(start_date, end_date)
    filter_scheduled_df = scheduled_df[
        scheduled_df["date"] >= pd.to_datetime(current_date, utc="UTC")
    ]
    rbp_expire_vec = filter_scheduled_df["total_rb"].values
    qap_expire_vec = filter_scheduled_df["total_qa"].values
    # we need the entire history of known_scheduled_pledge_release, so get the
    # data from the entire time-window, not just from current-date onwards
    pledge_release_vec = scheduled_df["total_pledge"].values
    return rbp_expire_vec, qap_expire_vec, pledge_release_vec


def get_day_renewed_power_stats(
    start_date: datetime.date,
    current_date: datetime.date,
    end_date: datetime.date,
) -> Tuple[np.array, np.array]:
    scheduled_df = query_spacescope_sector_expirations(start_date, end_date)
    filter_scheduled_df = scheduled_df[
        scheduled_df["date"] < pd.to_datetime(current_date, utc="UTC")
    ]
    rb_renewal_rate = (
        filter_scheduled_df["extended_rb"] / filter_scheduled_df["total_rb"]
    ).values
    day_renewed_qa_power = filter_scheduled_df["extended_qa"].values
    return rb_renewal_rate, day_renewed_qa_power


def chunk_dates(start_date: datetime.date, 
                end_date: datetime.date,
                chunks_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> List:
    chunk_start = start_date
    dates_chunked = []
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + datetime.timedelta(days=chunks_days), end_date)
        dates_chunked.append((chunk_start, chunk_end))

        chunk_start = chunk_end + datetime.timedelta(days=1)

    return dates_chunked

def query_spacescope_sector_expirations(
    start_date: datetime.date, 
    end_date: datetime.date,
    chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS
) -> pd.DataFrame:
    # See: https://docs.spacescope.io/network_core/power/#request-url-4
    #  NOTE: this is a bit weird compared to the rest of the Spacescope API, where scheduled expirations
    #  does not need a start/end date and returns the entire dataset.  For now, we use this and filter
    #  but this may need to change in the future if Spacescope changes their API.
    url = "https://api.spacescope.io/v2/power/sectors_schedule_expiration"
    scheduled_df = spacescope_query_to_df(url)
    
    # Convert bytes to pebibytes
    scheduled_df["extended_rb"] = scheduled_df["extended_bytes"].astype(float) / PIB
    scheduled_df["expired_rb"] = scheduled_df["expired_bytes"].astype(float) / PIB
    # scheduled_df["open_rb"] = scheduled_df["schedule_expire_bytes"].astype(float) / PIB
    scheduled_df["extended_qa"] = scheduled_df["extended_bytes_qap"].astype(float) / PIB
    scheduled_df["expired_qa"] = scheduled_df["expired_bytes_qap"].astype(float) / PIB
    # scheduled_df["open_qa"] = (
    #     scheduled_df["schedule_expire_bytes_qap"].astype(float) / PIB
    # )

    scheduled_df["extended_pledge"] = scheduled_df["extended_pledge"].astype(float)
    scheduled_df["expired_pledge"] = scheduled_df["expired_pledge"].astype(float)
    
    scheduled_df['schedule_expire_rb'] = scheduled_df["schedule_expire_bytes"].astype(float) / PIB
    scheduled_df["schedule_expire_qa"] = scheduled_df["schedule_expire_bytes_qap"].astype(float) / PIB
    scheduled_df["schedule_expire_pledge"] = scheduled_df["schedule_expire_pledge"].astype(float)

    # Total scheduled to expire, excluding terminated
    scheduled_df["total_rb"] = (
        scheduled_df["extended_rb"]
        + scheduled_df["expired_rb"]
        # + scheduled_df["open_rb"]
    )
    scheduled_df["total_qa"] = (
        scheduled_df["extended_qa"]
        + scheduled_df["expired_qa"]
        # + scheduled_df["open_qa"]
    )
    scheduled_df["total_pledge"] = (
        scheduled_df["extended_pledge"]
        + scheduled_df["expired_pledge"]
        # + scheduled_df["schedule_expire_pledge"].astype(float)
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


def query_spacescope_daily_power_onboarded(
    start_date: datetime.date, 
    end_date: datetime.date,
    chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS
) -> pd.DataFrame:
    url_template = "https://api.spacescope.io/v2/power/daily_power_onboarding_by_sector_size?end_date=%s&start_date=%s"
    df = spacescope_query(start_date, end_date, url_template, chunk_days)
    df['day_onboarded_rb_power_pib'] = (df['commit_rbp_32gib'] + df['commit_rbp_64gib']) / PIB
    df['day_onboarded_qa_power_pib'] = (df['commit_qap_32gib'] + df['commit_qap_64gib']) / PIB
    df['date'] = pd.to_datetime(df['stat_date']).dt.date
    
    # Filter columns
    onboards_df = df[
        ["date", "day_onboarded_rb_power_pib", "day_onboarded_qa_power_pib"]
    ]
    return onboards_df

def query_spacescope_supply_stats(
    start_date: datetime.date,
    end_date: datetime.date,
    chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS
) -> pd.DataFrame:
    url_template = "https://api.spacescope.io/v2/circulating_supply/circulating_supply?end_date=%s&start_date=%s"
    raw_stats_df = spacescope_query(start_date, end_date, url_template, chunk_days)
    # Convert metrics to float
    stats_df = raw_stats_df[
        [
            "circulating_fil",
            "mined_fil",
            "vested_fil",
            "locked_fil",
            "burnt_fil",
            "reserve_disbursed_fil"
        ]
    ].astype(float)
    # Convert dates to datetime dates
    stats_df["date"] = pd.to_datetime(raw_stats_df["stat_date"]).dt.date
    # Filter dates
    stats_df = stats_df[
        (stats_df["date"] >= start_date) & (stats_df["date"] <= end_date)
    ]
    return stats_df


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
    stats_df = query_spacescope_supply_stats(start_date, end_date)
    date_stats = stats_df[stats_df["date"] == date]
    return date_stats["vested_fil"].iloc[0]


def spacescope_query_to_df(url):
    payload={}
    headers = {
    'authorization': SPACESCOPE_AUTH_KEY
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    df = pd.DataFrame(response.json()['data'])
    return df

def spacescope_query(start_date: datetime.date, 
                     end_date: datetime.date,
                     url_template: str,
                     chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
    dates_chunked = chunk_dates(start_date, end_date, chunks_days=chunk_days)
    df_list = []
    for d in dates_chunked:
        chunk_start = d[0].strftime('%Y-%m-%d')
        chunk_end = d[1].strftime('%Y-%m-%d')
        url = url_template % (chunk_end, chunk_start)
        df = spacescope_query_to_df(url)
        df_list.append(df)
    
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

def query_historical_power(start_date: datetime.date, 
                           end_date: datetime.date,
                           chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
    url_template = "https://api.spacescope.io/v2/power/network_storage_capacity?end_date=%s&start_date=%s"
    df = spacescope_query(start_date, end_date, url_template, chunk_days)
    
    df['date'] = pd.to_datetime(df['stat_date']).dt.date
    df['total_qa_bytes_power'] = df['total_qa_bytes_power'].astype(float)
    df['total_raw_bytes_power'] = df['total_raw_bytes_power'].astype(float)
    df['baseline_power'] = df['baseline_power'].astype(float)

    return df

def query_spacescope_power_stats(
    start_date: datetime.date, end_date: datetime.date,
    chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS
) -> pd.DataFrame:
    power_df = query_historical_power(
        start_date, end_date, chunk_days=chunk_days
    )
    # Convert power stats to exibytes
    power_df["total_raw_power_eib"] = (
        power_df["total_raw_bytes_power"] / EXBI
    )
    power_df["total_qa_power_eib"] = (
        power_df["total_qa_bytes_power"] / EXBI
    )
    # Select final columns
    power_df = power_df[["date", "total_raw_power_eib", "total_qa_power_eib"]]
    return power_df

def query_historical_baseline_power(start_date: datetime.date = None,
                              end_date: datetime.date = None,
                              chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
    if start_date is None:
        start_date = NETWORK_START
    if end_date is None:
        end_date = datetime.datetime.today()

    historical_power_df = query_historical_power(
        start_date, end_date, chunk_days=chunk_days
    )

    bp_df = historical_power_df[['date', 'baseline_power']]
    bp_df = bp_df.rename(columns={'baseline_power': 'baseline'})
    return bp_df
    
def query_historical_rb_power(start_date: datetime.date = None,
                              end_date: datetime.date = None,
                              chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
    if start_date is None:
        start_date = NETWORK_START
    if end_date is None:
        end_date = datetime.datetime.today()

    historical_power_df = query_historical_power(
        start_date, end_date, chunk_days=chunk_days
    )

    rbp_df = historical_power_df[['date', 'total_raw_bytes_power']]
    rbp_df = rbp_df.rename(columns={'total_raw_bytes_power': 'rb_power'})
    return rbp_df


def query_historical_qa_power(start_date: datetime.date = None,
                              end_date: datetime.date = None,
                              chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
    if start_date is None:
        start_date = NETWORK_START
    if end_date is None:
        end_date = datetime.datetime.today()

    historical_power_df = query_historical_power(
        start_date, end_date, chunk_days=chunk_days
    )
    qap_df = historical_power_df[['date', 'total_raw_bytes_power']]
    qap_df = qap_df.rename(columns={'total_qa_bytes_power': 'rb_power'})

    return qap_df
