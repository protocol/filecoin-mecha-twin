import requests
import pandas as pd
import numpy as np
import datetime
from typing import Tuple, List
import os
import json

from .data import NETWORK_START

EXBI = 2**60
PIB = 2**50

DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS = 90
DEFAULT_AUTH_CONFIG = os.path.join(os.path.dirname(__file__), 'cfg', 'spacescope_auth.json')

class SpacescopeDataConnection:
    auth_token = ""

    def __init__(self, auth_config_or_token: str):
        if os.path.isfile(auth_config_or_token):
            # assume it is a JSON config file with key: auth_key
            try:
                with open(auth_config_or_token, 'r') as f:
                    config = json.load(f)
                    SpacescopeDataConnection.auth_token = config['auth_key']
            except:
                raise ValueError("Invalid auth config file: %s" % (auth_config_or_token,))
        else:
            SpacescopeDataConnection.auth_token = auth_config_or_token

    @classmethod
    def spacescope_query_to_df(cls, url):
        payload={}
        headers = {
        'authorization': cls.auth_token
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        df = pd.DataFrame(response.json()['data'])
        return df

    @staticmethod
    def get_historical_network_stats(
        start_date: datetime.date, current_date: datetime.date, end_date: datetime.date
    ) -> pd.DataFrame:
        power_df = SpacescopeDataConnection.query_spacescope_power_stats(start_date, current_date)
        onboards_df = SpacescopeDataConnection.query_spacescope_daily_power_onboarded(start_date, current_date)
        stats_df = SpacescopeDataConnection.query_spacescope_supply_stats(start_date, current_date)
        stats_df = stats_df.merge(power_df, on="date", how="inner").merge(
            onboards_df, on="date", how="inner"
        )
        renewal_df = SpacescopeDataConnection.get_day_renewed_power_stats(
            start_date, current_date, end_date
        )
        stats_df = stats_df.merge(renewal_df, on='date', how='inner')
        return stats_df

    @staticmethod
    def get_sector_expiration_stats(
        start_date: datetime.date,
        current_date: datetime.date,
        end_date: datetime.date,
    ) -> pd.DataFrame:
        scheduled_df = SpacescopeDataConnection.query_spacescope_sector_expirations(start_date, end_date)
        filter_scheduled_df = scheduled_df[
            scheduled_df["date"] >= pd.to_datetime(current_date, utc="UTC")
        ]
        rbp_expire_vec = filter_scheduled_df["total_rb"].values
        qap_expire_vec = filter_scheduled_df["total_qa"].values
        # we need the entire history of known_scheduled_pledge_release, so get the
        # data from the entire time-window, not just from current-date onwards
        pledge_release_vec = scheduled_df["total_pledge"].values
        return rbp_expire_vec, qap_expire_vec, pledge_release_vec

    @staticmethod
    def get_day_renewed_power_stats(
        start_date: datetime.date,
        current_date: datetime.date,
        end_date: datetime.date,
    ) -> Tuple[np.array, np.array]:
        scheduled_df = SpacescopeDataConnection.query_spacescope_sector_expirations(start_date, end_date)
        filter_scheduled_df = scheduled_df[
            scheduled_df["date"] <= pd.to_datetime(current_date, utc="UTC")
        ]
        rb_renewal_rate = (
            filter_scheduled_df["extended_rb"] / filter_scheduled_df["total_rb"]
        ).values
        day_renewed_qa_power = filter_scheduled_df["extended_qa"].values
        renewal_df = pd.DataFrame(
            {
                'date': pd.to_datetime(filter_scheduled_df['date']).dt.date,
                'rb_renewal_rate': rb_renewal_rate,
                'day_renewed_qa_power_pib': day_renewed_qa_power
            }
        )
        return renewal_df

    @staticmethod
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

    @staticmethod
    def query_spacescope_sector_expirations(
        start_date: datetime.date, 
        end_date: datetime.date,
    ) -> pd.DataFrame:
        # See: https://docs.spacescope.io/network_core/power/#request-url-4
        #  NOTE: this is a bit weird compared to the rest of the Spacescope API, where scheduled expirations
        #  does not need a start/end date and returns the entire dataset.  For now, we use this and filter
        #  but this may need to change in the future if Spacescope changes their API.
        url = "https://api.spacescope.io/v2/power/sectors_schedule_expiration"
        scheduled_df = SpacescopeDataConnection.spacescope_query_to_df(url)
        
        # Convert bytes to pebibytes
        scheduled_df["extended_rb"] = scheduled_df["extended_bytes"].astype(float) / PIB
        scheduled_df["expired_rb"] = scheduled_df["expired_bytes"].astype(float) / PIB
        scheduled_df["terminated_rb"] = scheduled_df["terminated_bytes"].astype(float) / PIB
        scheduled_df['schedule_expire_rb'] = scheduled_df["schedule_expire_bytes"].astype(float) / PIB

        scheduled_df["extended_qa"] = scheduled_df["extended_bytes_qap"].astype(float) / PIB
        scheduled_df["expired_qa"] = scheduled_df["expired_bytes_qap"].astype(float) / PIB
        scheduled_df['terminated_qa'] = scheduled_df['terminated_bytes_qap'].astype(float) / PIB
        scheduled_df["schedule_expire_qa"] = scheduled_df["schedule_expire_bytes_qap"].astype(float) / PIB

        scheduled_df["extended_pledge"] = scheduled_df["extended_pledge"].astype(float)
        scheduled_df["expired_pledge"] = scheduled_df["expired_pledge"].astype(float)
        scheduled_df["terminated_pledge"] = scheduled_df["terminated_pledge"].astype(float)
        scheduled_df["schedule_expire_pledge"] = scheduled_df["schedule_expire_pledge"].astype(float)

        # Total scheduled to expire, excluding terminated. Exclude terminated because 
        scheduled_df["total_rb"] = (
            scheduled_df["schedule_expire_rb"] - scheduled_df['terminated_rb']
        )
        scheduled_df["total_qa"] = (
            scheduled_df["schedule_expire_qa"] - scheduled_df['terminated_qa']
        )
        scheduled_df["total_pledge"] = (
            scheduled_df["schedule_expire_pledge"] - scheduled_df['terminated_pledge']
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

    @staticmethod
    def query_spacescope_daily_power_onboarded(
        start_date: datetime.date, 
        end_date: datetime.date,
        chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS
    ) -> pd.DataFrame:
        url_template = "https://api.spacescope.io/v2/power/daily_power_onboarding_by_sector_size?end_date=%s&start_date=%s"
        df = SpacescopeDataConnection.spacescope_query(start_date, end_date, url_template, chunk_days)
        df['day_onboarded_rb_power_pib'] = (df['commit_rbp_32gib'] + df['commit_rbp_64gib']) / PIB
        df['day_onboarded_qa_power_pib'] = (df['commit_qap_32gib'] + df['commit_qap_64gib']) / PIB
        df['date'] = pd.to_datetime(df['stat_date']).dt.date
        
        # Filter columns
        onboards_df = df[
            ["date", "day_onboarded_rb_power_pib", "day_onboarded_qa_power_pib"]
        ]
        return onboards_df

    @staticmethod
    def query_spacescope_supply_stats(
        start_date: datetime.date,
        end_date: datetime.date,
        chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS
    ) -> pd.DataFrame:
        url_template = "https://api.spacescope.io/v2/circulating_supply/circulating_supply?end_date=%s&start_date=%s"
        raw_stats_df = SpacescopeDataConnection.spacescope_query(start_date, end_date, url_template, chunk_days)
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

    @staticmethod
    def spacescope_query(start_date: datetime.date, 
                        end_date: datetime.date,
                        url_template: str,
                        chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
        dates_chunked = SpacescopeDataConnection.chunk_dates(start_date, end_date, chunks_days=chunk_days)
        df_list = []
        for d in dates_chunked:
            chunk_start = d[0].strftime('%Y-%m-%d')
            chunk_end = d[1].strftime('%Y-%m-%d')
            url = url_template % (chunk_end, chunk_start)
            df = SpacescopeDataConnection.spacescope_query_to_df(url)
            df_list.append(df)
        
        df_all = pd.concat(df_list, ignore_index=True)
        return df_all

    @staticmethod
    def query_historical_power(start_date: datetime.date, 
                            end_date: datetime.date,
                            chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
        url_template = "https://api.spacescope.io/v2/power/network_storage_capacity?end_date=%s&start_date=%s"
        df = SpacescopeDataConnection.spacescope_query(start_date, end_date, url_template, chunk_days)
        
        df['date'] = pd.to_datetime(df['stat_date']).dt.date
        df['total_qa_bytes_power'] = df['total_qa_bytes_power'].astype(float)
        df['total_raw_bytes_power'] = df['total_raw_bytes_power'].astype(float)
        df['baseline_power'] = df['baseline_power'].astype(float)

        return df

    @staticmethod
    def query_spacescope_power_stats(
        start_date: datetime.date, end_date: datetime.date,
        chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS
    ) -> pd.DataFrame:
        power_df = SpacescopeDataConnection.query_historical_power(
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

    @staticmethod
    def query_historical_baseline_power(start_date: datetime.date = None,
                                end_date: datetime.date = None,
                                chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
        if start_date is None:
            start_date = NETWORK_START
        if end_date is None:
            end_date = datetime.datetime.today()

        historical_power_df = SpacescopeDataConnection.query_historical_power(
            start_date, end_date, chunk_days=chunk_days
        )

        bp_df = historical_power_df[['date', 'baseline_power']]
        bp_df = bp_df.rename(columns={'baseline_power': 'baseline'})
        return bp_df
        
    @staticmethod
    def query_historical_rb_power(start_date: datetime.date = None,
                                end_date: datetime.date = None,
                                chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
        if start_date is None:
            start_date = NETWORK_START
        if end_date is None:
            end_date = datetime.datetime.today()

        historical_power_df = SpacescopeDataConnection.query_historical_power(
            start_date, end_date, chunk_days=chunk_days
        )

        rbp_df = historical_power_df[['date', 'total_raw_bytes_power']]
        rbp_df = rbp_df.rename(columns={'total_raw_bytes_power': 'rb_power'})
        return rbp_df

    @staticmethod
    def query_historical_qa_power(start_date: datetime.date = None,
                                end_date: datetime.date = None,
                                chunk_days: int = DEFAULT_SPACESCOPE_CHUNK_SIZE_IN_DAYS) -> pd.DataFrame:
        if start_date is None:
            start_date = NETWORK_START
        if end_date is None:
            end_date = datetime.datetime.today()

        historical_power_df = SpacescopeDataConnection.query_historical_power(
            start_date, end_date, chunk_days=chunk_days
        )
        qap_df = historical_power_df[['date', 'total_raw_bytes_power']]
        qap_df = qap_df.rename(columns={'total_qa_bytes_power': 'qa_power'})

        return qap_df