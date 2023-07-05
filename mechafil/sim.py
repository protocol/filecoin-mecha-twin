import os

import pandas as pd
import numpy as np
import datetime
from datetime import date, timedelta
from typing import Union, Dict

import mechafil.data as mecha_data
from .data import get_historical_network_stats, get_sector_expiration_stats, setup_spacescope
from .power import (
    forecast_power_stats,
    build_full_power_stats_df,
    scalar_or_vector_to_vector,
)
from .vesting import compute_vesting_trajectory_df
from .minting import compute_minting_trajectory_df
from .supply import forecast_circulating_supply_df
import mechafil.minting as minting

from .utils import validate_qap_method

def setup_data_access(bearer_token_or_cfg: str):
    setup_spacescope(bearer_token_or_cfg)

def validate_current_date(current_date: datetime.date):
    if current_date > (datetime.date.today() - datetime.timedelta(days=2)):
        raise ValueError("Current date must be at least 2 days in the past!")

def run_simple_sim(
    start_date: datetime.date,
    current_date: datetime.date,
    forecast_length: int,
    renewal_rate: Union[np.array, float],
    rb_onboard_power: Union[np.array, float],
    fil_plus_rate: Union[np.array, float],
    duration: int,
    bearer_token_or_cfg: str,
    qap_method: str = 'basic' # can be set to tunable or basic
                              # see: https://hackmd.io/O6HmAb--SgmxkjLWSpbN_A?view
) -> pd.DataFrame:
    validate_qap_method(qap_method)
    setup_data_access(bearer_token_or_cfg)
    validate_current_date(current_date)

    end_date = current_date + datetime.timedelta(days=forecast_length)
    # Get sector scheduled expirations
    res = get_sector_expiration_stats(start_date, current_date, end_date)
    rb_known_scheduled_expire_vec = res[0]
    qa_known_scheduled_expire_vec = res[1]
    known_scheduled_pledge_release_full_vec = res[2]
    # Get daily stats
    fil_stats_df = get_historical_network_stats(start_date, current_date, end_date)
    current_day_stats = fil_stats_df[fil_stats_df["date"] >= current_date].iloc[0]
    # Forecast power stats
    rb_power_zero = current_day_stats["total_raw_power_eib"] * 1024.0
    qa_power_zero = current_day_stats["total_qa_power_eib"] * 1024.0
    rb_power_df, qa_power_df = forecast_power_stats(
        rb_power_zero,
        qa_power_zero,
        rb_onboard_power,
        rb_known_scheduled_expire_vec,
        qa_known_scheduled_expire_vec,
        renewal_rate,
        fil_plus_rate,
        duration,
        forecast_length,
        qap_method=qap_method
    )
    rb_power_df["total_raw_power_eib"] = rb_power_df["total_power"] / 1024.0
    qa_power_df["total_qa_power_eib"] = qa_power_df["total_power"] / 1024.0
    power_df = build_full_power_stats_df(
        fil_stats_df,
        rb_power_df,
        qa_power_df,
        start_date,
        current_date,
        end_date,
    )
    # Forecast Vesting
    vest_df = compute_vesting_trajectory_df(start_date, end_date)
    # Forecast minting stats and baseline
    rb_total_power_eib = power_df["total_raw_power_eib"].values
    qa_total_power_eib = power_df["total_qa_power_eib"].values
    qa_day_onboarded_power_pib = power_df["day_onboarded_qa_power_pib"].values
    qa_day_renewed_power_pib = power_df["day_renewed_qa_power_pib"].values
    mint_df = compute_minting_trajectory_df(
        start_date,
        end_date,
        rb_total_power_eib,
        qa_total_power_eib,
        qa_day_onboarded_power_pib,
        qa_day_renewed_power_pib,
    )
    # Forecast circulating supply
    start_day_stats = fil_stats_df.iloc[0]
    circ_supply_zero = start_day_stats["circulating_fil"]
    locked_fil_zero = start_day_stats["locked_fil"]
    daily_burnt_fil = fil_stats_df["burnt_fil"].diff().mean()
    burnt_fil_vec = fil_stats_df["burnt_fil"].values
    forecast_renewal_rate_vec = scalar_or_vector_to_vector(
        renewal_rate, forecast_length
    )
    past_renewal_rate_vec = fil_stats_df["rb_renewal_rate"].values[:-1]
    renewal_rate_vec = np.concatenate(
        [past_renewal_rate_vec, forecast_renewal_rate_vec]
    )
    cil_df = forecast_circulating_supply_df(
        start_date,
        current_date,
        end_date,
        circ_supply_zero,
        locked_fil_zero,
        daily_burnt_fil,
        duration,
        renewal_rate_vec,
        burnt_fil_vec,
        vest_df,
        mint_df,
        known_scheduled_pledge_release_full_vec,
    )

    # supply_inputs = {
    #     "start_date": start_date,
    #     "current_date": current_date,
    #     "end_date": end_date,
    #     "circ_supply_zero": circ_supply_zero,
    #     "locked_fil_zero": locked_fil_zero,
    #     "daily_burnt_fil": daily_burnt_fil,
    #     "duration": duration,
    #     "renewal_rate_vec": renewal_rate_vec,
    #     "burnt_fil_vec": burnt_fil_vec,
    #     "vest_df": vest_df,
    #     "mint_df": mint_df,
    #     "known_scheduled_pledge_release_full_vec": known_scheduled_pledge_release_full_vec,
    # }

    return cil_df #, supply_inputs, rb_power_df, qa_power_df, power_df

def get_offline_data(bearer_token_or_auth_file:str, 
                        start_date:date, current_date:date, end_date:date):
    # setup data access
    setup_spacescope(bearer_token_or_auth_file)

    res = get_sector_expiration_stats(start_date, current_date, end_date)
    rb_known_scheduled_expire_vec = res[0]
    qa_known_scheduled_expire_vec = res[1]
    known_scheduled_pledge_release_full_vec = res[2]
    # Get daily stats
    fil_stats_df = get_historical_network_stats(start_date, current_date, end_date)

    start_vested_amt = mecha_data.get_vested_amount(start_date)
    zero_cum_capped_power = mecha_data.get_cum_capped_rb_power(start_date)
    init_baseline = minting.compute_baseline_power_array(start_date, end_date)

    data_dict = {
        "rb_known_scheduled_expire_vec": rb_known_scheduled_expire_vec,
        "qa_known_scheduled_expire_vec": qa_known_scheduled_expire_vec,
        "known_scheduled_pledge_release_full_vec": known_scheduled_pledge_release_full_vec,
        "fil_stats_df": fil_stats_df,
        "start_vested_amt": start_vested_amt,
        "zero_cum_capped_power": zero_cum_capped_power,
        "init_baseline": init_baseline
    }

    return data_dict

def run_simple_sim_offline(
    start_date: datetime.date,
    current_date: datetime.date,
    forecast_length: int,
    renewal_rate: Union[np.array, float],
    rb_onboard_power: Union[np.array, float],
    fil_plus_rate: Union[np.array, float],
    duration: int,
    data: Dict,
    qap_method: str = 'basic' # can be set to tunable or basic
                              # see: https://hackmd.io/O6HmAb--SgmxkjLWSpbN_A?view
) -> pd.DataFrame:
    validate_qap_method(qap_method)
    validate_current_date(current_date)

    end_date = current_date + datetime.timedelta(days=forecast_length)

    # extract data
    fil_stats_df = data["fil_stats_df"]
    rb_known_scheduled_expire_vec = data["rb_known_scheduled_expire_vec"]
    qa_known_scheduled_expire_vec = data["qa_known_scheduled_expire_vec"]
    known_scheduled_pledge_release_full_vec = data["known_scheduled_pledge_release_full_vec"]
    start_vested_amt = data["start_vested_amt"]
    zero_cum_capped_power = data["zero_cum_capped_power"]
    init_baseline = data["init_baseline"]

    current_day_stats = fil_stats_df[fil_stats_df["date"] >= current_date].iloc[0]
    # Forecast power stats
    rb_power_zero = current_day_stats["total_raw_power_eib"] * 1024.0
    qa_power_zero = current_day_stats["total_qa_power_eib"] * 1024.0
    rb_power_df, qa_power_df = forecast_power_stats(
        rb_power_zero,
        qa_power_zero,
        rb_onboard_power,
        rb_known_scheduled_expire_vec,
        qa_known_scheduled_expire_vec,
        renewal_rate,
        fil_plus_rate,
        duration,
        forecast_length,
        qap_method=qap_method
    )
    rb_power_df["total_raw_power_eib"] = rb_power_df["total_power"] / 1024.0
    qa_power_df["total_qa_power_eib"] = qa_power_df["total_power"] / 1024.0
    power_df = build_full_power_stats_df(
        fil_stats_df,
        rb_power_df,
        qa_power_df,
        start_date,
        current_date,
        end_date,
    )
    # Forecast Vesting
    vest_df = compute_vesting_trajectory_df(start_date, end_date, start_vested_amt)
    # Forecast minting stats and baseline
    rb_total_power_eib = power_df["total_raw_power_eib"].values
    qa_total_power_eib = power_df["total_qa_power_eib"].values
    qa_day_onboarded_power_pib = power_df["day_onboarded_qa_power_pib"].values
    qa_day_renewed_power_pib = power_df["day_renewed_qa_power_pib"].values
    mint_df = compute_minting_trajectory_df(
        start_date,
        end_date,
        rb_total_power_eib,
        qa_total_power_eib,
        qa_day_onboarded_power_pib,
        qa_day_renewed_power_pib,
        baseline_power_array=init_baseline,
        zero_cum_capped_power=zero_cum_capped_power,
    )
    # Forecast circulating supply
    start_day_stats = fil_stats_df.iloc[0]
    circ_supply_zero = start_day_stats["circulating_fil"]
    locked_fil_zero = start_day_stats["locked_fil"]
    daily_burnt_fil = fil_stats_df["burnt_fil"].diff().mean()
    burnt_fil_vec = fil_stats_df["burnt_fil"].values
    forecast_renewal_rate_vec = scalar_or_vector_to_vector(
        renewal_rate, forecast_length
    )
    past_renewal_rate_vec = fil_stats_df["rb_renewal_rate"].values[:-1]
    renewal_rate_vec = np.concatenate(
        [past_renewal_rate_vec, forecast_renewal_rate_vec]
    )
    cil_df = forecast_circulating_supply_df(
        start_date,
        current_date,
        end_date,
        circ_supply_zero,
        locked_fil_zero,
        daily_burnt_fil,
        duration,
        renewal_rate_vec,
        burnt_fil_vec,
        vest_df,
        mint_df,
        known_scheduled_pledge_release_full_vec,
    )

    return cil_df
