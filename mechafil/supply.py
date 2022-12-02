import numpy as np
import pandas as pd
import datetime
from typing import Union

from .locking import (
    get_day_schedule_pledge_release,
    compute_day_reward_release,
    compute_day_delta_pledge,
    compute_day_locked_rewards,
    compute_day_locked_pledge,
)
from .power import scalar_or_vector_to_vector

"""
There is still a small discrepancy between the actual locked FIL and forecasted
locked FIL. We believe that it could be due to the following reasons:
  a) Sector durations are not unlocked after exactly 1y. In general they're distributed and slightly longer. But in that case I’d expect the sign of the error to be the opposite to observed.
  b) The error between actual and forecasted locked FIL is 0 for day_idx=1. This might imply that a build up of errors due to an error in `day_locked_pledge` sounds more like it could be the issue.
  c) If we're sure the locking discrepancy is not a bug but rather a deficiency in the model popping up via the approximations used, we may way want to include a learnable factor to correct the difference
"""


def forecast_circulating_supply_df(
    start_date: datetime.date,
    current_date: datetime.date,
    end_date: datetime.date,
    circ_supply_zero: float,
    locked_fil_zero: float,
    daily_burnt_fil: float,
    duration: int,
    renewal_rate: Union[np.array, float],
    burnt_fil_vec: np.array,
    vest_df: pd.DataFrame,
    mint_df: pd.DataFrame,
    known_scheduled_pledge_release_vec: np.array,
    lock_target: float = 0.3,
) -> pd.DataFrame:
    # we assume all stats started at main net launch, in 2020-10-15
    start_day = (start_date - datetime.date(2020, 10, 15)).days
    current_day = (current_date - datetime.date(2020, 10, 15)).days
    end_day = (end_date - datetime.date(2020, 10, 15)).days
    # initialise dataframe and auxilialy variables
    df = initialise_circulating_supply_df(
        start_date,
        end_date,
        circ_supply_zero,
        locked_fil_zero,
        burnt_fil_vec,
        vest_df,
        mint_df,
    )
    circ_supply = circ_supply_zero
    sim_len = end_day - start_day
    renewal_rate_vec = scalar_or_vector_to_vector(renewal_rate, sim_len)
    # Simulation for loop
    current_day_idx = current_day - start_day
    for day_idx in range(1, sim_len):
        # Compute daily change in initial pledge collateral
        day_pledge_locked_vec = df["day_locked_pledge"].values
        scheduled_pledge_release = get_day_schedule_pledge_release(
            day_idx,
            current_day_idx,
            day_pledge_locked_vec,
            known_scheduled_pledge_release_vec,
            duration,
        )
        pledge_delta = compute_day_delta_pledge(
            df["day_network_reward"].iloc[day_idx],
            circ_supply,
            df["day_onboarded_power_QAP"].iloc[day_idx],
            df["day_renewed_power_QAP"].iloc[day_idx],
            df["network_QAP"].iloc[day_idx],
            df["network_baseline"].iloc[day_idx],
            renewal_rate_vec[day_idx],
            scheduled_pledge_release,
            lock_target,
        )
        # Get total locked pledge (needed for future day_locked_pledge)
        day_locked_pledge = compute_day_locked_pledge(
            df["day_network_reward"].iloc[day_idx],
            circ_supply,
            df["day_onboarded_power_QAP"].iloc[day_idx],
            df["day_renewed_power_QAP"].iloc[day_idx],
            df["network_QAP"].iloc[day_idx],
            df["network_baseline"].iloc[day_idx],
            renewal_rate_vec[day_idx],
            scheduled_pledge_release,
            lock_target,
        )
        # Compute daily change in block rewards collateral
        day_locked_rewards = compute_day_locked_rewards(
            df["day_network_reward"].iloc[day_idx]
        )
        day_reward_release = compute_day_reward_release(
            df["network_locked_reward"].iloc[day_idx - 1]
        )
        reward_delta = day_locked_rewards - day_reward_release
        # Update dataframe
        df["day_locked_pledge"].iloc[day_idx] = day_locked_pledge
        df["network_locked_pledge"].iloc[day_idx] = (
            df["network_locked_pledge"].iloc[day_idx - 1] + pledge_delta
        )
        df["network_locked_reward"].iloc[day_idx] = (
            df["network_locked_reward"].iloc[day_idx - 1] + reward_delta
        )
        df["network_locked"].iloc[day_idx] = (
            df["network_locked"].iloc[day_idx - 1] + pledge_delta + reward_delta
        )
        # Update gas burnt
        if df["network_gas_burn"].iloc[day_idx] == 0.0:
            df["network_gas_burn"].iloc[day_idx] = (
                df["network_gas_burn"].iloc[day_idx - 1] + daily_burnt_fil
            )
        # Find circulating supply balance and update
        circ_supply = (
            df["disbursed_reserve"].iloc[
                day_idx
            ]  # from initialise_circulating_supply_df
            + df["cum_network_reward"].iloc[day_idx]  # from the minting_model
            + df["total_vest"].iloc[day_idx]  # from vesting_model
            - df["network_locked"].iloc[day_idx]  # from simulation loop
            - df["network_gas_burn"].iloc[day_idx]  # comes from user inputs
        )
        df["circ_supply"].iloc[day_idx] = max(circ_supply, 0)
    return df


def initialise_circulating_supply_df(
    start_date: datetime.date,
    end_date: datetime.date,
    circ_supply_zero: float,
    locked_fil_zero: float,
    burnt_fil_vec: np.array,
    vest_df: pd.DataFrame,
    mint_df: pd.DataFrame,
) -> pd.DataFrame:
    # we assume days start at main net launch, in 2020-10-15
    start_day = (start_date - datetime.date(2020, 10, 15)).days
    end_day = (end_date - datetime.date(2020, 10, 15)).days
    len_sim = end_day - start_day
    df = pd.DataFrame(
        {
            "days": np.arange(start_day, end_day),
            "date": pd.date_range(start_date, end_date, freq="d")[:-1],
            "circ_supply": np.zeros(len_sim),
            "network_gas_burn": np.pad(
                burnt_fil_vec, (0, len_sim - len(burnt_fil_vec))
            ),
            "day_locked_pledge": np.zeros(len_sim),
            "network_locked_pledge": np.zeros(len_sim),
            "network_locked": np.zeros(len_sim),
            "network_locked_reward": np.zeros(len_sim),
            "disbursed_reserve": np.ones(len_sim)
            * (17066618961773411890063046 * 10**-18),
        }
    )
    df["date"] = df["date"].dt.date
    df["network_locked_pledge"].iloc[0] = locked_fil_zero / 2.0
    df["network_locked_reward"].iloc[0] = locked_fil_zero / 2.0
    df["network_locked"].iloc[0] = locked_fil_zero
    df["circ_supply"].iloc[0] = circ_supply_zero
    df = df.merge(vest_df, on="date", how="inner")
    df = df.merge(mint_df.drop(columns=["days"]), on="date", how="inner")
    return df
