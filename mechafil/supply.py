import numpy as np
import pandas as pd

from .locking import (
    get_day_schedule_pledge_release,
    compute_day_reward_release,
    compute_day_delta_pledge,
    compute_day_locked_rewards,
    compute_day_locked_pledge,
)


def forecast_circulating_supply_df(
    start_day: int,
    end_day: int,
    current_day: int,
    circ_supply_zero: float,
    locked_fil_zero: float,
    daily_burnt_fil: float,
    duration: int,
    renewal_rate: float,
    burnt_fil_vec: np.array,
    vest_df: pd.DataFrame,
    mint_df: pd.DataFrame,
    known_scheduled_pledge_release_vec: np.array,
    lock_target: float = 0.3,
    use_termination_renewals: bool = False,
) -> pd.DataFrame:
    # initialise dataframe and auxilialy variables
    df = initialise_circulating_supply_df(
        start_day,
        end_day,
        circ_supply_zero,
        locked_fil_zero,
        burnt_fil_vec,
        mint_df,
        vest_df,
    )
    circ_supply = circ_supply_zero
    # Simulation for loop
    sim_len = end_day - start_day
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
            df["day_network_reward"][day_idx],
            circ_supply,
            df["day_onboarded_power_QAP"][day_idx],
            df["day_renewed_power_QAP"][day_idx],
            df["network_QAP"][day_idx],
            df["network_baseline"][day_idx],
            renewal_rate,
            scheduled_pledge_release,
            lock_target,
            use_termination_renewals,
        )
        # Get total locked pledge (needed for future day_locked_pledge)
        day_locked_pledge = compute_day_locked_pledge(
            df["day_network_reward"][day_idx],
            circ_supply,
            df["day_onboarded_power_QAP"][day_idx],
            df["day_renewed_power_QAP"][day_idx],
            df["network_QAP"][day_idx],
            df["network_baseline"][day_idx],
            renewal_rate,
            scheduled_pledge_release,
            lock_target,
            use_termination_renewals,
        )
        # Compute daily change in block rewards collateral
        day_locked_rewards = compute_day_locked_rewards(
            df["day_network_reward"][day_idx]
        )
        day_reward_release = compute_day_reward_release(
            df["network_locked_reward"][day_idx - 1]
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
            df["disbursed_reserve"][day_idx]  # from initialise_circulating_supply_df
            + df["cum_network_reward"][day_idx]  # from the minting_model
            + df["total_vest"][day_idx]  # from vesting_model
            - df["network_locked"][day_idx]  # from simulation loop
            - df["network_gas_burn"][day_idx]  # comes from user inputs
        )
        df["circ_supply"].iloc[day_idx] = max(circ_supply, 0)
    return df


def initialise_circulating_supply_df(
    start_day: int,
    end_day: int,
    circ_supply_zero: float,
    locked_fil_zero: float,
    burnt_fil_vec: np.array,
    vest_df: pd.DataFrame,
    mint_df: pd.DataFrame,
) -> pd.DataFrame:
    len_sim = end_day - start_day
    df = pd.DataFrame(
        {
            "days": np.arange(start_day, end_day),
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
    df["network_locked_pledge"].iloc[0] = locked_fil_zero / 2.0
    df["network_locked_reward"].iloc[0] = locked_fil_zero / 2.0
    df["network_locked"].iloc[0] = locked_fil_zero
    df["circ_supply"].iloc[0] = circ_supply_zero
    df = df.merge(vest_df, on="days", how="inner")
    df = df.merge(mint_df, on="days", how="inner")
    return df
