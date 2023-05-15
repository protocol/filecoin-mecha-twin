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
import mechafil.locking as locking
from .power import scalar_or_vector_to_vector
from .data import NETWORK_START

import scenario_generator.utils as u
from .locking import (
    compute_new_pledge_for_added_power
)

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
    fil_plus_rate: Union[np.array, float] = None,
    intervention_config: dict = None,
    fpr_hist_info: tuple = None,
) -> pd.DataFrame:
    start_day = (start_date - NETWORK_START.date()).days
    current_day = (current_date - NETWORK_START.date()).days
    end_day = (end_date - NETWORK_START.date()).days
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
    locked_fil = locked_fil_zero
    locked_reward = df['network_locked_reward'].iloc[0]
    sim_len = end_day - start_day
    renewal_rate_vec = scalar_or_vector_to_vector(renewal_rate, sim_len)

    #########################################################################################################
    # Setup for intervention
    cs_tvec = np.asarray([start_date + datetime.timedelta(days=x) for x in range(sim_len)])

    if intervention_config is not None:
        intervention_type = intervention_config['type']
        num_days_shock_behavior = intervention_config.get('num_days_shock_behavior', 360) 

        lock_target_update_date = intervention_config.get('lock_target_update_date', None)
        lock_target_update_value = intervention_config.get('lock_target_update_value', 0.3)

        upgrade_date = intervention_config['intervention_date']
        sim_start_date = intervention_config['simulation_start_date']

        consensus_pledge_base_before_intervention = intervention_config.get('consensus_pledge_base_before_intervention', 'circulating_supply').lower()
        consensus_pledge_base_after_intervention = intervention_config.get('consensus_pledge_base_after_intervention', 'circulating_supply').lower()
        onboard_ratio_callable = intervention_config.get('onboard_ratio_callable', locking.spec_onboard_ratio)

        # upgrade_day = (upgrade_date - sim_start_date).days
        upgrade_day = (upgrade_date - start_date).days  # CS simulation starts from start_date, not sim_start_date
    else:
        raise Exception("TODO")
    if fil_plus_rate is None:
        raise Exception("mechaFIL currently hardcoded for intervention - must supply FIL+ rate vector!")
    
    def compute_pledge_base(cs, network_locked, locked_reward, day_in):
        if day_in > upgrade_day:
            consensus_pledge_method = consensus_pledge_base_after_intervention
        else:
            consensus_pledge_method = consensus_pledge_base_before_intervention
        
        if consensus_pledge_method == 'circulating_supply':
            return cs
        elif consensus_pledge_method == 'available_supply':
            return cs + network_locked
        elif consensus_pledge_method == 'as_less_lockedrewards':
            avail_supply = cs + network_locked
            return avail_supply - locked_reward

    df_tmp = initialise_circulating_supply_df(
        start_date,
        end_date,
        circ_supply_zero,
        locked_fil_zero,
        burnt_fil_vec,
        vest_df,
        mint_df,
    )
    # run simulation loop to precompute metrics that need to change in the intervention timeperiod
    # while this is horribly inefficient, its more bug-free - but look to change this in the future
    current_day_idx = current_day - start_day
    scheduled_pledge_release_vec = np.zeros(sim_len)

    lock_target_in = lock_target
    for day_idx in range(1, sim_len):
        pledge_base = compute_pledge_base(circ_supply, locked_fil, locked_reward, day_idx)

        cur_date = start_date + datetime.timedelta(days=day_idx)
        if lock_target_update_date is not None:
            if cur_date == lock_target_update_date:
                lock_target_in = lock_target_update_value

        day_pledge_locked_vec = df_tmp["day_locked_pledge"].values
        scheduled_pledge_release = get_day_schedule_pledge_release(
            day_idx,
            current_day_idx,
            day_pledge_locked_vec,
            known_scheduled_pledge_release_vec,
            duration,
        )
        scheduled_pledge_release_vec[day_idx] = scheduled_pledge_release
        pledge_delta, onboards_delta, renews_delta = compute_day_delta_pledge(
            df_tmp["day_network_reward"].iloc[day_idx],
            pledge_base,
            df_tmp["day_onboarded_power_QAP"].iloc[day_idx],
            df_tmp["day_renewed_power_QAP"].iloc[day_idx],
            df_tmp["network_QAP"].iloc[day_idx],
            df_tmp["network_baseline"].iloc[day_idx],
            renewal_rate_vec[day_idx],
            scheduled_pledge_release,
            lock_target_in,
            onboard_ratio_callable,
        )
        day_locked_pledge, day_onboard_pledge, day_renewed_pledge = compute_day_locked_pledge(
            df_tmp["day_network_reward"].iloc[day_idx],
            pledge_base,
            df_tmp["day_onboarded_power_QAP"].iloc[day_idx],
            df_tmp["day_renewed_power_QAP"].iloc[day_idx],
            df_tmp["network_QAP"].iloc[day_idx],
            df_tmp["network_baseline"].iloc[day_idx],
            renewal_rate_vec[day_idx],
            scheduled_pledge_release,
            lock_target_in,
            onboard_ratio_callable,
        )
        # Compute daily change in block rewards collateral
        day_locked_rewards = compute_day_locked_rewards(
            df_tmp["day_network_reward"].iloc[day_idx]
        )
        day_reward_release = compute_day_reward_release(
            df_tmp["network_locked_reward"].iloc[day_idx - 1]
        )
        reward_delta = day_locked_rewards - day_reward_release
        # Update dataframe
        df_tmp["day_locked_pledge"].iloc[day_idx] = day_locked_pledge
        df_tmp["day_onboard_pledge"].iloc[day_idx] = day_onboard_pledge
        df_tmp["day_renewed_pledge"].iloc[day_idx] = day_renewed_pledge
        df_tmp["network_locked_pledge"].iloc[day_idx] = (
            df_tmp["network_locked_pledge"].iloc[day_idx - 1] + pledge_delta
        )
        df_tmp["network_locked_reward"].iloc[day_idx] = (
            df_tmp["network_locked_reward"].iloc[day_idx - 1] + reward_delta
        )
        df_tmp["network_locked"].iloc[day_idx] = (
            df_tmp["network_locked"].iloc[day_idx - 1] + pledge_delta + reward_delta
        )
        # Update gas burnt
        if df_tmp["network_gas_burn"].iloc[day_idx] == 0.0:
            df_tmp["network_gas_burn"].iloc[day_idx] = (
                df_tmp["network_gas_burn"].iloc[day_idx - 1] + daily_burnt_fil
            )
        # Find circulating supply balance and update
        circ_supply = (
            df_tmp["disbursed_reserve"].iloc[
                day_idx
            ]  # from initialise_circulating_supply_df
            + df_tmp["cum_network_reward"].iloc[day_idx]  # from the minting_model
            + df_tmp["total_vest"].iloc[day_idx]  # from vesting_model
            - df_tmp["network_locked"].iloc[day_idx]  # from simulation loop
            - df_tmp["network_gas_burn"].iloc[day_idx]  # comes from user inputs
        )
        df_tmp["circ_supply"].iloc[day_idx] = max(circ_supply, 0)
        locked_fil = df_tmp["network_locked"].iloc[day_idx]
        locked_reward = df_tmp["network_locked_reward"].iloc[day_idx]

    forecast_length = (end_date-current_date).days
    if fpr_hist_info is None:
        #t_fpr_hist, fpr_hist = u.get_historical_filplus_rate(datetime.date(2021,3,15), datetime.date(2022,12,1))
        #_, fpr_hist = u.get_historical_filplus_rate(datetime.date(2021,3,15), datetime.date(2022,12,1))
        raise ValueError('fpr_hist_info must be provided')
    else:
        t_fpr_hist = fpr_hist_info[0]  # unused but keep for API
        fpr_hist = fpr_hist_info[1]
    fpr_all = np.concatenate([fpr_hist, fil_plus_rate])
    fpr_all_simindex_start = len(fpr_hist)

    shock_days_vec = [upgrade_day + k for k in range(num_days_shock_behavior)]

    t_input_vec = np.asarray([sim_start_date + datetime.timedelta(days=x) for x in range(forecast_length)])
    t_input_intervention_start_ii = np.where(t_input_vec == upgrade_date)[0][0]
    cs_offset_ii = np.where(cs_tvec == upgrade_date)[0][0]
    df_offset_ii = np.where(df_tmp['date'] == upgrade_date)[0][0]
    tmp_duration = 365
    cc_fil_locked_in_window_vec = np.zeros(num_days_shock_behavior)
    cc_fil_locked_in_window_renewal_vec = np.zeros(num_days_shock_behavior)
    qap_renewed_during_window = np.zeros(num_days_shock_behavior)
    network_qap_byday_during_window = np.zeros(num_days_shock_behavior)
    pledge_renewed_power_window = np.zeros(num_days_shock_behavior)
    day_network_reward_vec = np.zeros(num_days_shock_behavior)
    cc_pct_at_time_of_onboard_and_renew_vec = np.zeros(num_days_shock_behavior)
    for jj in range(num_days_shock_behavior):
        jj_base = jj+t_input_intervention_start_ii
        rr_jj = renewal_rate_vec[jj_base]

        fpr_jj = jj_base+fpr_all_simindex_start-tmp_duration
        fpr_at_time_of_onboard_and_renew = fpr_all[fpr_jj] if fpr_jj > 0 else 0.001
        
        cc_fil_locked_in_window_vec[jj] = scheduled_pledge_release_vec[jj+cs_offset_ii] * (1-fpr_at_time_of_onboard_and_renew)
        cc_fil_locked_in_window_renewal_vec[jj] = scheduled_pledge_release_vec[jj+cs_offset_ii] * (1-fpr_at_time_of_onboard_and_renew) * rr_jj
        df_day = df_tmp.iloc[jj+df_offset_ii]
        qap_renewed_during_window[jj] = df_day['day_renewed_power_QAP']
        network_qap_byday_during_window[jj] = df_day['network_QAP']
        
        pledge_renewed_power_window[jj] = compute_new_pledge_for_added_power(
            df_day['day_network_reward'],
            compute_pledge_base(df_tmp.iloc[jj+df_offset_ii-1]['circ_supply'], 
                                df_tmp.iloc[jj+df_offset_ii-1]['network_locked'], 
                                df_tmp.iloc[jj+df_offset_ii-1]['network_locked_reward'], 
                                day_idx),
            qap_renewed_during_window[jj],
            network_qap_byday_during_window[jj],
            df_day['network_baseline'],
            lock_target,
            onboard_ratio_callable,
        )
        
        day_network_reward_vec[jj] = df_day['day_network_reward']
        cc_pct_at_time_of_onboard_and_renew_vec[jj] = (1-fpr_at_time_of_onboard_and_renew)
    # cc_fil_locked_in_window_total = np.sum(cc_fil_locked_in_window_vec)
    cc_fil_locked_in_window_total = np.sum(cc_fil_locked_in_window_renewal_vec)
    termination_fee_days=90
    termination_fee_in_FIL = np.mean(np.convolve(day_network_reward_vec*cc_pct_at_time_of_onboard_and_renew_vec, np.ones(termination_fee_days, dtype=int), 'valid'))
    
    # Simulation for loop

    ########################################################################################
    # NOTE: I think this reset of values was missing before and was a bug??
    circ_supply = circ_supply_zero
    locked_fil = locked_fil_zero
    ########################################################################################

    current_day_idx = current_day - start_day
    lock_target_in = lock_target
    for day_idx in range(1, sim_len):
        pledge_base = compute_pledge_base(circ_supply, locked_fil, locked_reward, day_idx)

        cur_date = start_date + datetime.timedelta(days=day_idx)
        if lock_target_update_date is not None:
            if cur_date == lock_target_update_date:
                lock_target_in = lock_target_update_value


        # Compute daily change in initial pledge collateral
        day_pledge_locked_vec = df["day_locked_pledge"].values
        scheduled_pledge_release = get_day_schedule_pledge_release(
            day_idx,
            current_day_idx,
            day_pledge_locked_vec,
            known_scheduled_pledge_release_vec,
            duration,
        )
        if intervention_type == 'cc_early_terminate_and_onboard' or intervention_type == 'cc_early_renewal':
            if day_idx in shock_days_vec:
                scheduled_pledge_release -= cc_fil_locked_in_window_vec[day_idx-cs_offset_ii]
        
        pledge_delta, onboards_delta, renews_delta = compute_day_delta_pledge(
            df["day_network_reward"].iloc[day_idx],
            pledge_base,
            df["day_onboarded_power_QAP"].iloc[day_idx],
            df["day_renewed_power_QAP"].iloc[day_idx],
            df["network_QAP"].iloc[day_idx],
            df["network_baseline"].iloc[day_idx],
            renewal_rate_vec[day_idx],
            scheduled_pledge_release,
            lock_target_in,
            onboard_ratio_callable,
        )
        if intervention_type == 'cc_early_renewal' and day_idx == cs_offset_ii:
            pledge_delta -= cc_fil_locked_in_window_total
        
        ###### Helpful outputs for probing system ######
        df['scheduled_pledge_release'].iloc[day_idx] = scheduled_pledge_release
        df['pledge_delta'].iloc[day_idx] = pledge_delta
        df['onboards_delta'].iloc[day_idx] = onboards_delta
        df['renews_delta'].iloc[day_idx] = renews_delta
        #######################
        # Get total locked pledge (needed for future day_locked_pledge)
        day_locked_pledge, day_onboard_pledge, day_renewed_pledge = compute_day_locked_pledge(
            df["day_network_reward"].iloc[day_idx],
            pledge_base,
            df["day_onboarded_power_QAP"].iloc[day_idx],
            df["day_renewed_power_QAP"].iloc[day_idx],
            df["network_QAP"].iloc[day_idx],
            df["network_baseline"].iloc[day_idx],
            renewal_rate_vec[day_idx],
            scheduled_pledge_release,
            lock_target_in,
            onboard_ratio_callable,
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
        df["day_onboard_pledge"].iloc[day_idx] = day_onboard_pledge
        df["day_renewed_pledge"].iloc[day_idx] = day_renewed_pledge
        df["network_locked_pledge"].iloc[day_idx] = (
            df["network_locked_pledge"].iloc[day_idx - 1] + pledge_delta
        )
        df["network_locked_reward"].iloc[day_idx] = (
            df["network_locked_reward"].iloc[day_idx - 1] + reward_delta
        )
        df["network_locked"].iloc[day_idx] = (
            df["network_locked"].iloc[day_idx - 1] + pledge_delta + reward_delta
        )
        if intervention_type == 'cc_early_terminate_and_onboard':
            if day_idx == cs_offset_ii:
                df["network_locked"].iloc[day_idx] -= cc_fil_locked_in_window_total
        # Update gas burnt
        if df["network_gas_burn"].iloc[day_idx] == 0.0:
            df["network_gas_burn"].iloc[day_idx] = (
                df["network_gas_burn"].iloc[day_idx - 1] + daily_burnt_fil
            )
        # if intervention_type == 'cc_early_terminate_and_onboard':
        if intervention_type == 'cc_early_terminate_and_onboard' or intervention_type == 'cc_early_renewal':
            if day_idx == cs_offset_ii:
                df["network_gas_burn"].iloc[day_idx] += termination_fee_in_FIL
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
        locked_fil = df_tmp["network_locked"].iloc[day_idx]
        locked_reward = df_tmp["network_locked_reward"].iloc[day_idx]

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
            "day_renewed_pledge": np.zeros(len_sim),
            "network_locked_pledge": np.zeros(len_sim),
            "network_locked": np.zeros(len_sim),
            "network_locked_reward": np.zeros(len_sim),
            "disbursed_reserve": np.ones(len_sim)
            * (17066618961773411890063046 * 10**-18),

            "pledge_delta": np.zeros(len_sim),
            "onboards_delta": np.zeros(len_sim),
            "renews_delta": np.zeros(len_sim),
            "scheduled_pledge_release": np.zeros(len_sim),
            "day_onboard_pledge": np.zeros(len_sim),
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
