import numpy as np
import pandas as pd
import datetime
from typing import Callable, Tuple, Union
import numbers

from .utils import validate_qap_method
import scenario_generator.utils as u

# --------------------------------------------------------------------------------------
#  Utility functions
# --------------------------------------------------------------------------------------
def scalar_or_vector_to_vector(
    input_x: Union[np.array, float], expected_len: int, err_msg: str = None
) -> np.array:
    if isinstance(input_x, numbers.Number):
        return np.ones(expected_len) * input_x
    else:
        err_msg_out = (
            "vector input does not match expected length!"
            if err_msg is None
            else err_msg
        )
        assert len(input_x) == expected_len, err_msg_out
        return input_x


# --------------------------------------------------------------------------------------
#  QA Multiplier functions
# --------------------------------------------------------------------------------------
def compute_qa_factor(
    fil_plus_rate: Union[np.array, float],
    fil_plus_m: float = 10.0,
    duration_m: Callable = None,
    duration: int = None,
) -> Union[np.array, float]:
    fil_plus_multipler = 1.0 + (fil_plus_m - 1) * fil_plus_rate
    if duration_m is None:
        return fil_plus_multipler
    else:
        return duration_m(duration) * fil_plus_multipler


# --------------------------------------------------------------------------------------
#  Onboardings
# --------------------------------------------------------------------------------------
def forecast_rb_daily_onboardings(
    rb_onboard_power: float, forecast_lenght: int
) -> np.array:
    rb_onboarded_power_vec = scalar_or_vector_to_vector(
        rb_onboard_power,
        forecast_lenght,
        err_msg="If rb_onboard_power is provided as a vector, it must be the same length as the forecast length",
    )
    return rb_onboarded_power_vec


def forecast_qa_daily_onboardings(
    rb_onboard_power: Union[np.array, float],
    fil_plus_rate: Union[np.array, float],
    forecast_lenght: int,
    fil_plus_m: float = 10.0,
    duration_m: Callable = None,
    duration: int = None,
    intervention_day: int = None,
    sdm_onboard_before_intervention: bool = False,
    sdm_onboard_after_intervention: bool = True
) -> np.array:
    # If duration_m is not provided, qa_factor = 1.0 + 9.0 * fil_plus_rate
    # qa_factor = compute_qa_factor(fil_plus_rate, fil_plus_m, duration_m, duration)
    # qa_onboard_power = qa_factor * rb_onboard_power
    filplus_rbp = rb_onboard_power * fil_plus_rate
    notfilplus_rbp = rb_onboard_power * (1-fil_plus_rate)  # includes CC and regular deal, which both get SDM

    # filplus_factor_with_duration = fil_plus_m if duration_m is None else fil_plus_m * duration_m(duration)
    filplus_factor_vec = np.ones(forecast_lenght) * fil_plus_m
    if sdm_onboard_before_intervention:
        filplus_factor_vec[0:intervention_day] *= duration_m(duration)
    if sdm_onboard_after_intervention:
        filplus_factor_vec[intervention_day:] *= duration_m(duration)
    filplus_qap = filplus_rbp * filplus_factor_vec

    rbp_factor = 1 if duration_m is None else duration_m(duration)
    # convert it to a vector to handle when durations happen and when they do not
    rbp_factor_vec = np.ones(forecast_lenght)
    if sdm_onboard_before_intervention:
        rbp_factor_vec[0:intervention_day] = rbp_factor
    if sdm_onboard_after_intervention:
        rbp_factor_vec[intervention_day:] = rbp_factor
    notfilplus_qap = notfilplus_rbp * rbp_factor_vec

    qa_onboard_power = filplus_qap + notfilplus_qap
    
    qa_onboard_power_vec = scalar_or_vector_to_vector(
        qa_onboard_power,
        forecast_lenght,
        err_msg="If qa_onboard_power is provided as a vector, it must be the same length as the forecast length",
    )
    return qa_onboard_power_vec


# --------------------------------------------------------------------------------------
#  Renewals
# --------------------------------------------------------------------------------------
def compute_basic_day_renewed_power(
    day_i: int,
    day_scheduled_expire_power_vec: np.array,
    renewal_rate_vec: np.array,
) -> float:
    day_renewed_power = renewal_rate_vec[day_i] * day_scheduled_expire_power_vec[day_i]
    return day_renewed_power


def compute_day_qa_renewed_power(
    day_i: int,
    day_rb_scheduled_expire_power_vec: np.array,
    renewal_rate_vec: np.array,
    fil_plus_rate: float,
    fil_plus_m: float = 10.0,
    duration_m: Callable = None,
    duration: int = None,
    intervention_day: int = None,
    sdm_renew_before_intervention: bool = False,
    sdm_renew_after_intervention: bool = True
) -> float:
    fpr = (
        fil_plus_rate
        if isinstance(fil_plus_rate, numbers.Number)
        else fil_plus_rate[day_i]
    )
    # qa_factor = compute_qa_factor(fpr, fil_plus_m, duration_m, duration)
    # day_renewed_power = (
    #     qa_factor * renewal_rate_vec[day_i] * day_rb_scheduled_expire_power_vec[day_i]
    # )
    # return day_renewed_power

    rb_power_to_renew = renewal_rate_vec[day_i] * day_rb_scheduled_expire_power_vec[day_i]
    filplus_renew = rb_power_to_renew * fpr
    notfilplus_renew = rb_power_to_renew * (1-fpr)  # includes CC and regular deal, which both get SDM

    filplus_factor_with_duration = fil_plus_m if duration_m is None else fil_plus_m * duration_m(duration)
    if day_i < intervention_day:
        if not sdm_renew_before_intervention:
            filplus_factor = fil_plus_m
        else:
            filplus_factor = filplus_factor_with_duration
    if day_i >= intervention_day:
        if not sdm_renew_after_intervention:
            filplus_factor = fil_plus_m
        else:
            filplus_factor = filplus_factor_with_duration
    # filplus_factor = 10
    filplus_qap = filplus_renew * filplus_factor

    rbp_factor_with_duration = 1 if duration_m is None else duration_m(duration)
    if day_i < intervention_day:
        if not sdm_renew_before_intervention:
            rbp_factor = 1
        else:
            rbp_factor = rbp_factor_with_duration
    if day_i >= intervention_day:
        if not sdm_renew_after_intervention:
            rbp_factor = 1
        else:
            rbp_factor = rbp_factor_with_duration
    # rbp_factor = 1
    notfilplus_qap = notfilplus_renew * rbp_factor

    day_renewed_power = filplus_qap + notfilplus_qap

    return day_renewed_power


# --------------------------------------------------------------------------------------
#  Scheduled expirations
# --------------------------------------------------------------------------------------
def compute_day_se_power(
    day_i: int,
    known_scheduled_expire_vec: np.array,
    day_onboard_vec: np.array,
    day_renewed_vec: np.array,
    duration: int,
) -> float:
    # Scheduled expirations coming from known active sectors
    if day_i > len(known_scheduled_expire_vec) - 1:
        known_day_se_power = 0.0
    else:
        known_day_se_power = known_scheduled_expire_vec[day_i]
    # Scheduled expirations coming from modeled sectors
    if day_i - duration >= 0:
        model_day_se_power = (
            day_onboard_vec[day_i - duration] + day_renewed_vec[day_i - duration]
        )
    else:
        model_day_se_power = 0.0
    # Total scheduled expirations
    day_se_power = known_day_se_power + model_day_se_power
    return day_se_power, known_day_se_power, model_day_se_power


# --------------------------------------------------------------------------------------
#  Forecast Power stats
# --------------------------------------------------------------------------------------
def forecast_power_stats(
    rb_power_zero: float,
    qa_power_zero: float,
    rb_onboard_power: Union[np.array, float],
    rb_known_scheduled_expire_vec: np.array,
    qa_known_scheduled_expire_vec: np.array,
    renewal_rate: Union[np.array, float],
    fil_plus_rate: Union[np.array, float],
    duration: int,
    forecast_lenght: int,
    fil_plus_m: float = 10.0,
    duration_m: Callable = None,
    qap_method: str = 'tunable',  # can be set to tunable or basic
                                  # see: https://hackmd.io/O6HmAb--SgmxkjLWSpbN_A?view
    intervention_config: dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # validate_qap_method(qap_method)

    t_fpr_hist, fpr_hist = u.get_historical_filplus_rate(datetime.date(2021,3,15), datetime.date(2022,12,1))
    t_fpr_cur = [datetime.date(2022,12,1) + datetime.timedelta(days=x) for x in range(forecast_lenght)]
    fpr_all = np.concatenate([fpr_hist, fil_plus_rate])
    fpr_all_simindex_start = len(fpr_hist)
    
    if intervention_config is not None:
        intervention_type = intervention_config['type']
        cc_reonboard_time_days = intervention_config.get('cc_reonboard_time_days', 30)

        intervention_date = intervention_config['intervention_date']
        sim_start_date = intervention_config['simulation_start_date']
        intervention_day = (intervention_date - sim_start_date).days
        sdm_onboard_before_intervention = intervention_config.get('sdm_onboard_before_intervention', False)
        sdm_onboard_after_intervention = intervention_config.get('sdm_onboard_after_intervention', True)
        sdm_renew_before_intervention = intervention_config.get('sdm_renew_before_intervention', False)
        sdm_renew_after_intervention = intervention_config.get('sdm_renew_after_intervention', True)
    else:
        raise Exception("TODO")

    # Forecast onboards
    renewal_rate_vec = scalar_or_vector_to_vector(
        renewal_rate,
        forecast_lenght,
        err_msg="If renewal_rate is provided as a vector, it must be the same length as the forecast length",
    )

    day_rb_onboarded_power = forecast_rb_daily_onboardings(
        rb_onboard_power, forecast_lenght
    )

    num_days_shock_behavior = 360
    shock_days_vec = [intervention_day + k for k in range(num_days_shock_behavior)]
    reonboard_days_vec = [intervention_day + k for k in range(1, cc_reonboard_time_days+1)]  # reonboard from intervention_day + 1

    # inputs are RenewalRate, RB Onboarding, and FIL+ rate.  
    # Setup the time indices for each to align the times when arrays are sliced.
    t_input_vec = np.asarray([sim_start_date + datetime.timedelta(days=x) for x in range(forecast_lenght)])
    t_input_intervention_start_ii = np.where(t_input_vec == intervention_date)[0][0]

    ########################################################################################################################
    ## Get the total power that will expire in the future that we want to move into the intervention day 
    
    # To do so, we run the simulation loop to determine how many modeled sectors would expire during that time-period
    #  we then back that power out of those times into the intervention day.  This is notably messy, but was the quickest
    #  way to simulate the effect we needed.  We can clean this up later.
    day_rb_scheduled_expire_power_tmp = np.zeros(forecast_lenght)
    day_rb_renewed_power_tmp = np.zeros(forecast_lenght)
    tmp_duration = 365
    for day_i in range(forecast_lenght):  # don't need to run the whole len
        rb_sched_expire_pwr_i, known_rb_se_power_i, model_rb_se_power_i = compute_day_se_power(
            day_i,
            rb_known_scheduled_expire_vec,
            day_rb_onboarded_power,
            day_rb_renewed_power_tmp,
            tmp_duration,
        )
        day_rb_scheduled_expire_power_tmp[day_i] = rb_sched_expire_pwr_i
        day_rb_renewed_power_tmp[day_i] = compute_basic_day_renewed_power(
            day_i, day_rb_scheduled_expire_power_tmp, renewal_rate_vec
        )
    notfilplus_future_expire_power_to_transfer = 0
    for jj in range(num_days_shock_behavior):
        jj_base = jj+t_input_intervention_start_ii
        rb_sched_expire_jj = day_rb_scheduled_expire_power_tmp[jj_base]
        rr_jj = renewal_rate_vec[jj_base]

        fpr_jj = jj_base+fpr_all_simindex_start-tmp_duration
        fpr_at_time_of_onboard_and_renew = fpr_all[fpr_jj] if fpr_jj > 0 else 0.001
        
        notfilplus_future_expire_power_to_transfer += rb_sched_expire_jj * (1-fpr_at_time_of_onboard_and_renew) * rr_jj
    print(notfilplus_future_expire_power_to_transfer)
    ########################################################################################################################
    
    day_qa_onboarded_power = forecast_qa_daily_onboardings(
        rb_onboard_power,
        fil_plus_rate,
        forecast_lenght,
        fil_plus_m,
        duration_m,
        duration,
        intervention_day=intervention_day,
        sdm_onboard_before_intervention=sdm_onboard_before_intervention,
        sdm_onboard_after_intervention=sdm_onboard_after_intervention,
    )
    # Initialize scheduled expirations and renewals
    day_rb_scheduled_expire_power = np.zeros(forecast_lenght)
    day_rb_scheduled_expire_power_known = np.zeros(forecast_lenght)
    day_rb_scheduled_expire_power_model = np.zeros(forecast_lenght)
    day_rb_renewed_power = np.zeros(forecast_lenght)
    day_qa_scheduled_expire_power = np.zeros(forecast_lenght)
    day_qa_scheduled_expire_power_known = np.zeros(forecast_lenght)
    day_qa_scheduled_expire_power_model = np.zeros(forecast_lenght)
    day_qa_renewed_power = np.zeros(forecast_lenght)
    # Run loop to forecast daily scheduled expirations and renewals
    for day_i in range(forecast_lenght):
        if intervention_type == 'cc_early_terminate_and_onboard' and day_i in reonboard_days_vec:
            rbp_to_reonboard = notfilplus_future_expire_power_to_transfer/cc_reonboard_time_days
            day_rb_onboarded_power[day_i] += rbp_to_reonboard
            day_qa_onboarded_power[day_i] += (rbp_to_reonboard * duration_m(duration))  # get SDM when re-onboarding the CC sectors

        # Raw-power stats
        rb_sched_expire_pwr_i, known_rb_se_power_i, model_rb_se_power_i = compute_day_se_power(
            day_i,
            rb_known_scheduled_expire_vec,
            day_rb_onboarded_power,
            day_rb_renewed_power,
            duration,
        )
        day_rb_scheduled_expire_power[day_i] = rb_sched_expire_pwr_i
        day_rb_scheduled_expire_power_known[day_i] = known_rb_se_power_i
        day_rb_scheduled_expire_power_model[day_i] = model_rb_se_power_i
        cc_scheduled_renew_power_day = day_rb_scheduled_expire_power[day_i] * (1-fil_plus_rate[day_i]) * renewal_rate_vec[day_i]
        
        # during the "shock" window, only expire FIL+ % of power that was originally set to expire
        # because it was transferred over to either renewal or terminate+reonboard
        if (intervention_type == 'cc_early_renewal' or intervention_type == 'cc_early_terminate_and_onboard') and day_i in shock_days_vec:
            # only expire FIL+ sectors, b/c we early renewed those sectors that were to be expired during this time-period
            day_rb_scheduled_expire_power[day_i] *= fil_plus_rate[day_i]
        
        day_rb_renewed_power[day_i] = compute_basic_day_renewed_power(
            day_i, day_rb_scheduled_expire_power, renewal_rate_vec
        )
        
        # Quality-adjusted stats
        qa_sched_expire_pwr_i, known_qa_se_power_i, model_qa_se_power_i = compute_day_se_power(
            day_i,
            qa_known_scheduled_expire_vec,
            day_qa_onboarded_power,
            day_qa_renewed_power,
            duration,
        )
        day_qa_scheduled_expire_power[day_i] = qa_sched_expire_pwr_i
        day_qa_scheduled_expire_power_known[day_i] = known_qa_se_power_i
        day_qa_scheduled_expire_power_model[day_i] = model_qa_se_power_i
        if (intervention_type == 'cc_early_renewal' or intervention_type == 'cc_early_terminate_and_onboard') and day_i in shock_days_vec:
            # only expire FIL+ sectors, b/c we early renewed/terminated those sectors that were to be expired during this time-period
            # account for this in the QA power by removing CC power from power that is to be expired, since it was already. We also don't
            # apply the SDM here becuase the CC that was expired didn't have SDM.
            day_qa_scheduled_expire_power[day_i] -= cc_scheduled_renew_power_day

        # see https://hackmd.io/O6HmAb--SgmxkjLWSpbN_A?view for more details
        if qap_method == 'tunable':
            if (intervention_type == 'cc_early_renewal' or intervention_type == 'cc_early_terminate_and_onboard') and day_i in shock_days_vec:
                temp_fpr = 1
                # since we moved the RBP power out of the expirations for this time period [intervention, intervention+num_days_shock_behavior]
                # the only remaining power in day_rb_scheduled_expire_power is FIL+.  Setting temp_fpr=1 effectively tells qa_renewed_power that all
                # the remaining power to be renewed is FIL+ power.  There is no CC power to renew in this time window.
                day_qa_renewed_power[day_i] = compute_day_qa_renewed_power(
                    day_i,
                    day_rb_scheduled_expire_power,
                    renewal_rate_vec,
                    temp_fpr,
                    fil_plus_m,
                    duration_m,
                    duration,
                    intervention_day=intervention_day,
                    sdm_renew_before_intervention=sdm_renew_before_intervention,
                    sdm_renew_after_intervention=sdm_renew_after_intervention
                )
            else:
                # ################## THE ORIGINAL ######################
                day_qa_renewed_power[day_i] = compute_day_qa_renewed_power(
                    day_i,
                    day_rb_scheduled_expire_power,
                    renewal_rate_vec,
                    fil_plus_rate,
                    fil_plus_m,
                    duration_m,
                    duration,
                    intervention_day=intervention_day,
                    sdm_renew_before_intervention=sdm_renew_before_intervention,
                    sdm_renew_after_intervention=sdm_renew_after_intervention
                )
                ########################################################
                # temp_fpr_ii = day_i-duration+fpr_all_simindex_start
                # temp_fpr = fpr_all[temp_fpr_ii] if temp_fpr_ii > 0 else 0
                # day_qa_renewed_power[day_i] = compute_day_qa_renewed_power(
                #     day_i,
                #     day_rb_scheduled_expire_power,
                #     renewal_rate_vec,
                #     temp_fpr,
                #     fil_plus_m,
                #     duration_m,
                #     duration,
                #     intervention_day=intervention_day,
                #     sdm_renew_before_intervention=sdm_renew_before_intervention,
                #     sdm_renew_after_intervention=sdm_renew_after_intervention
                # )
        elif qap_method == 'basic':
            day_qa_renewed_power[day_i] = compute_basic_day_renewed_power(
                day_i, day_qa_scheduled_expire_power, renewal_rate_vec
            )
        elif qap_method == 'basic-sdm':
            day_qa_renewed_power[day_i] = compute_basic_day_renewed_power(
                day_i, day_qa_scheduled_expire_power, renewal_rate_vec
            )
            sdm_factor_with_duration = 1 if duration_m is None else duration_m(duration)
            if day_i < intervention_day:
                if not sdm_renew_before_intervention:
                    sdm_factor = 1
                else:
                    sdm_factor = sdm_factor_with_duration
            if day_i >= intervention_day:
                if not sdm_renew_after_intervention:
                    sdm_factor = 1
                else:
                    sdm_factor = sdm_factor_with_duration
            day_qa_renewed_power[day_i] *= sdm_factor

        if intervention_type == 'cc_early_renewal' and day_i == intervention_day:
            #   add the same power into the renewed power
            #   NOTE: we do this here to be independent of renewal_rate b/c and renew the exact amount of power we choose to
            #     instead of having it be factored by the renewal_rate
            day_rb_renewed_power[day_i] += notfilplus_future_expire_power_to_transfer
            day_qa_renewed_power[day_i] += duration_m(duration)*notfilplus_future_expire_power_to_transfer  # gets SDM when renewed

    # compute total powers
    total_rb_onboarded_power = day_rb_onboarded_power.cumsum()
    total_qa_onboarded_power = day_qa_onboarded_power.cumsum()

    # Compute total scheduled expirations and renewals
    total_rb_scheduled_expire_power = day_rb_scheduled_expire_power.cumsum()
    total_rb_renewed_power = day_rb_renewed_power.cumsum()
    total_qa_scheduled_expire_power = day_qa_scheduled_expire_power.cumsum()
    total_qa_renewed_power = day_qa_renewed_power.cumsum()
    # Total RB power
    rb_power_zero_vec = np.ones(forecast_lenght) * rb_power_zero
    rb_total_power = (
        rb_power_zero_vec
        + total_rb_onboarded_power
        - total_rb_scheduled_expire_power
        + total_rb_renewed_power
    )
    # Total QA power
    qa_power_zero_vec = np.ones(forecast_lenght) * qa_power_zero
    qa_total_power = (
        qa_power_zero_vec
        + total_qa_onboarded_power
        - total_qa_scheduled_expire_power
        + total_qa_renewed_power
    )

    # terminate power to simulate early-termination, onboarding done in the loop above
    if intervention_type == 'cc_early_terminate_and_onboard':
        rb_total_power[intervention_day] -= notfilplus_future_expire_power_to_transfer
        qa_total_power[intervention_day] -= notfilplus_future_expire_power_to_transfer  # NO SDM when terminated

    # Build DataFrames
    rb_df = pd.DataFrame(
        {
            "forecasting_step": np.arange(forecast_lenght),
            "onboarded_power": day_rb_onboarded_power,
            "cum_onboarded_power": total_rb_onboarded_power,
            "expire_scheduled_power": day_rb_scheduled_expire_power,
            "expire_scheduled_power_known": day_rb_scheduled_expire_power_known,
            "expire_scheduled_power_model": day_rb_scheduled_expire_power_model,
            "cum_expire_scheduled_power": total_rb_scheduled_expire_power,
            "renewed_power": day_rb_renewed_power,
            "cum_renewed_power": total_rb_renewed_power,
            "total_power": rb_total_power,
        }
    )
    rb_df["power_type"] = "raw-byte"
    qa_df = pd.DataFrame(
        {
            "forecasting_step": np.arange(forecast_lenght),
            "onboarded_power": day_qa_onboarded_power,
            "cum_onboarded_power": total_qa_onboarded_power,
            "expire_scheduled_power": day_qa_scheduled_expire_power,
            "expire_scheduled_power_known": day_qa_scheduled_expire_power_known,
            "expire_scheduled_power_model": day_qa_scheduled_expire_power_model,
            "cum_expire_scheduled_power": total_qa_scheduled_expire_power,
            "renewed_power": day_qa_renewed_power,
            "cum_renewed_power": total_qa_renewed_power,
            "total_power": qa_total_power,
        }
    )
    qa_df["power_type"] = "quality-adjusted"
    return rb_df, qa_df


# --------------------------------------------------------------------------------------
#  Build power stats DataFrame
# --------------------------------------------------------------------------------------
def build_full_power_stats_df(
    stats_df: pd.DataFrame,
    rb_power_df: pd.DataFrame,
    qa_power_df: pd.DataFrame,
    start_date: datetime.date,
    current_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    # Past power
    past_power_df = stats_df[
        [
            "date",
            "total_raw_power_eib",
            "total_qa_power_eib",
            "day_onboarded_qa_power_pib",
            "day_renewed_qa_power_pib",
        ]
    ]
    # Forecasted power
    forecast_power_df = rb_power_df[["total_raw_power_eib"]]
    forecast_power_df.loc[:, "total_qa_power_eib"] = qa_power_df["total_qa_power_eib"]
    forecast_power_df.loc[:, "day_onboarded_qa_power_pib"] = qa_power_df[
        "onboarded_power"
    ]
    forecast_power_df.loc[:, "day_renewed_qa_power_pib"] = qa_power_df["renewed_power"]
    forecast_start_date = current_date + datetime.timedelta(days=1)
    forecast_power_df.loc[:, "date"] = pd.date_range(
        start=forecast_start_date, end=end_date, freq="d"
    )
    forecast_power_df["date"] = forecast_power_df["date"].dt.date

    forecast_power_df.loc[:, "rb_sched_expire_power"] = rb_power_df['expire_scheduled_power']
    forecast_power_df.loc[:, "rb_sched_expire_power_known"] = rb_power_df['expire_scheduled_power_known']
    forecast_power_df.loc[:, "rb_sched_expire_power_model"] = rb_power_df['expire_scheduled_power_model']
    forecast_power_df.loc[:, "day_renewed_rb_power_pib"] = rb_power_df["renewed_power"]
    forecast_power_df.loc[:, "qa_sched_expire_power"] = qa_power_df['expire_scheduled_power']
    forecast_power_df.loc[:, "qa_sched_expire_power_known"] = qa_power_df['expire_scheduled_power_known']
    forecast_power_df.loc[:, "qa_sched_expire_power_model"] = qa_power_df['expire_scheduled_power_model']

    # All power stats
    concat_df = pd.concat([past_power_df, forecast_power_df])
    power_df = pd.DataFrame(
        {"date": pd.date_range(start=start_date, end=end_date, freq="d")}
    )
    power_df["date"] = power_df["date"].dt.date
    power_df = power_df.merge(concat_df, on="date", how="left").fillna(
        method="backfill"
    )
    power_df = power_df.iloc[:-1]
    return power_df
