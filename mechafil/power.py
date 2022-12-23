import numpy as np
import pandas as pd
import datetime
from typing import Callable, Tuple, Union
import numbers

from .utils import validate_qap_method

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
    sdm_cc_onboard_before_intervention: bool = False,
    sdm_cc_onboard_after_intervention: bool = True
) -> np.array:
    # If duration_m is not provided, qa_factor = 1.0 + 9.0 * fil_plus_rate
    # qa_factor = compute_qa_factor(fil_plus_rate, fil_plus_m, duration_m, duration)
    # qa_onboard_power = qa_factor * rb_onboard_power
    filplus_rbp = rb_onboard_power * fil_plus_rate
    notfilplus_rbp = rb_onboard_power * (1-fil_plus_rate)  # includes CC and regular deal, which both get SDM

    filplus_factor = fil_plus_m if duration_m is None else fil_plus_m * duration_m(duration)
    filplus_qap = filplus_rbp * filplus_factor

    rbp_factor = 1 if duration_m is None else duration_m(duration)
    # convert it to a vector to handle when durations happen and when they do not
    rbp_factor_vec = np.ones(forecast_lenght)
    if sdm_cc_onboard_before_intervention:
        rbp_factor_vec[0:intervention_day] = rbp_factor
    if sdm_cc_onboard_after_intervention:
        rbp_factor_vec[intervention_day:] = rbp_factor
    # print(np.mean(rbp_factor_vec), np.mean(rbp_factor_vec[0:intervention_day]), np.mean(rbp_factor_vec[intervention_day:]))
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
    sdm_cc_renew_before_intervention: bool = False,
    sdm_cc_renew_after_intervention: bool = True
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

    rb_power_to_renew = renewal_rate_vec[day_i] * day_rb_scheduled_expire_power_vec[day_i]
    filplus_renew = rb_power_to_renew * fpr
    notfilplus_renew = rb_power_to_renew * (1-fpr)  # includes CC and regular deal, which both get SDM

    filplus_factor = fil_plus_m if duration_m is None else fil_plus_m * duration_m(duration)
    filplus_qap = filplus_renew * filplus_factor

    rbp_factor_with_duration = 1 if duration_m is None else duration_m(duration)
    if day_i < intervention_day:
        if not sdm_cc_renew_before_intervention:
            rbp_factor = 1
        else:
            rbp_factor = rbp_factor_with_duration
    if day_i >= intervention_day:
        if not sdm_cc_renew_after_intervention:
            rbp_factor = 1
        else:
            rbp_factor = rbp_factor_with_duration
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
    return day_se_power


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
    validate_qap_method(qap_method)
    
    if intervention_config is not None:
        intervention_type = intervention_config['type']

        intervention_date = intervention_config['intervention_date']
        sim_start_date = intervention_config['simulation_start_date']
        intervention_day = (intervention_date - sim_start_date).days
        sdm_cc_onboard_before_intervention = intervention_config.get('sdm_cc_onboard_before_intervention', False)
        sdm_cc_onboard_after_intervention = intervention_config.get('sdm_cc_onboard_after_intervention', True)
        sdm_cc_renew_before_intervention = intervention_config.get('sdm_cc_renew_before_intervention', False)
        sdm_cc_renew_after_intervention = intervention_config.get('sdm_cc_renew_after_intervention', True)
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

    total_rb_onboarded_power = day_rb_onboarded_power.cumsum()
    day_qa_onboarded_power = forecast_qa_daily_onboardings(
        rb_onboard_power,
        fil_plus_rate,
        forecast_lenght,
        fil_plus_m,
        duration_m,
        duration,
        intervention_day=intervention_day,
        sdm_cc_onboard_before_intervention=sdm_cc_onboard_before_intervention,
        sdm_cc_onboard_after_intervention=sdm_cc_onboard_after_intervention,
    )
    total_qa_onboarded_power = day_qa_onboarded_power.cumsum()
    # Initialize scheduled expirations and renewals
    day_rb_scheduled_expire_power = np.zeros(forecast_lenght)
    day_rb_renewed_power = np.zeros(forecast_lenght)
    day_qa_scheduled_expire_power = np.zeros(forecast_lenght)
    day_qa_renewed_power = np.zeros(forecast_lenght)
    # Run loop to forecast daily scheduled expirations and renewals
    for day_i in range(forecast_lenght):
        # Raw-power stats
        day_rb_scheduled_expire_power[day_i] = compute_day_se_power(
            day_i,
            rb_known_scheduled_expire_vec,
            day_rb_onboarded_power,
            day_rb_renewed_power,
            duration,
        )

        if intervention_type == 'cc_early_renewal' and day_i == intervention_day:
            # expire additional cc power to simulate renewal of that power on the intervention day
            #   move it from scheduled_expire_vec from the future dates to the current date
            known_days_future_expire = len(rb_known_scheduled_expire_vec[intervention_day+1:])
            remaining_days_expire_model = 360 - known_days_future_expire  
            # 360 means we will stop assuming sectors more that expire more than 360 days from intervention day will NOT engage in this behavior
            # because CC sectors are usually 1 year?
            notfilplus_future_expire_power_modeled = 0
            for jj in range(remaining_days_expire_model):
                day_jj = intervention_day - 360 + jj
                notfilplus_future_expire_power_modeled += (day_rb_onboarded_power[day_jj] + day_rb_renewed_power[day_jj]) * (1-fil_plus_rate[day_jj])
                # remove that power from these two vectors so that they aren't double counted
                day_rb_onboarded_power[day_jj] *= fil_plus_rate[day_jj]
                day_rb_renewed_power[day_jj] *= fil_plus_rate[day_jj]

            notfilplus_future_expire_power_known = np.sum(rb_known_scheduled_expire_vec[intervention_day+1:] * (1-fil_plus_rate[intervention_day+1:intervention_day+1+known_days_future_expire]))
            notfilplus_future_expire_power = notfilplus_future_expire_power_known + notfilplus_future_expire_power_modeled
            day_rb_scheduled_expire_power[day_i] += notfilplus_future_expire_power
            
            # remove that power from future where it was borrowed from
            rb_known_scheduled_expire_vec[intervention_day+1:] = rb_known_scheduled_expire_vec[intervention_day+1:] * fil_plus_rate[intervention_day+1:intervention_day+1+known_days_future_expire]  # only FIL+ remains
        
        day_rb_renewed_power[day_i] = compute_basic_day_renewed_power(
            day_i, day_rb_scheduled_expire_power, renewal_rate_vec
        )
        if intervention_type == 'cc_early_renewal' and day_i == intervention_day:
            #   add the same power into the renewed power
            #   NOTE: we do this independent of renewal-rate b/c it is unclear how to change that vector synchronously
            day_rb_renewed_power[day_i] += notfilplus_future_expire_power
        
        # Quality-adjusted stats
        day_qa_scheduled_expire_power[day_i] = compute_day_se_power(
            day_i,
            qa_known_scheduled_expire_vec,
            day_qa_onboarded_power,
            day_qa_renewed_power,
            duration,
        )
        # see https://hackmd.io/O6HmAb--SgmxkjLWSpbN_A?view for more details
        if qap_method == 'tunable':
            day_qa_renewed_power[day_i] = compute_day_qa_renewed_power(
                day_i,
                day_rb_scheduled_expire_power,
                renewal_rate_vec,
                fil_plus_rate,
                fil_plus_m,
                duration_m,
                duration,
                intervention_day=intervention_day,
                sdm_cc_renew_before_intervention=sdm_cc_renew_before_intervention,
                sdm_cc_renew_after_intervention=sdm_cc_renew_after_intervention
            )
        elif qap_method == 'basic':
            day_qa_renewed_power[day_i] = compute_basic_day_renewed_power(
                day_i, day_qa_scheduled_expire_power, renewal_rate_vec
            )
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
    # Build DataFrames
    rb_df = pd.DataFrame(
        {
            "forecasting_step": np.arange(forecast_lenght),
            "onboarded_power": day_rb_onboarded_power,
            "cum_onboarded_power": total_rb_onboarded_power,
            "expire_scheduled_power": day_rb_scheduled_expire_power,
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
