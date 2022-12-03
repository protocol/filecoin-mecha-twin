import numpy as np
import pandas as pd
import datetime
from typing import Callable, Tuple, Union
import numbers

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
) -> np.array:
    # If duration_m is not provided, qa_factor = 1.0 + 9.0 * fil_plus_rate
    qa_factor = compute_qa_factor(fil_plus_rate, fil_plus_m, duration_m, duration)
    qa_onboard_power = qa_factor * rb_onboard_power
    qa_onboard_power_vec = scalar_or_vector_to_vector(
        qa_onboard_power,
        forecast_lenght,
        err_msg="If qa_onboard_power is provided as a vector, it must be the same length as the forecast length",
    )
    return qa_onboard_power_vec


# --------------------------------------------------------------------------------------
#  Renewals
# --------------------------------------------------------------------------------------
def compute_day_rb_renewed_power(
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
) -> float:
    fpr = (
        fil_plus_rate
        if isinstance(fil_plus_rate, numbers.Number)
        else fil_plus_rate[day_i]
    )
    qa_factor = compute_qa_factor(fpr, fil_plus_m, duration_m, duration)
    day_renewed_power = (
        qa_factor * renewal_rate_vec[day_i] * day_rb_scheduled_expire_power_vec[day_i]
    )
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        day_rb_renewed_power[day_i] = compute_day_rb_renewed_power(
            day_i, day_rb_scheduled_expire_power, renewal_rate_vec
        )
        # Quality-adjusted stats
        day_qa_scheduled_expire_power[day_i] = compute_day_se_power(
            day_i,
            qa_known_scheduled_expire_vec,
            day_qa_onboarded_power,
            day_qa_renewed_power,
            duration,
        )
        day_qa_renewed_power[day_i] = compute_day_qa_renewed_power(
            day_i,
            day_rb_scheduled_expire_power,
            renewal_rate_vec,
            fil_plus_rate,
            fil_plus_m,
            duration_m,
            duration,
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
