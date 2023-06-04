import datetime
import numpy as np
import pandas as pd

from .constants import EXA, EXBI, PIB
from .data import get_storage_baseline_value, \
    get_cum_capped_rb_power, get_cum_capped_qa_power


LAMBDA = np.log(2) / (
    6.0 * 365
)  # minting exponential reward decay rate (6yrs half-life)
FIL_BASE = 2_000_000_000.0
STORAGE_MINING = 0.55 * FIL_BASE
SIMPLE_ALLOC = 0.3 * STORAGE_MINING  # total simple minting allocation
BASELINE_ALLOC = 0.7 * STORAGE_MINING  # total baseline minting allocation
GROWTH_RATE = float(
    np.log(2) / 365.0
)  # daily baseline growth rate (the "g" from https://spec.filecoin.io/#section-systems.filecoin_token)

# NOTE: the baseline storage value is the baseline storage power at the genesis
# The spec notes that this value is 2.888888888, but the actual data from starboard
# shows that the value is 2.766213637444971.  We use the actual data here.
#
# Query:
# 3189227188947035000 from https://observable-api.starboard.ventures/api/v1/observable/network-storage-capacity/new_baseline_power
BASELINE_STORAGE = (
    2.766213637444971
    * EXA
    # the b_0 from https://spec.filecoin.io/#section-systems.filecoin_token
)

def compute_minting_trajectory_df(
    start_date: datetime.date,
    end_date: datetime.date,
    rb_total_power_eib: np.array,
    qa_total_power_eib: np.array,
    qa_day_onboarded_power_pib: np.array,
    qa_day_renewed_power_pib: np.array,
    minting_base: str = 'RBP'
) -> pd.DataFrame:
    # we assume minting started at main net launch, in 2020-10-15
    start_day = (start_date - datetime.date(2020, 10, 15)).days
    end_day = (end_date - datetime.date(2020, 10, 15)).days

    minting_base = minting_base.lower()
    capped_power_reference = 'network_RBP' if minting_base == 'rbp' else 'network_QAP'

    # Init dataframe
    df = pd.DataFrame(
        {
            "days": np.arange(start_day, end_day),
            "date": pd.date_range(start_date, end_date, freq="d")[:-1],
            "network_RBP": rb_total_power_eib * EXBI,
            "network_QAP": qa_total_power_eib * EXBI,
            "day_onboarded_power_QAP": qa_day_onboarded_power_pib * PIB,
            "day_renewed_power_QAP": qa_day_renewed_power_pib * PIB,
        }
    )
    df["date"] = df["date"].dt.date
    # Compute cumulative rewards due to simple minting
    df["cum_simple_reward"] = df["days"].pipe(cum_simple_minting)
    # Compute cumulative rewards due to baseline minting
    df["network_baseline"] = compute_baseline_power_array(start_date, end_date)

    df["capped_power"] = np.min(df[["network_baseline", capped_power_reference]].values, axis=1)
    zero_cum_capped_power = get_cum_capped_rb_power(start_date)
    df["cum_capped_power"] = df["capped_power"].cumsum() + zero_cum_capped_power
    df["network_time"] = df["cum_capped_power"].pipe(network_time)
    df["cum_baseline_reward"] = df["network_time"].pipe(cum_baseline_reward)
    # Add cumulative rewards and get daily rewards minted
    df["cum_network_reward"] = df["cum_baseline_reward"] + df["cum_simple_reward"]
    df["day_network_reward"] = df["cum_network_reward"].diff().fillna(method="backfill")

    return df


def cum_simple_minting(day: int) -> float:
    """
    Simple minting - the total number of tokens that should have been emitted
    by simple minting up until date provided.
    """
    return SIMPLE_ALLOC * (1 - np.exp(-LAMBDA * day))


def compute_baseline_power_array(
    start_date: datetime.date, end_date: datetime.date
) -> np.array:
    arr_len = (end_date - start_date).days
    exponents = np.arange(0, arr_len)
    init_baseline = get_storage_baseline_value(start_date)
    baseline_power_arr = init_baseline * np.exp(GROWTH_RATE * exponents)
    return baseline_power_arr


def network_time(cum_capped_power: float) -> float:
    b0 = BASELINE_STORAGE
    g = GROWTH_RATE
    return (1 / g) * np.log(((g * cum_capped_power) / b0) + 1)


def cum_baseline_reward(network_time: float) -> float:
    return BASELINE_ALLOC * (1 - np.exp(-LAMBDA * network_time))
