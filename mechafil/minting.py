import numpy as np
import pandas as pd

EXBI = 2**60
PIB = 2**50
EPOCH_PER_DAY = 2880
LAMBDA = np.log(2) / (
    6.0 * 365 * EPOCH_PER_DAY
)  # 30s epoch minting exponential reward decay rate
SIMPLE_ALLOC = 0.3 * 1.1 * 10**9  # total simple minting allocation
BASELINE_ALLOC = 0.7 * 1.1 * 10**9  # total baseline minting allocation
BASELINE_B0 = 2.88888888 * EXBI  # initial storage
BASELINE_R = np.log(2) / (2880 * 365)  # 1_051_200 in epochs


def compute_minting_trajectory_df(
    start_day: int,
    end_day: int,
    cum_capped_power_zero: float,
    rb_total_power_eib: np.array,
    qa_total_power_eib: np.array,
    qa_day_onboarded_power_pib: np.array,
    qa_day_renewed_power_pib: np.array,
) -> pd.DataFrame:
    start_epoch = start_day * EPOCH_PER_DAY
    end_epoch = end_day * EPOCH_PER_DAY
    df = pd.DataFrame(
        {
            "days": np.arange(start_day, end_day),
            "epoch": np.arange(start_epoch, end_epoch, EPOCH_PER_DAY),
            "network_RBP": rb_total_power_eib * EXBI,
            "network_QAP": qa_total_power_eib * EXBI,
            "day_onboarded_power_QAP": qa_day_onboarded_power_pib * PIB,
            "day_renewed_power_QAP": qa_day_renewed_power_pib * PIB,
        }
    )
    df.loc[:, "simple_reward_epoch"] = df["epoch"].pipe(simple_reward_epoch)
    df.loc[:, "network_baseline"] = df["epoch"].pipe(baseline_storage)
    df.loc[:, "capped_power"] = np.min(
        df[["network_baseline", "network_RBP"]].values, axis=1
    )
    cum_capped_power_zero_vec = np.ones(len(df)) * cum_capped_power_zero
    df.loc[:, "cum_capped_power"] = (
        cum_capped_power_zero_vec + EPOCH_PER_DAY * df["capped_power"].cumsum().values
    )
    df.loc[:, "cum_simple_reward"] = df["epoch"].pipe(cum_simple_reward)
    df.loc[:, "network_time"] = df["cum_capped_power"].pipe(network_time)
    df.loc[:, "cum_baseline_reward"] = df["network_time"].pipe(
        cum_baseline_reward_epoch
    )
    df.loc[:, "cum_network_reward"] = (
        df["cum_baseline_reward"] + df["cum_simple_reward"]
    )
    df.loc[:, "day_network_reward"] = (
        df["cum_network_reward"].diff().fillna(method="backfill")
    )
    df["network_QAP_growth"] = df["network_QAP"].diff().fillna(method="backfill")
    df["network_RBP_growth"] = df["network_RBP"].diff().fillna(method="backfill")
    df["network_QAP_percentgrowth_day"] = df["network_QAP_growth"] / df["network_QAP"]
    df["network_RBP_percentgrowth_day"] = df["network_RBP_growth"] / df["network_RBP"]
    return df


def simple_reward_epoch(epoch: float) -> float:
    """
    Exponential decay simple reward
    """
    return SIMPLE_ALLOC * LAMBDA * np.exp(-LAMBDA * epoch)


def baseline_storage(epoch: float) -> float:
    """
    Baseline storage target function
    epoch -- time in 30s second epochs since first mint

    """
    return BASELINE_B0 * np.exp(BASELINE_R * epoch)


def baseline_reward(capped_power: np.array, cum_capped_power: np.array) -> np.array:
    """
    Derivative of cum_baseline_reward_epoch
    """
    a = BASELINE_ALLOC * LAMBDA / BASELINE_B0
    b = capped_power
    c = (1 + (BASELINE_R / BASELINE_B0) * cum_capped_power) ** (1 + LAMBDA / BASELINE_R)
    return a * b / c


def cum_simple_reward(epoch):
    """
    Cumulative exponential decay simple reward
    """
    return SIMPLE_ALLOC * (1 - np.exp(-LAMBDA * epoch))


def network_time(cum_capped_power):
    """ """
    return np.log1p(BASELINE_R * cum_capped_power / BASELINE_B0) / BASELINE_R


def cum_baseline_reward_epoch(network_time):
    """ """
    return BASELINE_ALLOC * (1 - np.exp(-LAMBDA * network_time))
