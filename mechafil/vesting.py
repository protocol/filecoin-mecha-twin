import numpy as np
import pandas as pd
import datetime

FIL_BASE = 2_000_000_000.0
FUNDRASING_AMOUNT = 0.1 * FIL_BASE
PL_AMOUNT = 0.15 * FIL_BASE
FOUNDATION_AMOUNT = 0.05 * FIL_BASE
STORAGE_MINING = 0.55 * FIL_BASE
MINING_RESERVE = 0.15 * FIL_BASE


def compute_vesting_trajectory_df(start_date: datetime.date, end_date: datetime.date):
    """
    15% to PL -> 6-year linear vesting
    5% to FIlecoin foundation -> 6-year linear vesting
    10% to Investors -> 6-month to 3-year linear vesting:
        - 58% of SAFT tokens vest linearly over 3 Years
        - 5% of SAFTs tokens vest linearly over 2 Years
        - 15% of SAFTs tokens vest linearly over 1 Years
        - 22% of SAFTs tokens vest linearly over 6 Months

    Info taken from:
        - https://coinlist.co/assets/index/filecoin_2017_index/Filecoin-Sale-Economics-e3f703f8cd5f644aecd7ae3860ce932064ce014dd60de115d67ff1e9047ffa8e.pdf
        - https://spec.filecoin.io/#section-systems.filecoin_token.token_allocation
        - https://filecoin.io/blog/filecoin-circulating-supply/
    """
    # we assume vesting started at main net launch, in 2020-10-15
    start_day = (start_date - datetime.date(2020, 10, 15)).days
    end_day = (end_date - datetime.date(2020, 10, 15)).days
    # TODO: understand where this adjustment is coming from!
    init_amt = 10_632_000
    saft_adj = FUNDRASING_AMOUNT - init_amt
    vesting_df = pd.DataFrame(
        {
            "six_month_vest_saft": vest(saft_adj, 0.22, 183, end_day),
            "one_year_vest_saft": vest(saft_adj, 0.15, 365 * 1, end_day),
            "two_year_vest_saft": vest(saft_adj, 0.05, 365 * 2, end_day),
            "three_year_vest_saft": vest(saft_adj, 0.58, 365 * 3, end_day),
            "six_year_vest_pl": vest(PL_AMOUNT, 1, 365 * 6, end_day),
            "six_year_vest_foundation": vest(FOUNDATION_AMOUNT, 1, 365 * 6, end_day),
        }
    )
    vesting_df["total_vest"] = vesting_df.sum(axis=1) + init_amt
    vest_df = pd.DataFrame(
        {
            "date": pd.date_range(start_date, end_date, freq="d")[:-1],
            "total_vest": vesting_df["total_vest"][start_day:end_day],
        }
    )
    return vest_df


def vest(amount: float, fraction: float, time: int, end_day: int) -> np.array:
    """
    amount -- total amount e.g 300M FIL for SAFT
    fraction -- fraction vested
    time -- days
    """
    ones_ = np.ones(int(time))[:end_day]
    extra_to_pad_ = max(0, end_day - int(time))
    ones_padded_ = np.pad(ones_, (0, extra_to_pad_))
    vest_ = np.cumsum(ones_padded_ / time)
    return amount * fraction * vest_
