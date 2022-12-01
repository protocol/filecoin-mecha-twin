import numpy as np
import pandas as pd
import datetime

from .data import get_vested_amount

FIL_BASE = 2_000_000_000.0
PL_AMOUNT = 0.15 * FIL_BASE
FOUNDATION_AMOUNT = 0.05 * FIL_BASE
STORAGE_MINING = 0.55 * FIL_BASE
MINING_RESERVE = 0.15 * FIL_BASE


def compute_vesting_trajectory_df(
    start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    """
    15% to PL -> 6-year linear vesting
    5% to FIlecoin foundation -> 6-year linear vesting
    10% to Investors -> Linear vesting with different durations (taken from lotus):
        - 0 days: 10_632_000
        - 6 months: 19_015_887 + 32_787_700
        - 1 yrs: 22_421_712 + 9_400_000
        - 2 yrs: 7_223_364
        - 3 yrs: 87_637_883 + 898_958
        - 6 yrs: 9_805_053
        (total of 199_822_557)

    Info taken from:
        - https://coinlist.co/assets/index/filecoin_2017_index/Filecoin-Sale-Economics-e3f703f8cd5f644aecd7ae3860ce932064ce014dd60de115d67ff1e9047ffa8e.pdf
        - https://spec.filecoin.io/#section-systems.filecoin_token.token_allocation
        - https://filecoin.io/blog/filecoin-circulating-supply/
        - https://github.com/filecoin-project/lotus/blob/e65fae28de2a44947dd24af8c7dafcade29af1a4/chain/stmgr/supply.go#L148
    """
    # we assume vesting started at main net launch, in 2020-10-15
    launch_date = datetime.date(2020, 10, 15)
    end_day = (end_date - launch_date).days
    # Get entire daily vesting trajectory
    full_vest_df = pd.DataFrame(
        {
            "date": pd.date_range(launch_date, end_date, freq="d")[:-1],
            "six_month_vest_saft": vest(19_015_887 + 32_787_700, 183, end_day),
            "one_year_vest_saft": vest(22_421_712 + 9_400_000, 365 * 1, end_day),
            "two_year_vest_saft": vest(7_223_364, 365 * 2, end_day),
            "three_year_vest_saft": vest(87_637_883 + 898_958, 365 * 3, end_day),
            "six_year_vest_saft": vest(9_805_053, 365 * 6, end_day),
            "six_year_vest_pl": vest(PL_AMOUNT, 365 * 6, end_day),
            "six_year_vest_foundation": vest(FOUNDATION_AMOUNT, 365 * 6, end_day),
        }
    )
    full_vest_df["date"] = full_vest_df["date"].dt.date
    # Filter vesting trajectory for desired dates
    vest_df = full_vest_df[full_vest_df["date"] >= start_date]
    # Compute total cumulative vesting
    vest_df.loc[:, "total_day_vest"] = vest_df.drop(columns=["date"]).sum(axis=1)
    start_vested_amt = get_vested_amount(start_date)
    vest_df.loc[:, "total_vest"] = vest_df["total_day_vest"].cumsum() + start_vested_amt
    vest_df = vest_df[["date", "total_vest"]]
    return vest_df


def vest(amount: float, time: int, end_day: int) -> np.array:
    """
    amount -- total amount e.g 300M FIL for SAFT
    time -- vesting time in days
    end_day -- end day for the vesting trajectory
    """
    ones_ = np.ones(int(time))[:end_day]
    extra_to_pad_ = max(0, end_day - int(time))
    ones_padded_ = np.pad(ones_, (0, extra_to_pad_))
    vest_ = ones_padded_ / time
    return amount * vest_
