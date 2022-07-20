import numpy as np
import pandas as pd

SAFT_AMOUNT = 200_000_000.0  # some one pls check these
PL_AMOUNT = 300_000_000.0  # some one pls check these
FOUND_AMOUNT = 100_000_000.0


def compute_vesting_trajectory_df(start_day: int, end_day: int):
    vesting_trajectory_dict = {
        "six_month_vest_saft": vest(SAFT_AMOUNT, 0.22, 366 // 2, end_day),
        "one_year_vest_saft": vest(SAFT_AMOUNT, 0.15, 365 * 1, end_day),
        "two_year_vest_saft": vest(SAFT_AMOUNT, 0.05, 365 * 2, end_day),
        "three_year_vest_saft": vest(SAFT_AMOUNT, 0.58, 365 * 3, end_day),
        "six_year_vest_pl": vest(PL_AMOUNT, 1, 365 * 6, end_day),
        "six_year_vest_found": vest(FOUND_AMOUNT, 1, 365 * 6, end_day),
        "total_vest0": (
            vest(SAFT_AMOUNT, 0.22, 366 // 2, end_day)
            + vest(SAFT_AMOUNT, 0.15, 365 * 1, end_day)
            + vest(SAFT_AMOUNT, 0.05, 365 * 2, end_day)
            + vest(SAFT_AMOUNT, 0.58, 365 * 3, end_day)
            + vest(PL_AMOUNT, 1, 365 * 6, end_day)
            + vest(FOUND_AMOUNT, 1, 365 * 6, end_day)
        ),
    }

    offset = 51
    init_amt = 10_632_000
    vesting_trajectory_dict["total_vest"] = np.pad(
        vesting_trajectory_dict["total_vest0"] + init_amt, (offset, 0)
    )

    vest_df = pd.DataFrame(
        {
            "days": np.arange(start_day, end_day),
            "total_vest": vesting_trajectory_dict["total_vest"][start_day:end_day],
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
