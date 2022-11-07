import numpy as np
import pandas as pd
import datetime

FIL_BASE = 2_000_000_000.0
INIT_AMT = 10_632_000 # added post the calico upgrade
# 0.0896928 = (19015887 + 32787700 + 9400000 + 22421712 + 7223364 + 87637883 + 898958)/2000000000.
FUNDRASING_AMOUNT = 0.0896928 * FIL_BASE
PL_AMOUNT = 0.15 * FIL_BASE
FOUNDATION_AMOUNT = 0.05 * FIL_BASE
# 0.00490253 = 9805053/2000000000.
UNKNOWN_6YR_AMOUNT = 0.00490253 * FIL_BASE
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

    #################################################
    Lotus Info 
    https://github.com/filecoin-project/lotus/blob/e65fae28de2a44947dd24af8c7dafcade29af1a4/chain/stmgr/supply.go#L341 
    
    Fraction of base
    0.20 : 6yr
    0.0442684 = (87637883 + 898958)/2000000000 : 3yr
    0.00361168 = 7223364/2000000000 : 2yr
    0.0159109 = (9400000 + 22421712)/2000000000 : 1yr
    0.0259018 = (19015887 + 32787700)/2000000000 : 6mo
    0.005316 = 10632000/2000000000 : 0mo

    SAFT as fraction of SAFT total
    (Unkown if INIT_AMT is SAFT or other, but here treated as not) 
    179385504 = 19015887 + 32787700 + 9400000 + 22421712 + 7223364 + 87637883 + 898958
    0.288784 = (19015887 + 32787700)/179385504. : 6mo
    0.177393 = (9400000 + 22421712)/179385504. : 1yr
    0.0402673 = 7223364/179385504. : 2yr    
    0.493556 = (87637883 + 898958)/179385504. : 3yr

    """
    # we assume vesting started at main net launch, in 2020-10-15
    start_day = (start_date - datetime.date(2020, 10, 15)).days
    end_day = (end_date - datetime.date(2020, 10, 15)).days
#    saft_adj = FUNDRASING_AMOUNT - init_amt # check int_amt was fundraising
#    also not sure it should be removed ^

    vesting_df = pd.DataFrame(
        {
            "six_month_vest_saft": vest(saft, 0.288784, 183, end_day),
            "one_year_vest_saft": vest(saft, 0.177393, 365 * 1, end_day),
            "two_year_vest_saft": vest(saft, 0.0402673, 365 * 2, end_day),
            "three_year_vest_saft": vest(saft, 0.493556, 365 * 3, end_day),
            "six_year_vest_pl": vest(PL_AMOUNT, 1, 365 * 6, end_day),
            "six_year_vest_foundation": vest(FOUNDATION_AMOUNT, 1, 365 * 6, end_day),
            "six_year_vest_unknown": vest(UNKNOWN_6YR_AMOUNT, 1, 365 * 6, end_day),
        }
    )
    vesting_df["total_vest"] = vesting_df.sum(axis=1) + INIT_AMT
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
