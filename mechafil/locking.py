import numpy as np

GIB = 2**30

# Block reward collateral
def compute_day_locked_rewards(day_network_reward: float) -> float:
    return 0.75 * day_network_reward


def compute_day_reward_release(prev_network_locked_reward: float) -> float:
    return prev_network_locked_reward / 180.0


# Initial pledge collateral
def compute_day_delta_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_onboarded_qa_power: float,
    day_renewed_qa_power: float,
    total_qa_power: float,
    baseline_power: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float = 0.3,
) -> float:
    onboards_delta = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_onboarded_qa_power,
        total_qa_power,
        baseline_power,
        lock_target,
    )
    renews_delta = compute_renewals_delta_pledge(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power,
        total_qa_power,
        baseline_power,
        renewal_rate,
        scheduled_pledge_release,
        lock_target,
    )
    return onboards_delta + renews_delta


def compute_day_locked_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_onboarded_qa_power: float,
    day_renewed_qa_power: float,
    total_qa_power: float,
    baseline_power: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float = 0.3,
) -> float:
    # Total locked from new onboards
    onboards_locked = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_onboarded_qa_power,
        total_qa_power,
        baseline_power,
        lock_target,
    )
    # Total locked from renewals
    original_pledge = renewal_rate * scheduled_pledge_release
    new_pledge = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power,
        total_qa_power,
        baseline_power,
        lock_target,
    )
    renews_locked = max(original_pledge, new_pledge)
    # Total locked pledge
    locked = onboards_locked + renews_locked
    return locked


def compute_renewals_delta_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_renewed_qa_power: float,
    total_qa_power: float,
    baseline_power: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float,
) -> float:
    # Delta from sectors expiring
    expire_delta = -(1 - renewal_rate) * scheduled_pledge_release
    # Delta from sector renewing
    original_pledge = renewal_rate * scheduled_pledge_release
    new_pledge = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power,
        total_qa_power,
        baseline_power,
        lock_target,
    )
    renew_delta = max(0.0, new_pledge - original_pledge)
    # Delta for all scheduled sectors
    delta = expire_delta + renew_delta
    return delta


def compute_new_pledge_for_added_power(
    day_network_reward: float,
    prev_circ_supply: float,
    day_added_qa_power: float,
    total_qa_power: float,
    baseline_power: float,
    lock_target: float,
) -> float:
    # storage collateral
    storage_pledge = 20.0 * day_network_reward * (day_added_qa_power / total_qa_power)
    # consensus collateral
    normalized_qap_growth = day_added_qa_power / max(total_qa_power, baseline_power)
    consensus_pledge = lock_target * prev_circ_supply * normalized_qap_growth
    # total added pledge
    added_pledge = storage_pledge + consensus_pledge

    pledge_cap = day_added_qa_power * 1.0 / GIB  # The # of bytes in a GiB (Gibibyte)
    return min(pledge_cap, added_pledge)


def get_day_schedule_pledge_release(
    day_i,
    current_day_i,
    day_pledge_locked_vec: np.array,
    known_scheduled_pledge_release_vec: np.array,
    duration: int,
) -> float:
    # scheduled releases coming from known active sectors
    if day_i > len(known_scheduled_pledge_release_vec) - 1:
        known_day_release = 0.0
    else:
        known_day_release = known_scheduled_pledge_release_vec[day_i]
    # schedule releases coming from modeled sectors
    if day_i - duration >= current_day_i:
        model_day_release = day_pledge_locked_vec[day_i - duration]
    else:
        model_day_release = 0.0
    # Total pledge schedule releases
    day_pledge_schedules_release = known_day_release + model_day_release
    return day_pledge_schedules_release
