"""Phase 3: Vectorized Feature Engine for Possession Classification.

Team: Visca Barca
Competition: More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)

Features Computed Across All 199,319 Frames (float32):
1. Player-to-Ball Proximity Signals:
   - Closest 3 Home & Away player distances: d_H1, d_H2, d_H3, d_A1, d_A2, d_A3
   - Closest player speeds: s_H1, s_A1
   - Symmetric proximity difference: diff_d1, norm_diff_d1
   - Exponential soft control: soft_ctrl_H, soft_ctrl_A, soft_ctrl_diff
   - Local team density counts: count_H_3m, count_A_3m, count_H_7m, count_A_7m
   - Endpoint gap proximity: last_endpoint_diff, next_endpoint_diff, endpoint_diff_d1
2. Dead-Ball (D) vs. Live-Play Discriminators:
   - min_player_dist: distance of closest player to the ball
   - ball_oob_dist: signed distance of ball inside the 105.3 x 68.0m pitch
   - player_speed_mean: collective mean player tempo
   - player_speed_p90: 90th percentile player speed
   - is_ball_stationary: binary indicator for visible ball with speed < 1.0 m/s
   - pre_2nd_half_mask: guaranteed dead-ball mask for frames 110808..113741
3. Smart Tagging Tactical Priors:
   - st_team_prior: empirical polarity prior (+1 for Home attack, -1 for Away attack;
     inverted -0.7/+0.7 for DEFENDING tags)
   - st_set_piece_active: binary indicator for active SET PIECES interval
   - st_set_piece_team: +1 for Home, -1 for Away, 0 for N/A
   - frames_to_set_piece_start, frames_to_set_piece_end
4. Multi-Scale Temporal Smoothing:
   - Rolling uniform filters (25, 75, 150 frames) for norm_diff_d1, soft_ctrl_diff,
     player_speed_mean, ball_speed, ball_vis.
5. High-Speed Caching:
   - Serialized to `cache/features.npz`.
"""

import os
import sys
import numpy as np
from scipy.ndimage import uniform_filter1d

# Path resolution
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import (
    N_VIDEO_FRAMES,
    PITCH_LENGTH,
    PITCH_WIDTH,
    DEFAULT_CACHE_PATH,
    get_or_load_tactical_data,
)
from src.ball_recovery import (
    RECOVERED_CACHE_PATH,
    get_or_build_recovered_ball_data,
)

FEATURES_CACHE_PATH = os.path.join(ROOT_DIR, "cache", "features.npz")

NON_DEF_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "CREATING CHANCES",
    "PLAYERS IN THE BOX",
    "ATTACKING TRANSITION",
    "LONG BALL",
    "GOALS",
}

DEF_CODES = {
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENSIVE TRANSITION",
}


def compute_proximity_features(home_players: np.ndarray,
                               away_players: np.ndarray,
                               ball_pos_rec: np.ndarray,
                               last_obs_pos: np.ndarray,
                               next_obs_pos: np.ndarray,
                               ball_vis_clean: np.ndarray):
    """Vectorized calculation of player proximity to reconstructed and endpoint ball positions."""
    n_frames = len(ball_pos_rec)

    # 1. Distances to ball_pos_rec
    pos_H = home_players[:, :, :2]  # (N, 24, 2)
    diff_H = pos_H - ball_pos_rec[:, None, :]
    dist_H = np.hypot(diff_H[:, :, 0], diff_H[:, :, 1])
    dist_H = np.where(np.isnan(dist_H), 999.0, dist_H)

    pos_A = away_players[:, :, :2]  # (N, 22, 2)
    diff_A = pos_A - ball_pos_rec[:, None, :]
    dist_A = np.hypot(diff_A[:, :, 0], diff_A[:, :, 1])
    dist_A = np.where(np.isnan(dist_A), 999.0, dist_A)

    # Sorted distances
    sort_H = np.sort(dist_H, axis=1)
    d_H1 = sort_H[:, 0].astype(np.float32)
    d_H2 = sort_H[:, 1].astype(np.float32)
    d_H3 = sort_H[:, 2].astype(np.float32)

    sort_A = np.sort(dist_A, axis=1)
    d_A1 = sort_A[:, 0].astype(np.float32)
    d_A2 = sort_A[:, 1].astype(np.float32)
    d_A3 = sort_A[:, 2].astype(np.float32)

    # Speeds of closest players
    spd_H = np.where(np.isnan(home_players[:, :, 2]), 0.0, home_players[:, :, 2])
    arg_H1 = np.argmin(dist_H, axis=1)
    s_H1 = np.take_along_axis(spd_H, arg_H1[:, None], axis=1)[:, 0].astype(np.float32)

    spd_A = np.where(np.isnan(away_players[:, :, 2]), 0.0, away_players[:, :, 2])
    arg_A1 = np.argmin(dist_A, axis=1)
    s_A1 = np.take_along_axis(spd_A, arg_A1[:, None], axis=1)[:, 0].astype(np.float32)

    # Proximity diffs
    diff_d1 = (d_A1 - d_H1).astype(np.float32)
    norm_diff_d1 = ((d_A1 - d_H1) / (d_A1 + d_H1 + 0.5)).astype(np.float32)

    # Exponential soft control
    soft_ctrl_H = (np.exp(-d_H1 / 1.8) + 0.5 * np.exp(-d_H2 / 3.0)).astype(np.float32)
    soft_ctrl_A = (np.exp(-d_A1 / 1.8) + 0.5 * np.exp(-d_A2 / 3.0)).astype(np.float32)
    soft_ctrl_diff = (soft_ctrl_H - soft_ctrl_A).astype(np.float32)

    # Player density counts
    count_H_3m = np.sum(dist_H <= 3.0, axis=1).astype(np.float32)
    count_A_3m = np.sum(dist_A <= 3.0, axis=1).astype(np.float32)
    count_H_7m = np.sum(dist_H <= 7.0, axis=1).astype(np.float32)
    count_A_7m = np.sum(dist_A <= 7.0, axis=1).astype(np.float32)

    # Endpoint proximity signals
    last_diff_H = np.hypot(pos_H[:, :, 0] - last_obs_pos[:, None, 0],
                           pos_H[:, :, 1] - last_obs_pos[:, None, 1])
    last_dH1 = np.nanmin(np.where(np.isnan(last_diff_H), 999.0, last_diff_H), axis=1)

    last_diff_A = np.hypot(pos_A[:, :, 0] - last_obs_pos[:, None, 0],
                           pos_A[:, :, 1] - last_obs_pos[:, None, 1])
    last_dA1 = np.nanmin(np.where(np.isnan(last_diff_A), 999.0, last_diff_A), axis=1)
    last_endpoint_diff = (last_dA1 - last_dH1).astype(np.float32)

    next_diff_H = np.hypot(pos_H[:, :, 0] - next_obs_pos[:, None, 0],
                           pos_H[:, :, 1] - next_obs_pos[:, None, 1])
    next_dH1 = np.nanmin(np.where(np.isnan(next_diff_H), 999.0, next_diff_H), axis=1)

    next_diff_A = np.hypot(pos_A[:, :, 0] - next_obs_pos[:, None, 0],
                           pos_A[:, :, 1] - next_obs_pos[:, None, 1])
    next_dA1 = np.nanmin(np.where(np.isnan(next_diff_A), 999.0, next_diff_A), axis=1)
    next_endpoint_diff = (next_dA1 - next_dH1).astype(np.float32)

    endpoint_diff_d1 = np.where(ball_vis_clean, diff_d1, last_endpoint_diff).astype(np.float32)

    return {
        "d_H1": d_H1, "d_H2": d_H2, "d_H3": d_H3,
        "d_A1": d_A1, "d_A2": d_A2, "d_A3": d_A3,
        "s_H1": s_H1, "s_A1": s_A1,
        "diff_d1": diff_d1,
        "norm_diff_d1": norm_diff_d1,
        "soft_ctrl_H": soft_ctrl_H,
        "soft_ctrl_A": soft_ctrl_A,
        "soft_ctrl_diff": soft_ctrl_diff,
        "count_H_3m": count_H_3m,
        "count_A_3m": count_A_3m,
        "count_H_7m": count_H_7m,
        "count_A_7m": count_A_7m,
        "last_endpoint_diff": last_endpoint_diff,
        "next_endpoint_diff": next_endpoint_diff,
        "endpoint_diff_d1": endpoint_diff_d1,
    }


def compute_dead_ball_features(ball_pos_rec: np.ndarray,
                               d_H1: np.ndarray,
                               d_A1: np.ndarray,
                               home_players: np.ndarray,
                               away_players: np.ndarray,
                               ball_vis_clean: np.ndarray,
                               ball_speed: np.ndarray):
    """Compute features that discriminate dead-ball (D) stretches from live play."""
    n_frames = len(ball_pos_rec)

    # Minimum player distance
    min_player_dist = np.minimum(d_H1, d_A1).astype(np.float32)

    # Signed distance inside pitch boundary 105.3 x 68.0m (negative if out-of-bounds)
    dist_x = np.minimum(ball_pos_rec[:, 0], PITCH_LENGTH - ball_pos_rec[:, 0])
    dist_y = np.minimum(ball_pos_rec[:, 1], PITCH_WIDTH - ball_pos_rec[:, 1])
    ball_oob_dist = np.minimum(dist_x, dist_y).astype(np.float32)

    # Collective player tempo
    all_spd = np.concatenate([home_players[:, :, 2], away_players[:, :, 2]], axis=1)
    valid = ~np.isnan(all_spd)
    n_valid = np.sum(valid, axis=1)
    spd_clean = np.where(valid, all_spd, 0.0)
    player_speed_mean = (np.sum(spd_clean, axis=1) / np.maximum(n_valid, 1)).astype(np.float32)

    # 90th percentile player speed
    sorted_spd = np.sort(np.where(valid, all_spd, -1.0), axis=1)
    p90_offset = np.floor(0.9 * np.maximum(n_valid - 1, 0)).astype(int)
    p90_idx = np.clip((46 - n_valid) + p90_offset, 0, 45)
    player_speed_p90 = np.take_along_axis(sorted_spd, p90_idx[:, None], axis=1)[:, 0].astype(np.float32)
    player_speed_p90 = np.where(n_valid > 0, player_speed_p90, 0.0).astype(np.float32)

    # Stationary ball indicator (ball visible and almost motionless)
    is_ball_stationary = (ball_vis_clean & (ball_speed < 1.0)).astype(np.float32)

    # Guaranteed dead-ball mask before 2nd half kickoff (110808..113741)
    frames = np.arange(n_frames)
    pre_2nd_half_mask = ((frames >= 110808) & (frames <= 113741)).astype(np.float32)

    return {
        "min_player_dist": min_player_dist,
        "ball_oob_dist": ball_oob_dist,
        "player_speed_mean": player_speed_mean,
        "player_speed_p90": player_speed_p90,
        "is_ball_stationary": is_ball_stationary,
        "pre_2nd_half_mask": pre_2nd_half_mask,
    }


def compute_smart_tagging_features(smart_tagging: np.ndarray,
                                   n_frames: int = N_VIDEO_FRAMES):
    """Per-frame alignment of Smart Tagging tactical priors and set piece channels."""
    st_team_prior = np.zeros(n_frames, dtype=np.float32)
    st_set_piece_active = np.zeros(n_frames, dtype=np.float32)
    st_set_piece_team = np.zeros(n_frames, dtype=np.float32)

    sp_starts = []
    sp_ends = []

    for row in smart_tagging:
        code = row["code"]
        sf = int(row["start_frame"])
        ef = int(row["end_frame"])
        tm = row["team"]

        if code in NON_DEF_CODES:
            val = 1.0 if tm == "H" else (-1.0 if tm == "A" else 0.0)
            st_team_prior[sf:ef] += val
        elif code in DEF_CODES:
            # DEFENDING tag names the defending team -> opposing team in possession!
            val = -0.7 if tm == "H" else (0.7 if tm == "A" else 0.0)
            st_team_prior[sf:ef] += val
        elif code == "SET PIECES":
            sp_starts.append(sf)
            sp_ends.append(ef)
            st_set_piece_active[sf:ef] = 1.0
            st_set_piece_team[sf:ef] = 1.0 if tm == "H" else (-1.0 if tm == "A" else 0.0)

    st_team_prior = np.clip(st_team_prior, -1.0, 1.0).astype(np.float32)

    # Frame distances to nearest set piece start and end
    frames_to_set_piece_start = np.full(n_frames, 99999.0, dtype=np.float32)
    frames_to_set_piece_end = np.full(n_frames, 99999.0, dtype=np.float32)

    if sp_starts:
        sp_starts_arr = np.array(sp_starts)
        sp_ends_arr = np.array(sp_ends)
        all_frames = np.arange(n_frames)

        idx_s = np.searchsorted(sp_starts_arr, all_frames)
        idx_s0 = np.clip(idx_s - 1, 0, len(sp_starts_arr) - 1)
        idx_s1 = np.clip(idx_s, 0, len(sp_starts_arr) - 1)
        d_s0 = all_frames - sp_starts_arr[idx_s0]
        d_s1 = all_frames - sp_starts_arr[idx_s1]
        use_s0 = np.abs(d_s0) <= np.abs(d_s1)
        frames_to_set_piece_start = np.where(use_s0, d_s0, d_s1).astype(np.float32)

        idx_e = np.searchsorted(sp_ends_arr, all_frames)
        idx_e0 = np.clip(idx_e - 1, 0, len(sp_ends_arr) - 1)
        idx_e1 = np.clip(idx_e, 0, len(sp_ends_arr) - 1)
        d_e0 = all_frames - sp_ends_arr[idx_e0]
        d_e1 = all_frames - sp_ends_arr[idx_e1]
        use_e0 = np.abs(d_e0) <= np.abs(d_e1)
        frames_to_set_piece_end = np.where(use_e0, d_e0, d_e1).astype(np.float32)

    return {
        "st_team_prior": st_team_prior,
        "st_set_piece_active": st_set_piece_active,
        "st_set_piece_team": st_set_piece_team,
        "frames_to_set_piece_start": frames_to_set_piece_start,
        "frames_to_set_piece_end": frames_to_set_piece_end,
    }


def compute_smoothed_features(base_features: dict,
                              windows: list = [25, 75, 150]):
    """Apply rolling uniform filters over multiple temporal scales."""
    smoothed = {}
    target_keys = [
        "norm_diff_d1",
        "soft_ctrl_diff",
        "player_speed_mean",
        "ball_speed",
        "ball_vis_clean",
    ]

    for key in target_keys:
        arr = base_features[key].astype(np.float32)
        for w in windows:
            out_key = f"{key}_smooth_{w}"
            smoothed[out_key] = uniform_filter1d(arr, size=w, mode="nearest").astype(np.float32)

    return smoothed


def build_and_cache_features(cache_path: str = FEATURES_CACHE_PATH,
                             tactical_cache_path: str = DEFAULT_CACHE_PATH,
                             recovered_cache_path: str = RECOVERED_CACHE_PATH,
                             force_reload: bool = False):
    """Execute complete vectorized feature pipeline and cache result to .npz file."""
    if os.path.isfile(cache_path) and not force_reload:
        return load_features_data(cache_path)

    print("Loading base tactical and recovered ball data...")
    tactical = get_or_load_tactical_data(tactical_cache_path)
    recovered = get_or_build_recovered_ball_data(recovered_cache_path)

    home_players = tactical["home_players"]
    away_players = tactical["away_players"]
    smart_tagging = tactical["smart_tagging"]

    ball_pos_rec = recovered["ball_pos_rec"]
    ball_vis_clean = recovered["ball_vis_clean"]
    ball_speed = recovered["ball_speed"]
    ball_accel = recovered["ball_accel"]
    ball_regime = recovered["ball_regime"]
    ball_conf = recovered["ball_conf"]
    last_obs_pos = recovered["last_obs_pos"]
    next_obs_pos = recovered["next_obs_pos"]
    frames_since_obs = recovered["frames_since_obs"]
    frames_until_obs = recovered["frames_until_obs"]
    current_gap_len = recovered["current_gap_len"]
    gap_exit_oob = recovered["gap_exit_oob"]

    print("1/4 Computing player-to-ball proximity & control signals...")
    prox = compute_proximity_features(
        home_players, away_players, ball_pos_rec,
        last_obs_pos, next_obs_pos, ball_vis_clean
    )

    print("2/4 Computing dead-ball vs live-play discriminators...")
    dead = compute_dead_ball_features(
        ball_pos_rec, prox["d_H1"], prox["d_A1"],
        home_players, away_players, ball_vis_clean, ball_speed
    )

    print("3/4 Computing Smart Tagging tactical prior channels...")
    tactical_priors = compute_smart_tagging_features(smart_tagging)

    print("4/4 Computing multi-scale temporal rolling features...")
    base_dict = {
        **prox,
        **dead,
        **tactical_priors,
        "ball_speed": ball_speed.astype(np.float32),
        "ball_accel": ball_accel.astype(np.float32),
        "ball_regime": ball_regime.astype(np.float32),
        "ball_conf": ball_conf.astype(np.float32),
        "ball_vis_clean": ball_vis_clean.astype(np.float32),
        "frames_since_obs": frames_since_obs.astype(np.float32),
        "frames_until_obs": frames_until_obs.astype(np.float32),
        "current_gap_len": current_gap_len.astype(np.float32),
        "gap_exit_oob": gap_exit_oob.astype(np.float32),
    }

    smoothed = compute_smoothed_features(base_dict, windows=[25, 75, 150])
    all_features = {**base_dict, **smoothed}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **all_features)
    print(f"Saved {len(all_features)} feature channels to {os.path.relpath(cache_path, ROOT_DIR)}.")

    return all_features


def load_features_data(cache_path: str = FEATURES_CACHE_PATH):
    """Load cached features dataset in < 1.0s."""
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Features cache not found at {cache_path}")

    with np.load(cache_path) as data:
        return {k: data[k] for k in data.files}


def get_or_build_features(cache_path: str = FEATURES_CACHE_PATH, force_reload: bool = False):
    """Load or generate features dataset."""
    if os.path.isfile(cache_path) and not force_reload:
        return load_features_data(cache_path)
    return build_and_cache_features(cache_path=cache_path, force_reload=force_reload)


if __name__ == "__main__":
    import time
    t0 = time.time()
    feats = build_and_cache_features(force_reload=True)
    t1 = time.time()
    print(f"Features built and cached in {t1 - t0:.2f}s ({len(feats)} channels).")
