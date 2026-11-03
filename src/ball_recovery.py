"""Phase 2 & 3: Ball Recovery, Trajectory Reconstruction & Kinematics Engine.

Team: Visca Barca
Competition: More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)

Key Capabilities:
1. Glitch & Tracker Discontinuity Handling:
   - Eliminates derivative spikes (> 42.0 m/s instantaneous steps split into independent sub-segments).
   - Speed clamped to [0.0, 40.0] m/s, acceleration clamped to [-50.0, 50.0] m/s^2.
2. Gap Regimes:
   - Regime 0 (OBSERVED): Clean raw coordinates, confidence = 1.0.
   - Regime 1 (MICRO_GAP): Missing gaps 1..30 frames (<= 1.2s), linear interpolation.
   - Regime 2 (MEDIUM_GAP): Missing gaps 31..300 frames (1.2s..12.0s), linear interpolation.
   - Regime 3 (LONG_GAP): Missing gaps > 300 frames (> 12.0s) / pre-kickoff / ungraded stretches,
     linear endpoint fill.
3. Gap Context & Boundary Signals:
   - `frames_since_obs`, `frames_until_obs`, `current_gap_len`, `last_obs_pos`, `next_obs_pos`, `gap_exit_oob`.
4. High-Performance Caching:
   - Serialized to `cache/ball_recovered.npz`.
"""

import os
import sys
import numpy as np

# Path resolution
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import (
    N_VIDEO_FRAMES,
    FPS,
    PITCH_LENGTH,
    PITCH_WIDTH,
    GRADED_BLOCKS,
    TOTAL_GRADED_FRAMES,
    DEFAULT_CACHE_PATH,
    get_or_load_tactical_data,
    load_public_ground_truth,
)

DT = 1.0 / FPS  # 0.04s
PITCH_CENTER = np.array([PITCH_LENGTH / 2.0, PITCH_WIDTH / 2.0], dtype=np.float32)  # (52.65, 34.0)
RECOVERED_CACHE_PATH = os.path.join(ROOT_DIR, "cache", "ball_recovered.npz")

LIVE_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
    "DEFENSIVE TRANSITION",
    "LONG BALL",
}


def filter_ball_glitches(ball_pos: np.ndarray, ball_vis: np.ndarray,
                         max_speed: float = 45.0,
                         snap_back_ratio: float = 0.5):
    """Detect and scrub single- or multi-frame teleportation spikes and isolated islands.

    Args:
        ball_pos: np.ndarray (N, 2)
        ball_vis: np.ndarray (N,) bool
        max_speed: speed threshold in m/s (45.0 m/s = 1.8 m per frame at 25 fps)
        snap_back_ratio: max ratio between skip distance and jump distance to trigger snap-back

    Returns:
        ball_pos_clean: np.ndarray (N, 2)
        ball_vis_clean: np.ndarray (N,) bool
        n_scrubbed: int count of scrubbed frames
    """
    ball_pos_clean = ball_pos.copy()
    ball_vis_clean = ball_vis.copy()
    n_frames = len(ball_vis)
    scrubbed_indices = []

    # 1. Single-frame spike check: t-1, t, t+1 are visible
    for t in range(1, n_frames - 1):
        if ball_vis_clean[t - 1] and ball_vis_clean[t] and ball_vis_clean[t + 1]:
            d1 = np.hypot(ball_pos_clean[t, 0] - ball_pos_clean[t - 1, 0],
                          ball_pos_clean[t, 1] - ball_pos_clean[t - 1, 1])
            d2 = np.hypot(ball_pos_clean[t + 1, 0] - ball_pos_clean[t, 0],
                          ball_pos_clean[t + 1, 1] - ball_pos_clean[t, 1])
            d_skip = np.hypot(ball_pos_clean[t + 1, 0] - ball_pos_clean[t - 1, 0],
                              ball_pos_clean[t + 1, 1] - ball_pos_clean[t - 1, 1])
            spd1 = d1 / DT
            spd2 = d2 / DT
            if (spd1 > max_speed or spd2 > max_speed) and d_skip < snap_back_ratio * min(d1, d2):
                scrubbed_indices.append(t)

    # 2. Isolated visible islands of <= 3 frames surrounded by NaN gaps with jump > 15m
    vis_runs = []
    in_run = False
    start = None
    for f in range(n_frames):
        if ball_vis_clean[f] and not in_run:
            in_run = True
            start = f
        elif not ball_vis_clean[f] and in_run:
            in_run = False
            vis_runs.append((start, f - 1))
    if in_run:
        vis_runs.append((start, n_frames - 1))

    for idx, (rs, re) in enumerate(vis_runs):
        run_len = re - rs + 1
        if run_len <= 3:
            prev_r = vis_runs[idx - 1] if idx > 0 else None
            next_r = vis_runs[idx + 1] if idx < len(vis_runs) - 1 else None
            d_prev = (np.hypot(ball_pos_clean[rs, 0] - ball_pos_clean[prev_r[1], 0],
                               ball_pos_clean[rs, 1] - ball_pos_clean[prev_r[1], 1])
                      if prev_r else 0.0)
            d_next = (np.hypot(ball_pos_clean[next_r[0], 0] - ball_pos_clean[re, 0],
                               ball_pos_clean[next_r[0], 1] - ball_pos_clean[re, 1])
                      if next_r else 0.0)
            if (prev_r and d_prev > 15.0) and (next_r and d_next > 15.0):
                scrubbed_indices.extend(range(rs, re + 1))

    unique_scrubbed = sorted(list(set(scrubbed_indices)))
    for idx in unique_scrubbed:
        ball_pos_clean[idx] = np.nan
        ball_vis_clean[idx] = False

    return ball_pos_clean, ball_vis_clean, len(unique_scrubbed)


def compute_player_swarm_centroid(home_players: np.ndarray,
                                  away_players: np.ndarray,
                                  window: int = 25) -> np.ndarray:
    """Compute player action centroid using speed-squared weighting smoothed over window frames."""
    n_frames = len(home_players)
    pos = np.concatenate([home_players[:, :, :2], away_players[:, :, :2]], axis=1)
    spd = np.concatenate([home_players[:, :, 2], away_players[:, :, 2]], axis=1)

    valid = ~np.isnan(spd)
    spd_clean = np.where(valid, spd, 0.0)
    pos_clean = np.where(np.isnan(pos), 0.0, pos)

    weights = spd_clean**2 + 0.1 * valid
    sum_w = np.sum(weights, axis=1, keepdims=True)
    has_players = np.sum(valid, axis=1) > 0

    swarm_raw = np.zeros((n_frames, 2), dtype=np.float32)
    safe_sum = np.where(sum_w > 0, sum_w, 1.0)
    weighted_pos = np.sum(pos_clean * weights[:, :, None], axis=1) / safe_sum
    swarm_raw[has_players] = weighted_pos[has_players]
    swarm_raw[~has_players] = PITCH_CENTER

    kernel = np.ones(window, dtype=np.float32) / float(window)
    swarm_pos = np.zeros_like(swarm_raw)
    swarm_pos[:, 0] = np.convolve(swarm_raw[:, 0], kernel, mode="same")
    swarm_pos[:, 1] = np.convolve(swarm_raw[:, 1], kernel, mode="same")

    return swarm_pos


def compute_distance_to_observed(ball_vis_clean: np.ndarray,
                                 graded_blocks: list = GRADED_BLOCKS) -> np.ndarray:
    """Compute number of frames to nearest ball_vis_clean == True frame within the same half."""
    n_frames = len(ball_vis_clean)
    dist_to_obs = np.zeros(n_frames, dtype=np.int32)
    all_obs = np.where(ball_vis_clean)[0]

    for half, first, last in graded_blocks:
        half_obs = np.where(ball_vis_clean[first:last + 1])[0] + first
        if len(half_obs) == 0:
            half_obs = all_obs

        frames = np.arange(first, last + 1)
        idx = np.searchsorted(half_obs, frames)
        idx_left = np.clip(idx - 1, 0, len(half_obs) - 1)
        idx_right = np.clip(idx, 0, len(half_obs) - 1)
        d_left = np.abs(frames - half_obs[idx_left])
        d_right = np.abs(frames - half_obs[idx_right])
        dist_to_obs[first:last + 1] = np.minimum(d_left, d_right)

    ungraded_mask = np.ones(n_frames, dtype=bool)
    for _, first, last in graded_blocks:
        ungraded_mask[first:last + 1] = False

    ungraded_frames = np.where(ungraded_mask)[0]
    if len(ungraded_frames) and len(all_obs):
        idx = np.searchsorted(all_obs, ungraded_frames)
        idx_left = np.clip(idx - 1, 0, len(all_obs) - 1)
        idx_right = np.clip(idx, 0, len(all_obs) - 1)
        d_left = np.abs(ungraded_frames - all_obs[idx_left])
        d_right = np.abs(ungraded_frames - all_obs[idx_right])
        dist_to_obs[ungraded_frames] = np.minimum(d_left, d_right)

    dist_to_obs[ball_vis_clean] = 0
    return dist_to_obs


def compute_gap_context(ball_pos_clean: np.ndarray,
                        ball_vis_clean: np.ndarray,
                        graded_blocks: list = GRADED_BLOCKS):
    """Compute per-frame gap context signals: frames_since_obs, frames_until_obs,
    current_gap_len, last_obs_pos, next_obs_pos, gap_exit_oob.
    """
    n_frames = len(ball_vis_clean)
    frames_since_obs = np.zeros(n_frames, dtype=np.int32)
    frames_until_obs = np.zeros(n_frames, dtype=np.int32)
    last_obs_pos = np.zeros((n_frames, 2), dtype=np.float32)
    next_obs_pos = np.zeros((n_frames, 2), dtype=np.float32)
    current_gap_len = np.zeros(n_frames, dtype=np.int32)
    gap_exit_oob = np.zeros(n_frames, dtype=bool)

    # Initialize last_obs_pos & next_obs_pos to pitch center
    last_obs_pos[:] = PITCH_CENTER
    next_obs_pos[:] = PITCH_CENTER

    for half, first, last in graded_blocks:
        # Forward pass in half
        cur = 0
        p_last = PITCH_CENTER.copy()
        for t in range(first, last + 1):
            if ball_vis_clean[t]:
                cur = 0
                p_last = ball_pos_clean[t]
            else:
                cur += 1
            frames_since_obs[t] = cur
            last_obs_pos[t] = p_last

        # Backward pass in half
        cur = 0
        p_next = PITCH_CENTER.copy()
        for t in range(last, first - 1, -1):
            if ball_vis_clean[t]:
                cur = 0
                p_next = ball_pos_clean[t]
            else:
                cur += 1
            frames_until_obs[t] = cur
            next_obs_pos[t] = p_next

    # Forward fill across halftime and boundaries
    p_last = PITCH_CENTER.copy()
    for t in range(n_frames):
        if ball_vis_clean[t]:
            p_last = ball_pos_clean[t]
        last_obs_pos[t] = p_last

    p_next = PITCH_CENTER.copy()
    for t in range(n_frames - 1, -1, -1):
        if ball_vis_clean[t]:
            p_next = ball_pos_clean[t]
        next_obs_pos[t] = p_next

    # Process all missing intervals
    in_gap = False
    gs = None
    all_gaps = []
    for f in range(n_frames):
        if not ball_vis_clean[f] and not in_gap:
            in_gap = True
            gs = f
        elif ball_vis_clean[f] and in_gap:
            in_gap = False
            all_gaps.append((gs, f - 1))
    if in_gap:
        all_gaps.append((gs, n_frames - 1))

    for gs, ge in all_gaps:
        dur = ge - gs + 1
        current_gap_len[gs:ge + 1] = dur

        p_exit = last_obs_pos[gs]
        is_oob = (p_exit[0] < 1.5 or p_exit[0] > (PITCH_LENGTH - 1.5) or
                  p_exit[1] < 1.5 or p_exit[1] > (PITCH_WIDTH - 1.5))
        gap_exit_oob[gs:ge + 1] = is_oob

    return frames_since_obs, frames_until_obs, current_gap_len, last_obs_pos, next_obs_pos, gap_exit_oob


def reconstruct_ball_regimes(ball_pos_clean: np.ndarray,
                             ball_vis_clean: np.ndarray,
                             dist_to_obs: np.ndarray,
                             graded_blocks: list = GRADED_BLOCKS):
    """Regime-aware ball coordinate reconstruction (Step 1 Patched).

    Regimes:
        0 = OBSERVED: ball_vis_clean == True (conf = 1.0)
        1 = MICRO_GAP: gap 1..30 frames (<= 1.2s), linear interpolation, conf = exp(-dist/25)
        2 = MEDIUM_GAP: gap 31..300 frames (1.2s..12.0s), linear interpolation, conf = 0.25
        3 = LONG_GAP: gap > 300 frames (> 12.0s) or pre-kickoff/ungraded, linear fill, conf = 0.0

    Returns:
        ball_pos_rec: np.ndarray (N, 2) float32
        ball_regime: np.ndarray (N,) int8
        ball_conf: np.ndarray (N,) float32
    """
    n_frames = len(ball_vis_clean)
    ball_pos_rec = ball_pos_clean.copy()
    ball_regime = np.full(n_frames, 3, dtype=np.int8)
    ball_conf = np.zeros(n_frames, dtype=np.float32)

    # Regime 0: Observed
    ball_regime[ball_vis_clean] = 0
    ball_conf[ball_vis_clean] = 1.0

    in_gap = False
    gs = None
    all_gaps = []
    for f in range(n_frames):
        if not ball_vis_clean[f] and not in_gap:
            in_gap = True
            gs = f
        elif ball_vis_clean[f] and in_gap:
            in_gap = False
            all_gaps.append((gs, f - 1))
    if in_gap:
        all_gaps.append((gs, n_frames - 1))

    def get_half(frame_idx):
        for h, first, last in graded_blocks:
            if first <= frame_idx <= last:
                return h
        return None

    for gs, ge in all_gaps:
        dur = ge - gs + 1
        half = get_half(gs)

        has_prev = (gs > 0 and ball_vis_clean[gs - 1])
        has_next = (ge + 1 < n_frames and ball_vis_clean[ge + 1])

        if has_prev and has_next:
            p_prev = ball_pos_clean[gs - 1]
            p_next = ball_pos_clean[ge + 1]
        elif has_prev and not has_next:
            p_prev = ball_pos_clean[gs - 1]
            p_next = p_prev
        elif not has_prev and has_next:
            p_next = ball_pos_clean[ge + 1]
            p_prev = p_next
        else:
            p_prev = PITCH_CENTER
            p_next = PITCH_CENTER

        gap_frames = np.arange(gs, ge + 1)
        alpha = (gap_frames - gs + 1.0) / float(dur + 1)
        p_interp_x = (1.0 - alpha) * p_prev[0] + alpha * p_next[0]
        p_interp_y = (1.0 - alpha) * p_prev[1] + alpha * p_next[1]
        p_interp = np.stack([p_interp_x, p_interp_y], axis=1)

        ball_pos_rec[gs:ge + 1] = p_interp

        if half is not None:
            if dur <= 30:
                ball_regime[gs:ge + 1] = 1  # MICRO_GAP
                ball_conf[gs:ge + 1] = np.exp(-dist_to_obs[gs:ge + 1] / 25.0)
            elif dur <= 300:
                ball_regime[gs:ge + 1] = 2  # MEDIUM_GAP
                ball_conf[gs:ge + 1] = 0.25
            else:
                ball_regime[gs:ge + 1] = 3  # LONG_GAP
                ball_conf[gs:ge + 1] = 0.0
        else:
            ball_regime[gs:ge + 1] = 3  # UNGRADED_GAP
            ball_conf[gs:ge + 1] = 0.0

    # Perimeter margin clipping [-5.0, 110.3] x [-5.0, 73.0]
    ball_pos_rec[:, 0] = np.clip(ball_pos_rec[:, 0], -5.0, PITCH_LENGTH + 5.0)
    ball_pos_rec[:, 1] = np.clip(ball_pos_rec[:, 1], -5.0, PITCH_WIDTH + 5.0)

    return ball_pos_rec, ball_regime, ball_conf


def compute_ball_kinematics(ball_pos_rec: np.ndarray,
                            ball_vis_clean: np.ndarray,
                            ball_regime: np.ndarray,
                            dt: float = DT,
                            max_step_speed: float = 42.0,
                            max_speed: float = 40.0,
                            max_accel: float = 50.0):
    """Compute ball kinematics on observed segments before gap interpolation,
    eliminating boundary derivative spikes and clamping outputs.

    Clamping:
        ball_speed: [0.0, 40.0] m/s
        ball_accel: [-50.0, 50.0] m/s^2
    """
    n_frames = len(ball_vis_clean)
    ball_vel = np.zeros((n_frames, 2), dtype=np.float32)
    ball_speed = np.zeros(n_frames, dtype=np.float32)
    ball_accel = np.zeros(n_frames, dtype=np.float32)

    # 1. Differentiate along contiguous observed segments
    vis_runs = []
    in_r = False
    rs = None
    for f in range(n_frames):
        if ball_vis_clean[f] and not in_r:
            in_r = True
            rs = f
        elif not ball_vis_clean[f] and in_r:
            in_r = False
            vis_runs.append((rs, f - 1))
    if in_r:
        vis_runs.append((rs, n_frames - 1))

    # Split observed runs at any frame-to-frame step > 42.0 m/s
    sub_runs = []
    for rs, re in vis_runs:
        curr_s = rs
        for t in range(rs + 1, re + 1):
            step_spd = np.hypot(ball_pos_rec[t, 0] - ball_pos_rec[t - 1, 0],
                                ball_pos_rec[t, 1] - ball_pos_rec[t - 1, 1]) / dt
            if step_spd > max_step_speed:
                if t - 1 >= curr_s:
                    sub_runs.append((curr_s, t - 1))
                curr_s = t
        if re >= curr_s:
            sub_runs.append((curr_s, re))

    kernel = np.ones(5, dtype=np.float32) / 5.0
    for s, e in sub_runs:
        dur = e - s + 1
        if dur >= 2:
            v_x = np.gradient(ball_pos_rec[s:e + 1, 0], dt).astype(np.float32)
            v_y = np.gradient(ball_pos_rec[s:e + 1, 1], dt).astype(np.float32)
            if dur >= 5:
                v_x = np.convolve(v_x, kernel, mode="same")
                v_y = np.convolve(v_y, kernel, mode="same")
            s_vals = np.hypot(v_x, v_y)
            a_vals = np.gradient(s_vals, dt).astype(np.float32)
            ball_vel[s:e + 1, 0] = v_x
            ball_vel[s:e + 1, 1] = v_y
            ball_speed[s:e + 1] = s_vals
            ball_accel[s:e + 1] = a_vals

    # 2. Linear velocity across micro gaps (Regime 1)
    in_gap = False
    gs = None
    all_gaps = []
    for f in range(n_frames):
        if (ball_regime[f] == 1) and not in_gap:
            in_gap = True
            gs = f
        elif (ball_regime[f] != 1) and in_gap:
            in_gap = False
            all_gaps.append((gs, f - 1))
    if in_gap:
        all_gaps.append((gs, n_frames - 1))

    for gs, ge in all_gaps:
        dur = ge - gs + 1
        if gs > 0 and ge + 1 < n_frames:
            p_prev = ball_pos_rec[gs - 1]
            p_next = ball_pos_rec[ge + 1]
            v_const = (p_next - p_prev) / float((dur + 1) * dt)
            spd_const = float(np.hypot(v_const[0], v_const[1]))
            if spd_const > max_speed:
                v_const = v_const * (max_speed / max(spd_const, 1e-6))
                spd_const = max_speed
            ball_vel[gs:ge + 1] = v_const
            ball_speed[gs:ge + 1] = spd_const
            ball_accel[gs:ge + 1] = 0.0

    # 3. Regime 2 & 3: zero out kinematics
    stoppage = (ball_regime >= 2)
    ball_vel[stoppage] = 0.0
    ball_speed[stoppage] = 0.0
    ball_accel[stoppage] = 0.0

    # 4. Strict clamping
    ball_speed = np.clip(ball_speed, 0.0, max_speed)
    ball_accel = np.clip(ball_accel, -max_accel, max_accel)

    return ball_vel, ball_speed, ball_accel


def build_and_cache_ball_recovery(cache_path: str = RECOVERED_CACHE_PATH,
                                  tactical_cache_path: str = DEFAULT_CACHE_PATH,
                                  force_reload: bool = False):
    """Execute full patched ball recovery pipeline and cache result to .npz file."""
    if os.path.isfile(cache_path) and not force_reload:
        return load_recovered_ball_data(cache_path)

    print("Loading tactical base data...")
    tactical = get_or_load_tactical_data(tactical_cache_path)

    ball_pos = tactical["ball_pos"]
    ball_vis = tactical["ball_vis"]
    home_players = tactical["home_players"]
    away_players = tactical["away_players"]

    print("1/5 Scrubbing ball tracker glitches...")
    ball_pos_clean, ball_vis_clean, n_scrubbed = filter_ball_glitches(ball_pos, ball_vis)
    print(f"    Scrubbed {n_scrubbed} glitch frames.")

    print("2/5 Computing player-swarm centroid & gap context...")
    swarm_pos = compute_player_swarm_centroid(home_players, away_players)
    (frames_since_obs, frames_until_obs, current_gap_len,
     last_obs_pos, next_obs_pos, gap_exit_oob) = compute_gap_context(ball_pos_clean, ball_vis_clean)

    print("3/5 Computing distance to nearest observed ball...")
    dist_to_obs = compute_distance_to_observed(ball_vis_clean)

    print("4/5 Reconstructing ball trajectory across updated regimes...")
    ball_pos_rec, ball_regime, ball_conf = reconstruct_ball_regimes(
        ball_pos_clean, ball_vis_clean, dist_to_obs
    )

    print("5/5 Calculating clamped ball kinematics (velocity, speed, acceleration)...")
    ball_vel, ball_speed, ball_accel = compute_ball_kinematics(ball_pos_rec, ball_vis_clean, ball_regime)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        ball_vis_clean=ball_vis_clean,
        ball_pos_rec=ball_pos_rec,
        ball_regime=ball_regime,
        ball_conf=ball_conf,
        dist_to_obs=dist_to_obs,
        ball_vel=ball_vel,
        ball_speed=ball_speed,
        ball_accel=ball_accel,
        swarm_pos=swarm_pos,
        frames_since_obs=frames_since_obs,
        frames_until_obs=frames_until_obs,
        current_gap_len=current_gap_len,
        last_obs_pos=last_obs_pos,
        next_obs_pos=next_obs_pos,
        gap_exit_oob=gap_exit_oob,
    )
    print(f"Saved patched recovered ball cache to {os.path.relpath(cache_path, ROOT_DIR)}.")

    return {
        "ball_vis_clean": ball_vis_clean,
        "ball_pos_rec": ball_pos_rec,
        "ball_regime": ball_regime,
        "ball_conf": ball_conf,
        "dist_to_obs": dist_to_obs,
        "ball_vel": ball_vel,
        "ball_speed": ball_speed,
        "ball_accel": ball_accel,
        "swarm_pos": swarm_pos,
        "frames_since_obs": frames_since_obs,
        "frames_until_obs": frames_until_obs,
        "current_gap_len": current_gap_len,
        "last_obs_pos": last_obs_pos,
        "next_obs_pos": next_obs_pos,
        "gap_exit_oob": gap_exit_oob,
        "n_scrubbed": n_scrubbed,
    }


def load_recovered_ball_data(cache_path: str = RECOVERED_CACHE_PATH):
    """Load cached recovered ball data in < 0.5s."""
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Recovered ball cache not found at {cache_path}")

    with np.load(cache_path) as data:
        return {
            "ball_vis_clean": data["ball_vis_clean"],
            "ball_pos_rec": data["ball_pos_rec"],
            "ball_regime": data["ball_regime"],
            "ball_conf": data["ball_conf"],
            "dist_to_obs": data["dist_to_obs"],
            "ball_vel": data["ball_vel"],
            "ball_speed": data["ball_speed"],
            "ball_accel": data["ball_accel"],
            "swarm_pos": data["swarm_pos"],
            "frames_since_obs": data["frames_since_obs"],
            "frames_until_obs": data["frames_until_obs"],
            "current_gap_len": data["current_gap_len"],
            "last_obs_pos": data["last_obs_pos"],
            "next_obs_pos": data["next_obs_pos"],
            "gap_exit_oob": data["gap_exit_oob"],
            "n_scrubbed": 0,
        }


def get_or_build_recovered_ball_data(cache_path: str = RECOVERED_CACHE_PATH,
                                     force_reload: bool = False):
    """Load or generate recovered ball dataset."""
    if os.path.isfile(cache_path) and not force_reload:
        return load_recovered_ball_data(cache_path)
    return build_and_cache_ball_recovery(cache_path=cache_path, force_reload=force_reload)


def forensic_analysis_public_gt():
    """Detailed forensic analysis of public ground truth vs ball visibility."""
    tactical = get_or_load_tactical_data()
    gt, (w_start, w_end) = load_public_ground_truth()
    vis = tactical["ball_vis"]
    ball = tactical["ball_pos"]

    w_vis = vis[w_start:w_end + 1]
    w_gt = gt[w_start:w_end + 1]

    ct = {}
    for v in [True, False]:
        sub = w_gt[w_vis == v]
        u, c = np.unique(sub, return_counts=True)
        ct[v] = dict(zip(u, c))
        ct[v]["total"] = len(sub)

    gaps = []
    in_gap = False
    start = None
    for f in range(w_start, w_end + 1):
        if not vis[f] and not in_gap:
            in_gap = True
            start = f
        elif vis[f] and in_gap:
            in_gap = False
            gaps.append((start, f - 1))
    if in_gap:
        gaps.append((start, w_end))

    gap_details = []
    for idx, (gs, ge) in enumerate(gaps, start=1):
        dur = ge - gs + 1
        dur_s = dur * DT
        labels = "".join(gt[gs:ge + 1])
        runs = []
        for l in labels:
            if not runs or runs[-1][0] != l:
                runs.append([l, 1])
            else:
                runs[-1][1] += 1
        seq_str = " -> ".join([f"{r[0]}({r[1]})" for r in runs])
        prev_coord = f"({ball[gs - 1, 0]:.1f}, {ball[gs - 1, 1]:.1f})" if gs > 0 and vis[gs - 1] else "None"
        next_coord = f"({ball[ge + 1, 0]:.1f}, {ball[ge + 1, 1]:.1f})" if ge + 1 < len(vis) and vis[ge + 1] else "None"
        gap_details.append({
            "idx": idx,
            "start": gs,
            "end": ge,
            "dur_f": dur,
            "dur_s": dur_s,
            "prev_coord": prev_coord,
            "next_coord": next_coord,
            "seq": seq_str,
        })

    return {
        "contingency": ct,
        "gaps": gap_details,
        "n_gaps": len(gaps),
    }


def forensic_analysis_graded_gaps():
    """Bucket analysis of all 150 graded missing-ball episodes."""
    tactical = get_or_load_tactical_data()
    vis = tactical["ball_vis"]
    ball = tactical["ball_pos"]
    tagging = tactical["smart_tagging"]
    hp = tactical["home_players"]
    ap = tactical["away_players"]
    all_spd = np.concatenate([hp[:, :, 2], ap[:, :, 2]], axis=1)

    set_pieces = [row for row in tagging if row["code"] == "SET PIECES"]

    graded_gaps = []
    for half, first, last in GRADED_BLOCKS:
        in_g = False
        gs = None
        for f in range(first, last + 1):
            if not vis[f] and not in_g:
                in_g = True
                gs = f
            elif vis[f] and in_g:
                in_g = False
                graded_gaps.append((half, gs, f - 1))
        if in_g:
            graded_gaps.append((half, gs, last))

    buckets = {
        "Micro (1-25f)": [],
        "Short (26-100f)": [],
        "Medium (101-300f)": [],
        "Long (>300f)": []
    }

    for half, gs, ge in graded_gaps:
        dur = ge - gs + 1
        if dur <= 25:
            b_name = "Micro (1-25f)"
        elif dur <= 100:
            b_name = "Short (26-100f)"
        elif dur <= 300:
            b_name = "Medium (101-300f)"
        else:
            b_name = "Long (>300f)"

        spd_slice = all_spd[gs:ge + 1]
        valid_spd = spd_slice[~np.isnan(spd_slice)]
        mean_spd = float(np.mean(valid_spd)) if len(valid_spd) else 0.0

        start_at_bound = False
        if gs > 0 and vis[gs - 1]:
            bx, by = ball[gs - 1, 0], ball[gs - 1, 1]
            if bx < 1.5 or bx > (PITCH_LENGTH - 1.5) or by < 1.5 or by > (PITCH_WIDTH - 1.5):
                start_at_bound = True

        end_near_sp = False
        for sp in set_pieces:
            if abs(ge - sp["start_frame"]) <= 75 or (sp["start_frame"] <= ge <= sp["end_frame"]):
                end_near_sp = True
                break

        buckets[b_name].append({
            "half": half,
            "gs": gs,
            "ge": ge,
            "dur": dur,
            "mean_spd": mean_spd,
            "start_at_bound": start_at_bound,
            "end_near_sp": end_near_sp,
        })

    bucket_summary = {}
    for b_name, items in buckets.items():
        cnt = len(items)
        tot_f = sum(x["dur"] for x in items)
        avg_spd = float(np.mean([x["mean_spd"] for x in items])) if cnt else 0.0
        bound_pct = float(np.mean([x["start_at_bound"] for x in items])) * 100.0 if cnt else 0.0
        sp_pct = float(np.mean([x["end_near_sp"] for x in items])) * 100.0 if cnt else 0.0
        bucket_summary[b_name] = {
            "count": cnt,
            "frames": tot_f,
            "pct_frames": (tot_f / TOTAL_GRADED_FRAMES) * 100.0,
            "mean_spd": avg_spd,
            "boundary_pct": bound_pct,
            "set_piece_pct": sp_pct,
        }

    return bucket_summary


if __name__ == "__main__":
    import time
    t0 = time.time()
    data = build_and_cache_ball_recovery(force_reload=True)
    t1 = time.time()
    print(f"Patched Phase 2 complete in {t1 - t0:.2f}s.")
