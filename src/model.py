"""Phase 4: Physical Bayesian Possession Inference Engine.

Team: Visca Barca
Competition: More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)

Methodology:
In strict compliance with RULES.md and the organizers' clarification:
- Zero supervised tuning or calibration against ground truth or video.
- All parameters derived from first-principles football physics, FIFA pitch geometry,
  and unsupervised dataset statistics.

States:
  0: Home ('H' - Barcelona)
  1: Away ('A' - Athletic Club)
  2: Dead Ball ('D' - Stoppage)
"""

import os
import sys
import numpy as np

# Path resolution
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.features import load_features_data, FEATURES_CACHE_PATH
from src.data_loader import load_cached_tactical_data

# --- A PRIORI PHYSICAL & GEOMETRIC CONSTANTS ---
# FIFA Laws of the Game & Pitch Dimensions
PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

# Human Biomechanics & Kinematics
PHYSICAL_CONTROL_RADIUS_M = 1.8   # Maximum radius for controlled foot-ball interaction
DUEL_PRESSURE_RADIUS_M = 3.0      # Immediate contesting pressure threshold
PASS_FLIGHT_MIN_SPEED_MPS = 5.0   # Velocity above which ball is traveling in flight / pass
BALL_STATIONARY_MAX_SPEED_MPS = 1.0 # Velocity below which ball is stationary on pitch

# Tracking Continuity & Stoppage Context
BOUNDARY_EXIT_MIN_GAP_FRAMES = 25 # 1.0s missing after boundary exit indicates dead ball
LONG_GAP_STOPPAGE_FRAMES = 300    # > 12.0s tracking loss corresponds to match stoppages

# Tactical Prior Weighting (from 638 Smart Tagging intervals)
TACTICAL_PRIOR_WEIGHT = 0.40

# Graded Frame Constants
HALVES_RANGES = [
    (11185, 90651),   # Half 1 (79,467 frames)
    (110808, 190197)  # Half 2 (79,390 frames)
]
PRE_2ND_HALF_KICKOFF_END = 113741 # Guaranteed dead ball window 110808..113741


def compute_unsupervised_tempo_threshold(player_speed_mean: np.ndarray) -> float:
    """Compute unsupervised dead-ball tempo threshold (p20 of collective speed in graded halves).
    
    During dead-ball stoppages, players decelerate and walk into position, causing
    a prominent lower mode in collective player velocity.
    """
    graded_mask = np.zeros(len(player_speed_mean), dtype=bool)
    for s, e in HALVES_RANGES:
        graded_mask[s:e + 1] = True
    
    valid_speeds = player_speed_mean[graded_mask & ~np.isnan(player_speed_mean)]
    p20 = float(np.percentile(valid_speeds, 20.0))
    return p20


def compute_set_piece_preparation_mask(smart_tagging: np.ndarray,
                                       ball_speed: np.ndarray,
                                       ball_vis_clean: np.ndarray,
                                       player_speed_mean: np.ndarray,
                                       dead_tempo_p20: float) -> np.ndarray:
    """Compute Law 3 (DATA.md) pre-kick stoppage window across all 59 set pieces.
    
    For each set piece interval [start_f, end_f]:
    1. Identify physical restart kick frame k_f where ball_speed > 3.0 m/s or accelerated motion.
    2. Mark preparation window [start_f - 50, k_f], extended backward across contiguous
       missing or stationary ball stretches (< 1.5 m/s or low tempo).
    """
    n_frames = len(ball_speed)
    prep_mask = np.zeros(n_frames, dtype=bool)
    sp_intervals = smart_tagging[smart_tagging["code"] == "SET PIECES"]

    for sp in sp_intervals:
        s_f = int(sp["start_frame"])
        e_f = int(sp["end_frame"])

        # 1. Identify physical restart frame k_f
        win_end = min(e_f + 50, n_frames)
        w_speed = ball_speed[s_f:win_end]
        w_vis = ball_vis_clean[s_f:win_end]
        spikes = np.where((w_speed > 3.0) & w_vis)[0]

        if len(spikes) > 0:
            k_f = s_f + spikes[0]
        else:
            k_f = s_f + int(0.65 * (e_f - s_f))
        k_f = min(k_f, n_frames - 1)

        # 2. Preparation window start
        p_s = max(0, s_f - 50)
        while p_s > max(0, s_f - 250):
            if not ball_vis_clean[p_s] or ball_speed[p_s] < 1.5 or player_speed_mean[p_s] < dead_tempo_p20:
                p_s -= 1
            else:
                break

        prep_mask[p_s:k_f] = True

    return prep_mask


def compute_frame_likelihoods(features: dict = None) -> np.ndarray:
    """Compute a priori 3-class log-likelihoods for all 199,319 frames.
    
    Args:
        features: Dictionary of feature channels from cache/features.npz.
        
    Returns:
        log_likelihoods: float32 array of shape (199319, 3) where columns are [H, A, D].
    """
    if features is None:
        features = load_features_data()
        
    n_frames = len(features["d_H1"])
    log_lik = np.zeros((n_frames, 3), dtype=np.float32)

    # 1. Unsupervised Tempo Threshold
    dead_tempo_p20 = compute_unsupervised_tempo_threshold(features["player_speed_mean"])

    # Extract required feature channels
    norm_diff_d1 = features["norm_diff_d1"]
    norm_diff_d1_s25 = features["norm_diff_d1_smooth_25"]
    norm_diff_d1_s75 = features["norm_diff_d1_smooth_75"]
    soft_ctrl_diff = features["soft_ctrl_diff"]
    soft_ctrl_diff_s25 = features["soft_ctrl_diff_smooth_25"]
    st_team_prior = features["st_team_prior"]
    
    d_H1 = features["d_H1"]
    d_A1 = features["d_A1"]
    min_player_dist = features["min_player_dist"]
    ball_oob_dist = features["ball_oob_dist"]
    is_ball_stationary = features["is_ball_stationary"]
    player_speed_mean = features["player_speed_mean"]
    ball_speed = features["ball_speed"]
    st_set_piece_active = features["st_set_piece_active"]
    gap_exit_oob = features["gap_exit_oob"]
    frames_since_obs = features["frames_since_obs"]
    current_gap_len = features["current_gap_len"]
    pre_2nd_half_mask = features["pre_2nd_half_mask"]

    # -------------------------------------------------------------
    # 2. LIVE POSSESSION LOG-ODDS: S_live = log(P(H)/P(A))
    # -------------------------------------------------------------
    # Multi-scale spatial control signal
    # norm_diff_d1 > 0 means Home is closer; < 0 means Away is closer.
    prox_signal = (0.45 * norm_diff_d1 +
                   0.35 * norm_diff_d1_s25 +
                   0.20 * norm_diff_d1_s75)

    ctrl_signal = (0.50 * soft_ctrl_diff +
                   0.50 * soft_ctrl_diff_s25)

    # Pass flight damping: when ball velocity > 5.0 m/s and no player is within 1.8m,
    # the ball is in flight; downweight raw proximity so momentum persists.
    ball_in_flight = (ball_speed > PASS_FLIGHT_MIN_SPEED_MPS) & (min_player_dist > PHYSICAL_CONTROL_RADIUS_M)
    flight_scale = np.where(ball_in_flight, 0.5, 1.0).astype(np.float32)

    live_log_odds = (3.5 * prox_signal * flight_scale +
                     2.5 * ctrl_signal * flight_scale +
                     TACTICAL_PRIOR_WEIGHT * st_team_prior)

    # Clamp log odds to prevent numerical overflow
    live_log_odds = np.clip(live_log_odds, -6.0, 6.0)

    # Base live log-likelihoods
    # log(P(H)) = +0.5 * log_odds, log(P(A)) = -0.5 * log_odds
    log_lik[:, 0] = 0.5 * live_log_odds
    log_lik[:, 1] = -0.5 * live_log_odds

    # -------------------------------------------------------------
    # 3. DEAD-BALL (D) EVIDENCE SCORE
    # -------------------------------------------------------------
    # Positive evidence for dead ball increases log_lik[:, 2] relative to live play.
    d_evidence = np.zeros(n_frames, dtype=np.float32)

    # a) Out of Bounds Evidence: ball outside the pitch (105.3m x 68.0m)
    oob_active = ball_oob_dist < 0.0
    d_evidence += np.where(oob_active, 4.0, 0.0)

    # b) Missing Gap Following Boundary Exit: ball kicked into stands/touchline
    oob_gap = (gap_exit_oob > 0.5) & (frames_since_obs > BOUNDARY_EXIT_MIN_GAP_FRAMES)
    d_evidence += np.where(oob_gap, 3.5, 0.0)

    # c) Stationary Ball Stoppage: ball stopped on pitch with players detached or low tempo
    stationary_stoppage = (is_ball_stationary > 0.5) & (
        (min_player_dist > PHYSICAL_CONTROL_RADIUS_M) | (player_speed_mean < dead_tempo_p20)
    )
    d_evidence += np.where(stationary_stoppage, 3.0, 0.0)

    # d) Smart Tagging Set Piece Preparation (Law 3 - DATA.md):
    # Boost dead ball score during preparation window leading up to physical restart
    tactical_data = load_cached_tactical_data()
    smart_tagging = tactical_data["smart_tagging"]
    sp_prep_mask = compute_set_piece_preparation_mask(
        smart_tagging, ball_speed, features["ball_vis_clean"] > 0.5,
        player_speed_mean, dead_tempo_p20
    )
    d_evidence += np.where(sp_prep_mask, 6.0, 0.0)

    # e) Extended Missing Gap: tracking loss > 300 frames (12s)
    long_gap_dead = (current_gap_len > LONG_GAP_STOPPAGE_FRAMES) & (player_speed_mean < dead_tempo_p20)
    d_evidence += np.where(long_gap_dead, 3.0, 0.0)

    # f) Negative Evidence (Live Play Counter-Indicators):
    # When ball is inside pitch, moving, and tight to a player (< 1.5m), stoppage is very unlikely
    live_play_confident = (ball_oob_dist > 1.5) & (min_player_dist < 1.5) & (ball_speed > 1.0)
    d_evidence -= np.where(live_play_confident, 3.0, 0.0)

    # Baseline dead-ball threshold offset (-1.0 so live play is the default when no stoppage cues exist)
    log_lik[:, 2] = d_evidence - 1.0

    # -------------------------------------------------------------
    # 4. HARD PHYSICAL CONSTRAINTS & GUARANTEED WINDOWS
    # -------------------------------------------------------------
    # Pre-2nd-half window (110808..113741): Guaranteed dead ball before kickoff
    pre_2nd_half = pre_2nd_half_mask > 0.5
    log_lik[pre_2nd_half, 0] = -20.0
    log_lik[pre_2nd_half, 1] = -20.0
    log_lik[pre_2nd_half, 2] = +20.0

    # Normalize log-likelihoods via log-softmax along axis 1 for numerical stability
    max_lik = np.max(log_lik, axis=1, keepdims=True)
    exp_shifted = np.exp(log_lik - max_lik)
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    log_lik_norm = np.log(np.maximum(probs, 1e-12)).astype(np.float32)

    return log_lik_norm
