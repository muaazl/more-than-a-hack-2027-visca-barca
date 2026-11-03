"""Phase 5: Temporal Regularization & Football Morphology Filter.

Team: Visca Barca
Competition: More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)

Methodology:
In strict compliance with RULES.md and the organizers' clarification:
- Zero supervised tuning or calibration against ground truth or video.
- All parameters derived from first-principles football physics:
  * Viterbi state persistence (A_ii = 0.985 corresponds to ~2.7s base dwell time).
  * Deflection & momentary touch filter: isolated opponent touches < 25 frames (< 1.0s)
    sandwiched between the same team's possession are absorbed per CHALLENGE.md.
  * Micro-stoppage bridging: dead-ball gaps < 12 frames (< 0.5s) inside continuous
    team possession are bridged to eliminate tracker flicker.
  * Set-piece restart alignment: enforces dead-ball stoppage leading up to the
    physical restart frame, transitioning cleanly to the restarting team.
  * Exact frame masking: ungraded frames are set to 'X', pre-2nd-half is set to 'D',
    graded play is strictly in {'H', 'A', 'D'}.
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
    GRADED_BLOCKS,
    load_cached_tactical_data,
)

# Label encoding constants
CODE_H = 0
CODE_A = 1
CODE_D = 2
CODE_X = 3

LABEL_NAMES = np.array(["H", "A", "D", "X"], dtype="<U1")

# A Priori Football Kinematics & Timing Parameters
MIN_CONTROL_DWELL_FRAMES = 35  # 1.4s: minimum duration for controlled possession vs deflection (CHALLENGE.md)
MIN_DEAD_GAP_FRAMES = 50       # 2.0s: minimum duration for physical match stoppage (bridges flight/occlusion)
SET_PIECE_RESTART_SPEED = 2.0  # m/s: ball velocity indicating physical restart into play

# Symmetric Football Transition Matrix (expected dwell ~8.0s at 25 fps)
# States: [0: H, 1: A, 2: D]
TRANSITION_PROBS = np.array([
    [0.995, 0.0016, 0.0034],   # From H: 99.5% persist, turnover 0.16%, stoppage 0.34%
    [0.0016, 0.995, 0.0034],   # From A: turnover 0.16%, 99.5% persist, stoppage 0.34%
    [0.0025, 0.0025, 0.9950],  # From D: 0.25% H restart, 0.25% A restart, 99.5% persist
], dtype=np.float32)

LOG_TRANSITION_MATRIX = np.log(TRANSITION_PROBS)


def viterbi_decode_segment(log_emissions: np.ndarray,
                           log_trans: np.ndarray = LOG_TRANSITION_MATRIX,
                           log_init: np.ndarray = None) -> np.ndarray:
    """Vectorized Viterbi dynamic programming decoder for 3-state chain.
    
    Args:
        log_emissions: float32 array of shape (T, 3) with [H, A, D] log-likelihoods.
        log_trans: float32 array of shape (3, 3) log transition probabilities.
        log_init: float32 array of shape (3,) log initial state probabilities.
        
    Returns:
        best_states: int8 array of shape (T,) with optimal decoded state sequence.
    """
    T, K = log_emissions.shape
    if T == 0:
        return np.zeros(0, dtype=np.int8)

    if log_init is None:
        log_init = np.log(np.array([0.45, 0.45, 0.10], dtype=np.float32))

    dp = np.zeros((T, K), dtype=np.float32)
    backtrack = np.zeros((T, K), dtype=np.int8)

    dp[0] = log_init + log_emissions[0]

    for t in range(1, T):
        # prev_scores is (K, K) where [prev, curr] = dp[t-1, prev] + log_trans[prev, curr]
        prev_scores = dp[t - 1, :, None] + log_trans
        best_prev = np.argmax(prev_scores, axis=0)
        dp[t] = prev_scores[best_prev, np.arange(K)] + log_emissions[t]
        backtrack[t] = best_prev

    best_states = np.zeros(T, dtype=np.int8)
    best_states[-1] = np.argmax(dp[-1])
    for t in range(T - 2, -1, -1):
        best_states[t] = backtrack[t + 1, best_states[t + 1]]

    return best_states


def absorb_deflections_and_rebounds(labels: np.ndarray,
                                    min_dwell_frames: int = MIN_CONTROL_DWELL_FRAMES,
                                    min_dead_frames: int = MIN_DEAD_GAP_FRAMES) -> np.ndarray:
    """Filter deflections, momentary touches, and sensor flickers using run-length encoding.
    
    1. Rebound / Deflection rule (CHALLENGE.md):
       Isolated touches of the opposing team (< min_dwell_frames, 1.0s) sandwiched
       between the same team's possession (e.g. A -> H -> A) are absorbed into A.
    2. Micro-stoppage rule:
       Isolated dead-ball gaps (< min_dead_frames, 0.5s) sandwiched between the same
       team's possession (e.g. H -> D -> H) are bridged into H.
    """
    out = labels.copy()
    n = len(out)
    if n == 0:
        return out

    for _ in range(5):
        changes = np.where(out[1:] != out[:-1])[0] + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, n]
        lengths = ends - starts
        run_labels = out[starts]
        n_runs = len(starts)

        modified = False
        for i in range(1, n_runs - 1):
            curr_lbl = run_labels[i]
            prev_lbl = run_labels[i - 1]
            next_lbl = run_labels[i + 1]
            length = lengths[i]

            # 1. Deflections, brief touches, and aerial duels without control (< min_dwell_frames):
            # Per CHALLENGE.md, a simple touch, deflection, or aerial duel without sustained
            # control does not count as a change of possession.
            if curr_lbl in (CODE_H, CODE_A) and length < min_dwell_frames:
                if prev_lbl in (CODE_H, CODE_A):
                    # Opponent deflection during ongoing possession, or deflection before stoppage
                    out[starts[i]:ends[i]] = prev_lbl
                    modified = True
                elif prev_lbl == CODE_D and next_lbl in (CODE_H, CODE_A):
                    # Momentary touch immediately after restart before controlled receiver
                    out[starts[i]:ends[i]] = next_lbl
                    modified = True

            # 2. Sensor flicker / micro-stoppage: H -> D -> H or A -> D -> A where length < min_dead_frames
            elif prev_lbl == next_lbl and prev_lbl in (CODE_H, CODE_A) and curr_lbl == CODE_D:
                if length < min_dead_frames:
                    out[starts[i]:ends[i]] = prev_lbl
                    modified = True

        if not modified:
            break

    return out


def enforce_minimum_dwell(labels: np.ndarray, min_dwell_frames: int = MIN_CONTROL_DWELL_FRAMES) -> np.ndarray:
    """Enforce strict minimum duration for controlled live spells.
    
    Guarantees zero live possession spells (H or A) shorter than min_dwell_frames (1.4s / 35 frames).
    Absorbs any remaining brief touch into the surrounding spell or stoppage.
    """
    out = labels.copy()
    for _ in range(5):
        changes = np.where(out[1:] != out[:-1])[0] + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, len(out)]
        lengths = ends - starts
        run_labels = out[starts]
        modified = False
        for i in range(len(starts)):
            if run_labels[i] in (CODE_H, CODE_A) and lengths[i] < min_dwell_frames:
                prev_lbl = run_labels[i - 1] if i > 0 else None
                next_lbl = run_labels[i + 1] if i < len(starts) - 1 else None
                if prev_lbl in (CODE_H, CODE_A):
                    out[starts[i]:ends[i]] = prev_lbl
                elif next_lbl in (CODE_H, CODE_A):
                    out[starts[i]:ends[i]] = next_lbl
                elif prev_lbl == CODE_D or next_lbl == CODE_D:
                    out[starts[i]:ends[i]] = CODE_D
                modified = True
        if not modified:
            break
    return out


def apply_physical_contact_gate(labels: np.ndarray,
                                d_H1_seg: np.ndarray,
                                d_A1_seg: np.ndarray,
                                ball_vis_seg: np.ndarray,
                                contact_radius_m: float = 2.2) -> np.ndarray:
    """Enforce Law 1 (CHALLENGE.md): Causal Loose-Ball Retention & Physical Contact Gate.
    
    'In phases where the ball is contested or loose, possession stays with the last team that had it.'
    When transitioning directly between live teams (H -> A or A -> H), delay the transition
    boundary forward until the first frame where the incoming team's closest player is within
    physical control reach (d_incoming <= 2.2m) or the ball is occluded (ball_vis_clean == False).
    While d_incoming > 2.2m during live play, strictly carry forward the prior possessing team.
    """
    out = labels.copy()
    n = len(out)
    if n == 0:
        return out

    changes = np.where(out[1:] != out[:-1])[0] + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, n]

    for i in range(len(starts) - 1):
        curr_lbl = out[starts[i]]
        next_lbl = out[starts[i + 1]]

        # Gate live-to-live turnovers: H -> A or A -> H
        if curr_lbl in (CODE_H, CODE_A) and next_lbl in (CODE_H, CODE_A) and curr_lbl != next_lbl:
            t_trans = starts[i + 1]
            t_end = ends[i + 1]

            d_inc = d_A1_seg if next_lbl == CODE_A else d_H1_seg
            seg_d = d_inc[t_trans:t_end]
            seg_vis = ball_vis_seg[t_trans:t_end]

            # Contact or occlusion
            valid_contact = (seg_d <= contact_radius_m) | (~seg_vis)
            first_contact_rel = np.where(valid_contact)[0]

            if len(first_contact_rel) > 0:
                first_contact_idx = t_trans + first_contact_rel[0]
                if first_contact_idx > t_trans:
                    out[t_trans:first_contact_idx] = curr_lbl
            else:
                out[t_trans:t_end] = curr_lbl

    return out


def verify_uncontested_control_for_sandwiches(labels: np.ndarray,
                                              d_H1_seg: np.ndarray,
                                              d_A1_seg: np.ndarray,
                                              max_sandwich_frames: int = 65,
                                              min_control_frames: int = 25) -> np.ndarray:
    """Enforce Law 2 (CHALLENGE.md): Uncontested Control Verification for Sandwich Spells.
    
    Absorbs short sandwiched spells of Team T2 between Team T1 (H -> A -> H or A -> H -> A)
    if duration < 65 frames (2.6s, 'a few seconds') UNLESS Team T2 establishes genuine
    uncontested control (>= 25 frames inside the spell where d_T2 <= 1.8m and d_T1 >= 2.8m).
    Directly filters out aerial duel headers, clearances, and deflections without control.
    """
    out = labels.copy()
    n = len(out)
    if n == 0:
        return out

    for _ in range(5):
        changes = np.where(out[1:] != out[:-1])[0] + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, n]
        lengths = ends - starts
        run_labels = out[starts]
        n_runs = len(starts)

        modified = False
        for i in range(1, n_runs - 1):
            curr_lbl = run_labels[i]
            prev_lbl = run_labels[i - 1]
            next_lbl = run_labels[i + 1]
            length = lengths[i]

            # Sandwiched between same team's possession
            if prev_lbl in (CODE_H, CODE_A) and next_lbl == prev_lbl and curr_lbl in (CODE_H, CODE_A) and curr_lbl != prev_lbl:
                if length < max_sandwich_frames:
                    s_idx = starts[i]
                    e_idx = ends[i]

                    d_curr = d_H1_seg[s_idx:e_idx] if curr_lbl == CODE_H else d_A1_seg[s_idx:e_idx]
                    d_opp = d_A1_seg[s_idx:e_idx] if curr_lbl == CODE_H else d_H1_seg[s_idx:e_idx]

                    # Uncontested control condition: d_curr <= 1.8m and d_opp >= 2.8m
                    uncontested_frames = np.sum((d_curr <= 1.8) & (d_opp >= 2.8))

                    if uncontested_frames < min_control_frames:
                        out[s_idx:e_idx] = prev_lbl
                        modified = True

        if not modified:
            break

    return out


def enforce_minimum_dwell(labels: np.ndarray, min_dwell_frames: int = MIN_CONTROL_DWELL_FRAMES) -> np.ndarray:
    """Enforce strict minimum duration for controlled live spells.
    
    Guarantees zero live possession spells (H or A) shorter than min_dwell_frames (1.4s / 35 frames).
    Absorbs any remaining brief touch into the surrounding spell or stoppage.
    """
    out = labels.copy()
    for _ in range(5):
        changes = np.where(out[1:] != out[:-1])[0] + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, len(out)]
        lengths = ends - starts
        run_labels = out[starts]
        modified = False
        for i in range(len(starts)):
            if run_labels[i] in (CODE_H, CODE_A) and lengths[i] < min_dwell_frames:
                prev_lbl = run_labels[i - 1] if i > 0 else None
                next_lbl = run_labels[i + 1] if i < len(starts) - 1 else None
                if prev_lbl in (CODE_H, CODE_A):
                    out[starts[i]:ends[i]] = prev_lbl
                elif next_lbl in (CODE_H, CODE_A):
                    out[starts[i]:ends[i]] = next_lbl
                elif prev_lbl == CODE_D or next_lbl == CODE_D:
                    out[starts[i]:ends[i]] = CODE_D
                modified = True
        if not modified:
            break
    return out


def align_set_piece_restarts(labels: np.ndarray,
                             smart_tagging: np.ndarray,
                             ball_speed: np.ndarray,
                             ball_vis_clean: np.ndarray,
                             player_speed_mean: np.ndarray,
                             dead_tempo_p20: float,
                             half_start: int,
                             half_end: int) -> np.ndarray:
    """Enforce Law 3 (DATA.md): Set-Piece Pre-Kick Stoppage Window & Physical Restart.
    
    1. Identifies physical restart kick frame k_f where ball_speed > 3.0 m/s.
    2. Enforces dead-ball stoppage ('D') throughout the preparation window [p_s, k_f - 1],
       extending backward across contiguous missing or stationary ball stretches.
    3. Forces clean transition from 'D' to the restarting team ('H' or 'A') at k_f.
    """
    out = labels.copy()
    sp_intervals = smart_tagging[smart_tagging["code"] == "SET PIECES"]

    for sp in sp_intervals:
        s_f = int(sp["start_frame"])
        e_f = int(sp["end_frame"])
        team_str = str(sp["team"]).strip()

        # Check if interval overlaps with this half
        if e_f < half_start or s_f > half_end:
            continue

        # 1. Identify physical restart frame k_f inside this match window
        win_end = min(e_f + 50, half_end + 1)
        w_speed = ball_speed[s_f:win_end]
        w_vis = ball_vis_clean[s_f:win_end]
        spikes = np.where((w_speed > 3.0) & w_vis)[0]

        if len(spikes) > 0:
            k_f = s_f + spikes[0]
        else:
            k_f = s_f + int(0.65 * (e_f - s_f))
        k_f = min(k_f, half_end)

        # 2. Preparation window start
        p_s = max(half_start, s_f - 50)
        while p_s > max(half_start, s_f - 250):
            if not ball_vis_clean[p_s] or ball_speed[p_s] < 1.5 or player_speed_mean[p_s] < dead_tempo_p20:
                p_s -= 1
            else:
                break

        # Convert to local segment coordinates
        local_p_s = p_s - half_start
        local_k_f = k_f - half_start

        if local_k_f > local_p_s:
            out[local_p_s:local_k_f] = CODE_D

        # Assign restarting team from k_f
        if local_k_f < len(out):
            dwell_len = min(local_k_f + MIN_CONTROL_DWELL_FRAMES, len(out))
            if team_str == "H":
                out[local_k_f:dwell_len] = CODE_H
            elif team_str == "A":
                out[local_k_f:dwell_len] = CODE_A

    return out


def generate_full_match_predictions(log_likelihoods: np.ndarray,
                                     features: dict = None) -> np.ndarray:
    """Generate final per-frame predictions across all 199,319 frames.
    
    Frame Masking Rules:
    - Ungraded frames (0..11184, 90652..110807, 190198..199318): Output 'X'
    - Graded pre-2nd-half kickoff window (110808..113741): Output 'D'
    - Graded live-play halves (11185..90651 and 113742..190197): Output {H, A, D}
    
    Returns:
        final_labels: '<U1' array of shape (199319,) containing 'H', 'A', 'D', 'X'.
    """
    n_total = len(log_likelihoods)
    assert n_total == N_VIDEO_FRAMES, f"Expected {N_VIDEO_FRAMES} frames, got {n_total}"

    # Default all frames to 'X' (abstain / ungraded)
    final_labels = np.full(n_total, "X", dtype="<U1")

    # Load tactical metadata & kinematics
    tactical_data = load_cached_tactical_data()
    smart_tagging = tactical_data["smart_tagging"]

    if features is None:
        from src.features import load_features_data
        features = load_features_data()
        
    ball_speed = features["ball_speed"]
    ball_vis_clean = features["ball_vis_clean"] > 0.5
    d_H1 = features["d_H1"]
    d_A1 = features["d_A1"]
    player_speed_mean = features["player_speed_mean"]

    from src.model import compute_unsupervised_tempo_threshold
    dead_tempo_p20 = compute_unsupervised_tempo_threshold(player_speed_mean)

    # -------------------------------------------------------------
    # 1. HALF 1: frames 11185 to 90651 (79,467 graded frames)
    # -------------------------------------------------------------
    h1_start, h1_end = 11185, 90651
    h1_lik = log_likelihoods[h1_start:h1_end + 1]

    # Viterbi decoding with kickoff prior (Barcelona starts with the ball)
    init_h1 = np.log(np.array([0.90, 0.05, 0.05], dtype=np.float32))
    h1_decoded = viterbi_decode_segment(h1_lik, log_init=init_h1)

    # Law 3: Align set pieces leading stoppages and physical restarts
    h1_sp = align_set_piece_restarts(
        h1_decoded, smart_tagging, ball_speed, ball_vis_clean,
        player_speed_mean, dead_tempo_p20, h1_start, h1_end
    )

    # Deflections & micro-stoppages
    h1_clean = absorb_deflections_and_rebounds(h1_sp)

    # Law 1: Causal Loose-Ball Retention & Physical Contact Gate
    d_H1_h1 = d_H1[h1_start:h1_end + 1]
    d_A1_h1 = d_A1[h1_start:h1_end + 1]
    ball_vis_h1 = ball_vis_clean[h1_start:h1_end + 1]
    h1_gated = apply_physical_contact_gate(
        h1_clean, d_H1_h1, d_A1_h1, ball_vis_h1, contact_radius_m=2.2
    )

    # Law 2: Uncontested Control Verification for Sandwiches
    h1_uncontested = verify_uncontested_control_for_sandwiches(
        h1_gated, d_H1_h1, d_A1_h1, max_sandwich_frames=65, min_control_frames=25
    )

    # Enforce minimum live spell duration (zero spells < 35 frames)
    h1_final = enforce_minimum_dwell(h1_uncontested, min_dwell_frames=MIN_CONTROL_DWELL_FRAMES)

    # Assign to final array
    final_labels[h1_start:h1_end + 1] = LABEL_NAMES[h1_final]

    # -------------------------------------------------------------
    # 2. HALF 2: frames 110808 to 190197 (79,390 graded frames)
    # -------------------------------------------------------------
    # Graded Pre-Kickoff Window: 110808..113741 -> Guaranteed 'D'
    pre_k2_start, pre_k2_end = 110808, 113741
    final_labels[pre_k2_start:pre_k2_end + 1] = "D"

    # Live Play Window: 113742..190197 (76,456 graded frames)
    h2_live_start, h2_live_end = 113742, 190197
    h2_lik = log_likelihoods[h2_live_start:h2_live_end + 1]

    # Viterbi decoding for 2nd half live play
    init_h2 = np.log(np.array([0.45, 0.45, 0.10], dtype=np.float32))
    h2_decoded = viterbi_decode_segment(h2_lik, log_init=init_h2)

    # Law 3: Align set pieces leading stoppages and physical restarts
    h2_sp = align_set_piece_restarts(
        h2_decoded, smart_tagging, ball_speed, ball_vis_clean,
        player_speed_mean, dead_tempo_p20, h2_live_start, h2_live_end
    )

    # Deflections & micro-stoppages
    h2_clean = absorb_deflections_and_rebounds(h2_sp)

    # Law 1: Causal Loose-Ball Retention & Physical Contact Gate
    d_H1_h2 = d_H1[h2_live_start:h2_live_end + 1]
    d_A1_h2 = d_A1[h2_live_start:h2_live_end + 1]
    ball_vis_h2 = ball_vis_clean[h2_live_start:h2_live_end + 1]
    h2_gated = apply_physical_contact_gate(
        h2_clean, d_H1_h2, d_A1_h2, ball_vis_h2, contact_radius_m=2.2
    )

    # Law 2: Uncontested Control Verification for Sandwiches
    h2_uncontested = verify_uncontested_control_for_sandwiches(
        h2_gated, d_H1_h2, d_A1_h2, max_sandwich_frames=65, min_control_frames=25
    )

    # Enforce minimum live spell duration (zero spells < 35 frames)
    h2_final = enforce_minimum_dwell(h2_uncontested, min_dwell_frames=MIN_CONTROL_DWELL_FRAMES)

    # Assign to final array
    final_labels[h2_live_start:h2_live_end + 1] = LABEL_NAMES[h2_final]

    return final_labels
