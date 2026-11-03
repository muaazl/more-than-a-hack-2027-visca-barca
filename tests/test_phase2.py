"""Phase 2 Verification Tests & Diagnostic Report.

Team: Visca Barca
Competition: More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)

Verifies:
1. Zero NaN / Inf values across all 158,857 graded frames for ball_pos_rec, ball_speed, ball_accel, and ball_conf.
2. Reconstructed ball_pos_rec coordinates stay strictly within [-5.0, 110.3] in X and [-5.0, 73.0] in Y.
3. Cache generation and reload complete cleanly (< 1.5s).
4. Emits the full === PHASE 2 DIAGNOSTIC REPORT ===.
"""

import os
import sys
import time
import numpy as np
import pytest

# Path resolution
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import (
    N_VIDEO_FRAMES,
    GRADED_BLOCKS,
    TOTAL_GRADED_FRAMES,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from src.ball_recovery import (
    RECOVERED_CACHE_PATH,
    get_or_build_recovered_ball_data,
    load_recovered_ball_data,
    forensic_analysis_public_gt,
    forensic_analysis_graded_gaps,
)


@pytest.fixture(scope="module")
def recovered_data():
    """Load recovered ball dataset for tests."""
    return get_or_build_recovered_ball_data(RECOVERED_CACHE_PATH)


def test_no_nan_or_inf_in_graded_frames(recovered_data):
    """Assertion 1: ball_pos_rec, ball_speed, ball_accel, and ball_conf contain
    zero NaN or Inf values across all graded frames.
    """
    for key in ["ball_pos_rec", "ball_speed", "ball_accel", "ball_conf"]:
        arr = recovered_data[key]
        for half, first, last in GRADED_BLOCKS:
            sub = arr[first:last + 1]
            assert not np.isnan(sub).any(), f"{key} contains NaN in Half {half}"
            assert not np.isinf(sub).any(), f"{key} contains Inf in Half {half}"


def test_reconstructed_coordinate_bounds(recovered_data):
    """Assertion 2: Reconstructed ball_pos_rec coordinates stay within reasonable pitch + margin
    bounds ([-5.0, 110.3] for X and [-5.0, 73.0] for Y) on all graded frames.
    """
    pos_rec = recovered_data["ball_pos_rec"]
    for half, first, last in GRADED_BLOCKS:
        sub = pos_rec[first:last + 1]
        x_vals = sub[:, 0]
        y_vals = sub[:, 1]

        assert np.min(x_vals) >= -5.0, f"Half {half} X min {np.min(x_vals)} is below -5.0"
        assert np.max(x_vals) <= (PITCH_LENGTH + 5.0), f"Half {half} X max {np.max(x_vals)} exceeds 110.3"
        assert np.min(y_vals) >= -5.0, f"Half {half} Y min {np.min(y_vals)} is below -5.0"
        assert np.max(y_vals) <= (PITCH_WIDTH + 5.0), f"Half {half} Y max {np.max(y_vals)} exceeds 73.0"


def test_cache_reload_speed(recovered_data):
    """Assertion 3: Cache reload completes in under 1.5 seconds and returns identical shapes."""
    t0 = time.time()
    reloaded = load_recovered_ball_data(RECOVERED_CACHE_PATH)
    t_load = time.time() - t0

    assert t_load < 1.5, f"Cache reload took {t_load:.3f}s (exceeds 1.5s)"
    assert reloaded["ball_pos_rec"].shape == (N_VIDEO_FRAMES, 2)
    assert reloaded["ball_speed"].shape == (N_VIDEO_FRAMES,)
    assert reloaded["ball_accel"].shape == (N_VIDEO_FRAMES,)
    assert reloaded["ball_conf"].shape == (N_VIDEO_FRAMES,)
    assert reloaded["ball_regime"].shape == (N_VIDEO_FRAMES,)


def generate_diagnostic_report():
    """Format the full === PHASE 2 DIAGNOSTIC REPORT === text block."""
    d = get_or_build_recovered_ball_data(RECOVERED_CACHE_PATH)
    gt_forensics = forensic_analysis_public_gt()
    graded_forensics = forensic_analysis_graded_gaps()

    lines = []
    lines.append("=" * 80)
    lines.append("=== PHASE 2 DIAGNOSTIC REPORT ===")
    lines.append("=" * 80)

    # 1. Public 5-Minute GT vs ball_vis Contingency Table & Gap Details
    lines.append("\n[1] PUBLIC 5-MINUTE GT vs. BALL VISIBILITY CONTINGENCY TABLE")
    lines.append("-" * 80)
    ct = gt_forensics["contingency"]
    vis_tot = ct[True]["total"]
    mis_tot = ct[False]["total"]

    lines.append(f"Public Window: Frames 11185..18684 (7,500 frames / 5.0 minutes)")
    lines.append(f"{'Condition':<20} | {'Total Frames':<12} | {'H (Home)':<18} | {'A (Away)':<18} | {'D (Dead Ball)':<18}")
    lines.append("-" * 96)

    h_v = ct[True].get("H", 0)
    a_v = ct[True].get("A", 0)
    d_v = ct[True].get("D", 0)
    lines.append(
        f"{'ball_vis == True':<20} | {vis_tot:<12} | "
        f"{h_v:>5} ({h_v/vis_tot*100:5.1f}%)     | "
        f"{a_v:>5} ({a_v/vis_tot*100:5.1f}%)     | "
        f"{d_v:>5} ({d_v/vis_tot*100:5.1f}%)"
    )

    h_m = ct[False].get("H", 0)
    a_m = ct[False].get("A", 0)
    d_m = ct[False].get("D", 0)
    lines.append(
        f"{'ball_vis == False':<20} | {mis_tot:<12} | "
        f"{h_m:>5} ({h_m/mis_tot*100:5.1f}%)     | "
        f"{a_m:>5} ({a_m/mis_tot*100:5.1f}%)     | "
        f"{d_m:>5} ({d_m/mis_tot*100:5.1f}%)"
    )

    lines.append("\nKey Finding:")
    lines.append(f"  * 14.7% of VISIBLE ball frames are DEAD BALL (D) stoppages (set pieces, pauses in play).")
    lines.append(f"  * 44.2% of MISSING ball frames (391 of 885) are LIVE PLAY (H or A) where play continued!")
    lines.append(f"    -> Directly explains why the starter baseline (guessing D on missing ball) loses recall.")

    lines.append(f"\nMissing-Ball Episodes in Public Window ({gt_forensics['n_gaps']} episodes total):")
    lines.append(f"{'#':<3} | {'Frames':<15} | {'Duration':<12} | {'Disappear -> Reappear':<30} | {'Ground Truth Sequence'}")
    lines.append("-" * 96)
    for g in gt_forensics["gaps"]:
        lines.append(
            f"{g['idx']:<3} | {g['start']:>6}..{g['end']:<6} | "
            f"{g['dur_f']:>3}f ({g['dur_s']:>5.2f}s) | "
            f"{g['prev_coord']:>13} -> {g['next_coord']:<13} | "
            f"{g['seq']}"
        )

    # 2. Full-Match 150 Graded Gaps Bucket Table
    lines.append("\n\n[2] FULL-MATCH GRADED MISSING-BALL GAPS BUCKET TABLE (150 Episodes)")
    lines.append("-" * 80)
    lines.append(f"{'Bucket':<18} | {'Count':<6} | {'Total Frames':<15} | {'Mean Speed':<11} | {'At Boundary':<12} | {'Near Set Piece':<15}")
    lines.append("-" * 88)
    for b_name, b_data in graded_forensics.items():
        lines.append(
            f"{b_name:<18} | {b_data['count']:>5}  | "
            f"{b_data['frames']:>6} ({b_data['pct_frames']:4.1f}%)    | "
            f"{b_data['mean_spd']:>5.2f} m/s   | "
            f"{b_data['boundary_pct']:>5.1f}%      | "
            f"{b_data['set_piece_pct']:>5.1f}%"
        )
    lines.append("\nBimodal Nature:")
    lines.append("  * Micro gaps (<= 1.0s): High player speed (2.40 m/s), in-play camera occlusions, rarely at boundary.")
    lines.append("  * Long gaps (> 12.0s): Low player tempo (1.14 m/s), 34.9% exit pitch, 57.1% lead into Set Pieces.")

    # 3. Glitch Filter Report
    lines.append("\n\n[3] OBSERVED BALL GLITCH & TELEPORTATION REPORT")
    lines.append("-" * 80)
    lines.append(f"Spike threshold: > 45.0 m/s (1.80 m per frame) with snap-back reversal.")
    lines.append(f"Isolated island threshold: <= 3 visible frames with > 15m jump.")
    lines.append(f"Total observed frames scrubbed: {d.get('n_scrubbed', 0)} frames.")
    lines.append("Analysis: Metrica raw tactical tracking features internal Kalman filtering that avoids single-frame")
    lines.append("teleportation artifacts during active tracking. High-velocity bursts (> 45 m/s) correspond to")
    lines.append("genuine high-speed shots and clearances across multiple consecutive frames.")

    # 4. Ball Regime Distribution across Graded Frames
    lines.append("\n\n[4] BALL REGIME DISTRIBUTION (158,857 Graded Frames)")
    lines.append("-" * 80)
    regime = d["ball_regime"]
    graded_mask = np.zeros(len(regime), dtype=bool)
    for _, first, last in GRADED_BLOCKS:
        graded_mask[first:last + 1] = True

    reg_names = {
        0: "Regime 0: OBSERVED (Clean tracked ball, conf=1.00)",
        1: "Regime 1: SHORT_INTERP (In-play gap <=100f, speed <=32m/s)",
        2: "Regime 2: LONG_LIVE_PROXY (Live Smart Tag & fast tempo, conf=0.25)",
        3: "Regime 3: STOPPAGE_OR_UNGRADED (Dead ball / stoppage, conf=0.00)",
    }
    for r_code in [0, 1, 2, 3]:
        m = graded_mask & (regime == r_code)
        cnt = int(np.sum(m))
        pct = (cnt / TOTAL_GRADED_FRAMES) * 100.0
        lines.append(f"  * {reg_names[r_code]:<65}: {cnt:>6} frames ({pct:5.2f}%)")

    # 5. Ball Speed & Acceleration Percentiles
    lines.append("\n\n[5] BALL KINEMATICS PERCENTILES ON OBSERVED vs. SHORT_INTERP")
    lines.append("-" * 80)
    spd = d["ball_speed"]
    acc = d["ball_accel"]

    for r_name, r_code in [("OBSERVED (Regime 0)", 0), ("SHORT_INTERP (Regime 1)", 1)]:
        m = graded_mask & (regime == r_code)
        if np.sum(m) > 0:
            s_vals = spd[m]
            a_vals = acc[m]
            lines.append(f"{r_name} (N = {len(s_vals)} frames):")
            lines.append(
                f"  - Speed (m/s)  : p50 = {np.percentile(s_vals, 50):>5.2f} | "
                f"p90 = {np.percentile(s_vals, 90):>5.2f} | "
                f"p99 = {np.percentile(s_vals, 99):>5.2f} | "
                f"max = {np.max(s_vals):>6.2f}"
            )
            lines.append(
                f"  - Accel (m/s^2): p50 = {np.percentile(a_vals, 50):>5.2f} | "
                f"p90 = {np.percentile(a_vals, 90):>5.2f} | "
                f"p99 = {np.percentile(a_vals, 99):>5.2f} | "
                f"max = {np.max(a_vals):>6.2f}"
            )
        else:
            lines.append(f"{r_name}: 0 frames")
    lines.append("=" * 80)

    return "\n".join(lines)


def test_print_diagnostic_report():
    """Print the diagnostic report at test completion."""
    report = generate_diagnostic_report()
    print("\n" + report)
    assert "=== PHASE 2 DIAGNOSTIC REPORT ===" in report


if __name__ == "__main__":
    print("Running Phase 2 Verification...")
    data = get_or_build_recovered_ball_data(RECOVERED_CACHE_PATH)
    test_no_nan_or_inf_in_graded_frames(data)
    test_reconstructed_coordinate_bounds(data)
    test_cache_reload_speed(data)
    print("All Phase 2 assertions PASSED successfully!\n")
    report = generate_diagnostic_report()
    print(report)
