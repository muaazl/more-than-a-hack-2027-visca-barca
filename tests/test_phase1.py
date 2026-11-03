"""Phase 1 Verification Tests & Comprehensive Diagnostic Report.

Team: Visca Barca
Competition: More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)

Verifies:
1. Total frame count is exactly 199,319 (0 to 199318).
2. Graded half blocks from games.half_video_blocks("BAR-ATH") match [(1, 11185, 90651), (2, 110808, 190197)] (158,857 graded frames total).
3. Coordinate scaling check: valid ball_x and player x values fall within reasonable pitch bounds around [0, 105.3] and y around [0, 68.0].
4. Cache reload speed is under 2.0 seconds and produces identical arrays.
5. Emits the full === PHASE 1 DIAGNOSTIC REPORT ===.
"""

import os
import subprocess
import sys
import time
import numpy as np
import pytest

# Path resolution
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from games import GAME, half_video_blocks, source_frames
from src.data_loader import (
    N_VIDEO_FRAMES,
    PITCH_LENGTH,
    PITCH_WIDTH,
    GRADED_BLOCKS,
    TOTAL_GRADED_FRAMES,
    DEFAULT_CACHE_PATH,
    get_or_load_tactical_data,
    load_cached_tactical_data,
    load_public_ground_truth,
    get_team_slot_mapping,
)


@pytest.fixture(scope="module")
def tactical_data():
    """Load cached tactical data for all tests."""
    return get_or_load_tactical_data(DEFAULT_CACHE_PATH)


def test_frame_count(tactical_data):
    """Assertion 1: Total frame count is exactly 199,319 (0 to 199318)."""
    assert source_frames("tactical", GAME) == 199319
    assert N_VIDEO_FRAMES == 199319
    assert tactical_data["home_players"].shape[0] == 199319
    assert tactical_data["away_players"].shape[0] == 199319
    assert tactical_data["ball_pos"].shape[0] == 199319
    assert tactical_data["ball_vis"].shape[0] == 199319


def test_graded_half_blocks():
    """Assertion 2: Graded half blocks from games.half_video_blocks('BAR-ATH') match
    [(1, 11185, 90651), (2, 110808, 190197)] (158,857 graded frames total).
    """
    blocks = half_video_blocks(GAME)
    expected_blocks = [(1, 11185, 90651), (2, 110808, 190197)]
    assert blocks == expected_blocks

    total_graded = sum(last - first + 1 for _, first, last in blocks)
    assert total_graded == 158857
    assert TOTAL_GRADED_FRAMES == 158857


def test_coordinate_scaling(tactical_data):
    """Assertion 3: Valid ball_x and player x values fall within reasonable pitch bounds
    around [0, 105.3] and y around [0, 68.0].
    """
    ball_pos = tactical_data["ball_pos"]
    home_players = tactical_data["home_players"]
    away_players = tactical_data["away_players"]

    # Ball valid coordinates
    valid_ball = ~np.isnan(ball_pos[:, 0])
    bx = ball_pos[valid_ball, 0]
    by = ball_pos[valid_ball, 1]

    # Ball can travel into stands / bench area, so allow reasonable bounds around pitch
    # Pitch bounds are [0, 105.3] x [0, 68.0]
    assert np.min(bx) >= -15.0 and np.max(bx) <= 125.0
    assert np.min(by) >= -15.0 and np.max(by) <= 110.0

    # >95% of visible ball positions fall strictly within the pitch lines
    assert np.mean((bx >= 0.0) & (bx <= PITCH_LENGTH)) > 0.95
    assert np.mean((by >= 0.0) & (by <= PITCH_WIDTH)) > 0.95

    # Center pitch check: median ball coordinates should be near center pitch
    assert 45.0 <= np.median(bx) <= 60.0
    assert 30.0 <= np.median(by) <= 45.0

    # Player coordinates: >98% fall strictly within pitch lines
    valid_hx = home_players[..., 0][~np.isnan(home_players[..., 0])]
    valid_hy = home_players[..., 1][~np.isnan(home_players[..., 1])]
    valid_ax = away_players[..., 0][~np.isnan(away_players[..., 0])]
    valid_ay = away_players[..., 1][~np.isnan(away_players[..., 1])]

    assert np.mean((valid_hx >= 0.0) & (valid_hx <= PITCH_LENGTH)) > 0.98
    assert np.mean((valid_hy >= 0.0) & (valid_hy <= PITCH_WIDTH)) > 0.98
    assert np.mean((valid_ax >= 0.0) & (valid_ax <= PITCH_LENGTH)) > 0.98
    assert np.mean((valid_ay >= 0.0) & (valid_ay <= PITCH_WIDTH)) > 0.98

    # Extrema check for players (allowing bench / off-pitch warm-up areas)
    assert np.min(valid_hx) >= -10.0 and np.max(valid_hx) <= 115.0
    assert np.min(valid_hy) >= -10.0 and np.max(valid_hy) <= 80.0
    assert np.min(valid_ax) >= -10.0 and np.max(valid_ax) <= 115.0
    assert np.min(valid_ay) >= -10.0 and np.max(valid_ay) <= 80.0


def test_cache_reload_speed_and_fidelity(tactical_data):
    """Assertion 4: Cache reload speed is under 2.0 seconds and produces identical arrays."""
    t0 = time.time()
    reloaded = load_cached_tactical_data(DEFAULT_CACHE_PATH)
    reload_time = time.time() - t0

    assert reload_time < 2.0, f"Reload took {reload_time:.3f}s (exceeds 2.0s limit)"

    # Verify identical arrays
    np.testing.assert_allclose(tactical_data["home_players"], reloaded["home_players"], equal_nan=True)
    np.testing.assert_allclose(tactical_data["away_players"], reloaded["away_players"], equal_nan=True)
    np.testing.assert_allclose(tactical_data["ball_pos"], reloaded["ball_pos"], equal_nan=True)
    np.testing.assert_array_equal(tactical_data["ball_vis"], reloaded["ball_vis"])
    np.testing.assert_array_equal(tactical_data["smart_tagging"], reloaded["smart_tagging"])
    assert tactical_data["home_name"] == reloaded["home_name"]
    assert tactical_data["away_name"] == reloaded["away_name"]


def generate_diagnostic_report():
    """Build and format the full text block for === PHASE 1 DIAGNOSTIC REPORT ===."""
    data = get_or_load_tactical_data(DEFAULT_CACHE_PATH)

    lines = []
    lines.append("=" * 80)
    lines.append("=== PHASE 1 DIAGNOSTIC REPORT ===")
    lines.append("=" * 80)

    # 1. Official evaluation/run_evaluation.py output table
    lines.append("\n[1] STARTER BASELINES EVALUATION SCOREBOARD (evaluation/run_evaluation.py)")
    lines.append("-" * 80)
    eval_script = os.path.join(ROOT_DIR, "evaluation", "run_evaluation.py")
    res = subprocess.run(
        [sys.executable, eval_script, "--no-detail"],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    if res.returncode == 0:
        lines.append(res.stdout.strip())
    else:
        lines.append(f"Scorer error:\n{res.stderr.strip()}")

    # 2. Exact shapes and dtypes of all arrays saved in cache/tactical_data.npz
    lines.append("\n\n[2] CACHED ARRAYS SUMMARY (cache/tactical_data.npz)")
    lines.append("-" * 80)
    cache_stat = os.stat(DEFAULT_CACHE_PATH)
    cache_mb = cache_stat.st_size / (1024 * 1024)
    lines.append(f"Cache File Path: {os.path.relpath(DEFAULT_CACHE_PATH, ROOT_DIR)} ({cache_mb:.2f} MB)")
    lines.append(f"  * home_players : shape {data['home_players'].shape} | dtype {data['home_players'].dtype} | {data['home_players'].nbytes / (1024*1024):.1f} MB (Home slots 0..23)")
    lines.append(f"  * away_players : shape {data['away_players'].shape} | dtype {data['away_players'].dtype} | {data['away_players'].nbytes / (1024*1024):.1f} MB (Away slots 24..45)")
    lines.append(f"  * ball_pos     : shape {data['ball_pos'].shape}     | dtype {data['ball_pos'].dtype} | {data['ball_pos'].nbytes / (1024*1024):.2f} MB [x_m, y_m]")
    lines.append(f"  * ball_vis     : shape {data['ball_vis'].shape}        | dtype {data['ball_vis'].dtype}    | {data['ball_vis'].nbytes / 1024:.1f} KB boolean mask")
    lines.append(f"  * smart_tagging: shape {data['smart_tagging'].shape}          | structured dtype    | {len(data['smart_tagging'])} intervals")
    lines.append(f"  * pitch_dims   : {list(data['pitch_dims'])} metres [length, width]")
    lines.append(f"  * teams meta   : Home = '{data['home_name']}', Away = '{data['away_name']}'")

    # 3. How starter/load_atd.py identifies Home vs Away track slots
    meta_path = os.path.join(ROOT_DIR, "data", "tactical", "BAR-ATH_tactical_atd.xml")
    home_name, away_name, home_slots, away_slots = get_team_slot_mapping(meta_path)
    lines.append("\n\n[3] TRACK SLOT IDENTIFICATION & PARTITION")
    lines.append("-" * 80)
    lines.append(f"Metadata XML: {os.path.basename(meta_path)}")
    lines.append(f"Total track slots in ATD: {len(home_slots) + len(away_slots)}")
    lines.append(f"  - Home Team: '{home_name}' -> {len(home_slots)} slots: indices {min(home_slots)}..{max(home_slots)}")
    lines.append(f"  - Away Team: '{away_name}' -> {len(away_slots)} slots: indices {min(away_slots)}..{max(away_slots)}")
    lines.append("Note: Slots are anonymous tracklets rather than persistent individual player IDs.")

    # 4. Ball visibility statistics
    lines.append("\n\n[4] BALL VISIBILITY & GAP DISTRIBUTION")
    lines.append("-" * 80)
    vis = data["ball_vis"]
    total_frames = len(vis)
    total_vis = int(np.sum(vis))

    graded_vis = sum(int(np.sum(vis[first:last + 1])) for _, first, last in GRADED_BLOCKS)
    lines.append(f"Full Video (199,319 frames) : {total_vis:>6} visible frames ({100.0 * total_vis / total_frames:.2f}%) | {total_frames - total_vis:>6} missing ({100.0 * (total_frames - total_vis) / total_frames:.2f}%)")
    lines.append(f"Graded Play (158,857 frames): {graded_vis:>6} visible frames ({100.0 * graded_vis / TOTAL_GRADED_FRAMES:.2f}%) | {TOTAL_GRADED_FRAMES - graded_vis:>6} missing ({100.0 * (TOTAL_GRADED_FRAMES - graded_vis) / TOTAL_GRADED_FRAMES:.2f}%)")

    # Calculate gap lengths across graded play
    gaps = []
    for _, first, last in GRADED_BLOCKS:
        half_vis = vis[first:last + 1]
        cur_gap = 0
        for v in half_vis:
            if not v:
                cur_gap += 1
            elif cur_gap > 0:
                gaps.append(cur_gap)
                cur_gap = 0
        if cur_gap > 0:
            gaps.append(cur_gap)

    gaps = np.array(gaps)
    lines.append("\nMissing Ball Gap Statistics (Graded Play):")
    lines.append(f"  - Total gap occurrences : {len(gaps)}")
    lines.append(f"  - Median gap length     : {np.median(gaps):.1f} frames ({np.median(gaps) / 25.0:.2f} s)")
    lines.append(f"  - 90th percentile gap   : {np.percentile(gaps, 90):.1f} frames ({np.percentile(gaps, 90) / 25.0:.2f} s)")
    lines.append(f"  - Max gap length        : {np.max(gaps)} frames ({np.max(gaps) / 25.0:.2f} s = {np.max(gaps) / 25.0 / 60.0:.2f} min)")

    # 5. Smart Tagging summary
    tagging = data["smart_tagging"]
    lines.append("\n\n[5] SMART TAGGING SUMMARY")
    lines.append("-" * 80)
    codes, code_counts = np.unique(tagging["code"], return_counts=True)
    sorted_idx = np.argsort(-code_counts)
    lines.append("Codes Distribution (638 total intervals):")
    for idx in sorted_idx:
        lines.append(f"  - {codes[idx]:<30}: {code_counts[idx]:>3}")

    teams, team_counts = np.unique(tagging["team"], return_counts=True)
    lines.append(f"\nNormalized Team Tags: {dict(zip(teams, team_counts))}")
    lines.append("Raw XML Club Strings: 'Barcelona' (314), 'Atletic Club' (288), 'N/A' (36)")

    set_pieces = [row for row in tagging if row["code"] == "SET PIECES"]
    lines.append(f"\nTotal SET PIECES Intervals: {len(set_pieces)}")
    lines.append("First 5 SET PIECES Intervals:")
    for idx, sp in enumerate(set_pieces[:5], start=1):
        lines.append(
            f"  {idx}. Frames {sp['start_frame']:>6}..{sp['end_frame']:>6} "
            f"({sp['start_s']:>6.1f}s..{sp['end_s']:>6.1f}s) | "
            f"Team: {sp['team']:<3} | Type: {sp['type']:<22} | Side: {sp['side'] or 'N/A'}"
        )

    # 6. Warnings, edge cases & anomalies observed
    lines.append("\n\n[6] OBSERVED WARNINGS, EDGE CASES & ANOMALIES")
    lines.append("-" * 80)
    lines.append("1. Coordinate Systems & Offsets:")
    lines.append("   - ATD file uses 1-based indexing ('1..199319') whereas video frame indices are 0-based ('0..199318').")
    lines.append("   - Correct conversion: video_frame = atd_frame_id - 1. Halves-specific offsets in games.py apply ONLY to the private ground truth.")
    lines.append("2. High Missing-Ball Rate:")
    lines.append("   - The ball is unobserved in 40.69% of graded frames and 43.06% of total match frames.")
    lines.append("   - Max consecutive unobserved stretch reaches 5,537 frames (~221.5 seconds) during long dead-ball / VAR stoppages.")
    lines.append("   - Naive nearest-player heuristics (baseline.py) fail completely during these missing intervals by guessing NOBODY ('D').")
    lines.append("3. Track Slot Swapping & Fragmentation:")
    lines.append("   - There are 46 track slots for 22 on-pitch players; player tracklets break and switch slots frequently.")
    lines.append("4. Tie-Breaking Bias in Starter Baseline:")
    lines.append("   - starter/baseline.py breaks player distance ties in favor of Home ('H'), creating asymmetric bias.")
    lines.append("5. Spelling Consistency:")
    lines.append("   - ATD metadata and Smart Tagging XML spell the away team as 'Atletic Club' (missing 'h'), which normalizes cleanly to 'A'.")
    lines.append("=" * 80)

    report_str = "\n".join(lines)
    return report_str


def test_print_diagnostic_report():
    """Run and print the diagnostic report."""
    report = generate_diagnostic_report()
    print("\n" + report)
    assert "=== PHASE 1 DIAGNOSTIC REPORT ===" in report


if __name__ == "__main__":
    # If run directly as a script, execute verification and display report
    print("Running Phase 1 Verification...")
    data = get_or_load_tactical_data(DEFAULT_CACHE_PATH)
    test_frame_count(data)
    test_graded_half_blocks()
    test_coordinate_scaling(data)
    test_cache_reload_speed_and_fidelity(data)
    print("All assertions PASSED successfully!\n")
    report = generate_diagnostic_report()
    print(report)
