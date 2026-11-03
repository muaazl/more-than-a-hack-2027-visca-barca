"""Test suite for Phase 4, Phase 5, and Phase 6.

Automated Data-Only Quality Assurance (QA) and Pipeline Verification.
Team: Visca Barca
"""

import hashlib
import os
import sys
import numpy as np
import pytest

# Ensure root directory is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import (
    N_VIDEO_FRAMES,
    GRADED_BLOCKS,
    load_cached_tactical_data,
)
from src.features import load_features_data
from src.model import compute_frame_likelihoods
from src.postprocessing import (
    viterbi_decode_segment,
    absorb_deflections_and_rebounds,
    generate_full_match_predictions,
    CODE_H, CODE_A, CODE_D,
)
from src.submission import (
    generate_submission,
    compute_file_sha256,
    SUBMISSION_PATH,
)


def test_model_likelihoods_integrity():
    """Verify log-likelihood array dimensions, finite values, and valid softmax properties."""
    feats = load_features_data()
    log_lik = compute_frame_likelihoods(feats)

    assert log_lik.shape == (N_VIDEO_FRAMES, 3), f"Expected shape ({N_VIDEO_FRAMES}, 3), got {log_lik.shape}"
    assert log_lik.dtype == np.float32, f"Expected float32 dtype, got {log_lik.dtype}"
    assert not np.any(np.isnan(log_lik)), "Found NaN values in log-likelihoods"
    assert not np.any(np.isinf(log_lik)), "Found Inf values in log-likelihoods"

    # Softmax probabilities should sum to 1.0 along axis 1
    probs = np.exp(log_lik)
    sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-4, err_msg="Likelihoods do not normalize to 1.0")


def test_deflection_and_micro_stoppage_absorption():
    """Verify CHALLENGE.md deflection absorption and sensor flicker bridging."""
    # 1. Deflection: A(100) -> brief H(10) -> A(100) should absorb H into A
    seq = np.array([CODE_A] * 100 + [CODE_H] * 10 + [CODE_A] * 100, dtype=np.int8)
    clean = absorb_deflections_and_rebounds(seq, min_dwell_frames=25, min_dead_frames=12)
    assert np.all(clean == CODE_A), "Brief opponent touch was not absorbed into controlling team"

    # 2. Deflection: H(100) -> brief A(15) -> H(100) should absorb A into H
    seq2 = np.array([CODE_H] * 100 + [CODE_A] * 15 + [CODE_H] * 100, dtype=np.int8)
    clean2 = absorb_deflections_and_rebounds(seq2, min_dwell_frames=25, min_dead_frames=12)
    assert np.all(clean2 == CODE_H), "Brief opponent touch was not absorbed into controlling team"

    # 3. Micro-stoppage: H(100) -> brief D(8) -> H(100) should bridge D into H
    seq3 = np.array([CODE_H] * 100 + [CODE_D] * 8 + [CODE_H] * 100, dtype=np.int8)
    clean3 = absorb_deflections_and_rebounds(seq3, min_dwell_frames=25, min_dead_frames=12)
    assert np.all(clean3 == CODE_H), "Brief sensor dropout was not bridged"

    # 4. Legitimate turnover: A(100) -> sustained H(35) -> A(100) should NOT be absorbed
    seq4 = np.array([CODE_A] * 100 + [CODE_H] * 35 + [CODE_A] * 100, dtype=np.int8)
    clean4 = absorb_deflections_and_rebounds(seq4, min_dwell_frames=25, min_dead_frames=12)
    assert np.sum(clean4 == CODE_H) == 35, "Sustained controlled possession was incorrectly absorbed"


def test_frame_masking_and_label_alphabet():
    """Verify exact 199,319 frame timeline alignment and strict label sets."""
    feats = load_features_data()
    log_lik = compute_frame_likelihoods(feats)
    preds = generate_full_match_predictions(log_lik, feats)

    assert len(preds) == N_VIDEO_FRAMES, f"Expected {N_VIDEO_FRAMES} frames, got {len(preds)}"
    unique_labels = set(np.unique(preds))
    assert unique_labels.issubset({"H", "A", "D", "X"}), f"Invalid labels found: {unique_labels}"

    # Ungraded stretches must be strictly 'X'
    assert np.all(preds[0:11185] == "X"), "Frames before 1st half kickoff must be 'X'"
    assert np.all(preds[90652:110808] == "X"), "Halftime frames must be 'X'"
    assert np.all(preds[190198:199319] == "X"), "Frames after full-time whistle must be 'X'"

    # Pre-2nd-half kickoff window must be strictly 'D'
    assert np.all(preds[110808:113742] == "D"), "Pre-2nd-half kickoff frames must be 'D'"

    # Graded live play halves must contain only {H, A, D} (no 'X' on graded frames)
    h1_graded = preds[11185:90652]
    h2_graded_live = preds[113742:190198]
    assert "X" not in set(np.unique(h1_graded)), "Found abstention 'X' on graded 1st half frames"
    assert "X" not in set(np.unique(h2_graded_live)), "Found abstention 'X' on graded 2nd half frames"


def test_data_only_qa_checks():
    """Automated Data-Only QA verifying possession metrics and tactical consistency."""
    feats = load_features_data()
    log_lik = compute_frame_likelihoods(feats)
    preds = generate_full_match_predictions(log_lik, feats)

    tactical = load_cached_tactical_data()
    st = tactical["smart_tagging"]
    sp_intervals = st[st["code"] == "SET PIECES"]

    # 1. Total possession counts and duration distributions per half
    for half_num, (h_s, h_e) in enumerate([(11185, 90651), (113742, 190197)], start=1):
        half_preds = preds[h_s:h_e + 1]
        
        # Calculate spells of H and A
        is_alive = (half_preds == "H") | (half_preds == "A")
        changes = np.where(half_preds[1:] != half_preds[:-1])[0] + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, len(half_preds)]
        
        spells_s = []
        for s, e in zip(starts, ends):
            lbl = half_preds[s]
            if lbl in ("H", "A"):
                spells_s.append((e - s) / 25.0) # duration in seconds
                
        assert len(spells_s) > 0, f"Half {half_num} has no possession spells"
        spells_arr = np.array(spells_s)
        mean_dur = float(np.mean(spells_arr))
        median_dur = float(np.median(spells_arr))
        
        # Professional football checks: spell count between 50 and 300, mean duration 5s - 25s
        assert 50 <= len(spells_arr) <= 300, f"Half {half_num} possession count {len(spells_arr)} outside realistic range"
        assert 5.0 <= mean_dur <= 25.0, f"Half {half_num} mean duration {mean_dur:.1f}s outside realistic range"
        assert 3.0 <= median_dur <= 20.0, f"Half {half_num} median duration {median_dur:.1f}s outside realistic range"

    # 2. Percentage of Smart Tagging SET PIECES preceded by a predicted D stoppage
    # For every set piece interval, verify the stoppage state D was present
    preceding_d_count = 0
    total_sp = 0
    for sp in sp_intervals:
        s_f = int(sp["start_frame"])
        e_f = int(sp["end_frame"])
        # Check if inside graded halves
        if (11185 <= s_f <= 90651) or (113742 <= s_f <= 190197):
            total_sp += 1
            # Check if there is at least one D frame inside the set-piece preparation window
            window_preds = preds[s_f:e_f + 1]
            if "D" in window_preds:
                preceding_d_count += 1
                
    sp_d_pct = 100.0 * preceding_d_count / total_sp
    assert sp_d_pct >= 90.0, f"Set piece stoppage alignment {sp_d_pct:.1f}% below 90% threshold"

    # 3. Percentage of out-of-bounds missing-ball gaps classified as D
    gap_exit_oob = feats["gap_exit_oob"] > 0.5
    current_gap_len = feats["current_gap_len"]
    # Gaps > 5s (125 frames) that started near pitch boundary
    oob_long_gaps = gap_exit_oob & (current_gap_len > 125)
    
    # Filter to graded halves
    graded_mask = np.zeros(N_VIDEO_FRAMES, dtype=bool)
    graded_mask[11185:90652] = True
    graded_mask[110808:190198] = True
    
    oob_eval_mask = oob_long_gaps & graded_mask
    if np.sum(oob_eval_mask) > 0:
        oob_d_pct = 100.0 * np.sum(preds[oob_eval_mask] == "D") / np.sum(oob_eval_mask)
        assert oob_d_pct >= 95.0, f"Out-of-bounds missing gap D rate {oob_d_pct:.1f}% below 95% threshold"


def test_determinism_and_sha256():
    """Verify running the pipeline twice produces bit-for-bit identical outputs."""
    tmp_path1 = os.path.join(ROOT_DIR, "cache", "tmp_run1.csv")
    tmp_path2 = os.path.join(ROOT_DIR, "cache", "tmp_run2.csv")

    feats = load_features_data()
    
    # Run 1
    ll1 = compute_frame_likelihoods(feats)
    p1 = generate_full_match_predictions(ll1, feats)
    with open(tmp_path1, "w", newline="") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(p1))
    hash1 = compute_file_sha256(tmp_path1)

    # Run 2
    ll2 = compute_frame_likelihoods(feats)
    p2 = generate_full_match_predictions(ll2, feats)
    with open(tmp_path2, "w", newline="") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(p2))
    hash2 = compute_file_sha256(tmp_path2)

    # Clean up tmp files
    if os.path.isfile(tmp_path1):
        os.remove(tmp_path1)
    if os.path.isfile(tmp_path2):
        os.remove(tmp_path2)

    assert hash1 == hash2, f"Pipeline is non-deterministic! Hash1: {hash1}, Hash2: {hash2}"
