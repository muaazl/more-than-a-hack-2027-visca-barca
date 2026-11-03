"""One-Command Reproducible End-to-End Pipeline for Team Visca Barca.

More Than A Hack 2027: FC Barcelona vs Athletic Club Ball Possession Challenge.
Runs Phases 1 through 5 deterministically to produce the official submission.

Usage:
    python run_pipeline.py
    python run_pipeline.py --data-dir ./data --output Visca_Barca.csv --rebuild-cache
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import numpy as np

# Path setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import get_or_load_tactical_data, DEFAULT_CACHE_PATH
from src.ball_recovery import get_or_build_recovered_ball_data, RECOVERED_CACHE_PATH
from src.features import get_or_build_features, FEATURES_CACHE_PATH
from src.model import compute_frame_likelihoods
from src.postprocessing import generate_full_match_predictions
from src.submission import compute_file_sha256, write_submission_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Run Team Visca Barca End-to-End Possession Pipeline")
    parser.add_argument("--data-dir", default=os.path.join(ROOT_DIR, "data"),
                        help="Path to match raw data directory (default: ./data)")
    parser.add_argument("--output", default="Visca_Barca.csv",
                        help="Filename for root submission export (default: Visca_Barca.csv)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuilding of all intermediate caches (tactical, ball, features)")
    return parser.parse_args()


def main():
    args = parse_args()
    start_total_time = time.time()
    
    # Enforce deterministic execution
    np.random.seed(42)

    print("=" * 80)
    print("TEAM VISCA BARCA - OFFICIAL REPRODUCIBLE INFERENCE PIPELINE")
    print("More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)")
    print("=" * 80)
    print(f"Working Directory : {ROOT_DIR}")
    print(f"Data Directory    : {args.data_dir}")
    print(f"Root Output Target: {args.output}")
    print(f"Rebuild Cache     : {args.rebuild_cache}")
    print("-" * 80)

    # 1. Phase 1: Tactical ATD Tracking & Smart Tagging Cache
    t0 = time.time()
    print(">>> [Phase 1/5] Loading / Validating Tactical Tracking Data...")
    tactical_data = get_or_load_tactical_data(
        cache_path=DEFAULT_CACHE_PATH,
        force_reload=args.rebuild_cache
    )
    print(f"    Loaded {len(tactical_data['ball_pos'])} frames of tracking in {time.time() - t0:.2f}s")

    # 2. Phase 2: Ball Recovery & Kinematics Engine
    t0 = time.time()
    print(">>> [Phase 2/5] Recovering Ball Kinematics & Regime Classification...")
    recovered_ball = get_or_build_recovered_ball_data(
        cache_path=RECOVERED_CACHE_PATH,
        force_reload=args.rebuild_cache
    )
    print(f"    Recovered clean trajectory ({len(recovered_ball['ball_pos_rec'])} frames) in {time.time() - t0:.2f}s")

    # 3. Phase 3: Vectorized Feature Engine (55 channels)
    t0 = time.time()
    print(">>> [Phase 3/5] Extracting 55 Physical & Tactical Feature Channels...")
    features = get_or_build_features(
        cache_path=FEATURES_CACHE_PATH,
        force_reload=args.rebuild_cache
    )
    print(f"    Loaded 55 feature channels in {time.time() - t0:.2f}s")

    # 4. Phase 4: Physically Grounded Bayesian Likelihoods
    t0 = time.time()
    print(">>> [Phase 4/5] Computing A Priori Likelihoods & Law 3 Stoppage Windows...")
    likelihoods = compute_frame_likelihoods(features)
    print(f"    Computed normalized 3-class likelihoods in {time.time() - t0:.2f}s")

    # 5. Phase 5: Viterbi DP & Laws 1, 2, 3 Football Morphology
    t0 = time.time()
    print(">>> [Phase 5/5] Executing Viterbi DP, Law 1 Contact Gate & Law 2 Control Filters...")
    predictions = generate_full_match_predictions(likelihoods, features)
    print(f"    Generated {len(predictions)} final frame predictions in {time.time() - t0:.2f}s")

    # 6. Export Deliverables
    print("-" * 80)
    root_csv_path = os.path.join(ROOT_DIR, args.output)
    eval_csv_path = os.path.join(ROOT_DIR, "evaluation", "results", "Visca_Barca.csv")

    print(f"Writing root submission: {os.path.relpath(root_csv_path, ROOT_DIR)} ...")
    write_submission_csv(root_csv_path, predictions)

    print(f"Writing evaluation submission: {os.path.relpath(eval_csv_path, ROOT_DIR)} ...")
    write_submission_csv(eval_csv_path, predictions)

    total_duration = time.time() - start_total_time
    root_sha256 = compute_file_sha256(root_csv_path)
    eval_sha256 = compute_file_sha256(eval_csv_path)
    assert root_sha256 == eval_sha256, "Mismatch between root and evaluation CSV files!"

    print("-" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Total Runtime: {total_duration:.2f} seconds")
    print(f"Submission SHA-256 Checksum: {root_sha256}")
    print("-" * 80)

    # 7. Official Submission Format Validation
    print("Running official format validator on generated submission...")
    validate_script = os.path.join(ROOT_DIR, "evaluation", "validate_submission.py")
    subprocess.run([sys.executable, validate_script, root_csv_path], check=True)
    print("=" * 80)


if __name__ == "__main__":
    main()
