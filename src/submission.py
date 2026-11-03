"""Phase 6: Submission Exporter & Validator.

Team: Visca Barca
Competition: More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)

Generates:
  evaluation/results/Visca_Barca.csv
"""

import hashlib
import os
import sys
import time
import numpy as np

# Path resolution
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.features import get_or_build_features
from src.model import compute_frame_likelihoods
from src.postprocessing import generate_full_match_predictions

SUBMISSION_FILENAME = "Visca_Barca.csv"
SUBMISSION_PATH = os.path.join(ROOT_DIR, "evaluation", "results", SUBMISSION_FILENAME)


def compute_file_sha256(filepath: str) -> str:
    """Compute hex SHA-256 hash of a file for integrity and determinism checking."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def write_submission_csv(filepath: str, labels: np.ndarray) -> str:
    """Write frame,label CSV strictly matching official submission format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        f.write("frame,label\n")
        f.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return filepath


def generate_submission(output_path: str = SUBMISSION_PATH, verbose: bool = True) -> str:
    """End-to-end possession inference and submission file generation.
    
    Returns:
        output_path: Path to the generated CSV file.
    """
    t0 = time.time()
    if verbose:
        print(f"=== Generating Visca Barca Possession Submission ===")
        print("1/3 Loading features and computing physical log-likelihoods...")
        
    features = get_or_build_features()
    likelihoods = compute_frame_likelihoods(features)

    if verbose:
        print("2/3 Executing Viterbi dynamic programming & morphology filtering...")
    predictions = generate_full_match_predictions(likelihoods, features)

    if verbose:
        print(f"3/3 Exporting {len(predictions)} rows to {os.path.relpath(output_path, ROOT_DIR)}...")
    write_submission_csv(output_path, predictions)
    
    t1 = time.time()
    sha256_hash = compute_file_sha256(output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    if verbose:
        print(f"\nSubmission generated successfully in {t1 - t0:.2f}s!")
        print(f"File: {os.path.relpath(output_path, ROOT_DIR)}")
        print(f"Size: {file_size_mb:.2f} MB")
        print(f"SHA-256: {sha256_hash}")
        
        # Summary distribution
        u, c = np.unique(predictions, return_counts=True)
        print("\nLabel distribution:")
        for label, count in zip(u, c):
            pct = 100.0 * count / len(predictions)
            print(f"  {label}: {count:>6} frames ({pct:5.2f}%)")

    return output_path


if __name__ == "__main__":
    generate_submission()
