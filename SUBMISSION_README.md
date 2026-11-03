# Team Visca Barca — Submission & Reproduction Guide

## Fast Reproduction (Single Command)

To reproduce the exact submission file (`Visca_Barca.csv` and `evaluation/results/Visca_Barca.csv`) from raw data:

```bash
# 1. Activate environment
.venv\Scripts\activate       # Windows (or source .venv/bin/activate on Linux/macOS)

# 2. Run end-to-end pipeline
python run_pipeline.py
```

### Verification & Testing
```bash
# Run automated data-only UAT suite
python tests/run_uat_suite.py

# Run official evaluation against public 5-minute ground truth
python evaluation/run_evaluation.py --bootstrap 1000

# Run full pytest suite across all phases
pytest
```

## Key Submission Assets
- **Submission CSV:** `Visca_Barca.csv` (and `evaluation/results/Visca_Barca.csv`)
  - **SHA-256:** `63abd382d1a232a99be8a6d0906c14d1519fcb298f44b48adb315d993f5346b3`
  - **Rows:** 199,319
- **Technical Report:** `Visca_Barca.md`
- **UAT Report:** `reports/UAT_Report.md`
