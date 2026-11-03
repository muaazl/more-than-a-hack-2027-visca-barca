# Team "Visca Barca" — Technical Report
**Challenge:** More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)  
**Authors:** Team Visca Barca  
**Submission Deliverable:** `Visca_Barca.csv` (SHA-256: `63abd382d1a232a99be8a6d0906c14d1519fcb298f44b48adb315d993f5346b3`)  
**Pipeline Execution:** `python run_pipeline.py` (Runtime: ~2.5 seconds)

---

## 1. Executive Summary & Approach

### 1.1 Problem Formulation
The challenge requires labeling ball possession across all **199,319 tactical video frames** (0-indexed `0..199318`, 25.0 fps, 132.88 minutes) of the match between FC Barcelona (`H`) and Athletic Club (`A`) using the alphabet `{H, A, D, X}`. Exactly 158,857 frames (105.90 minutes) across both halves are evaluated by the official scorer.

The competition scoreboard is ranked strictly on **`spellF1`** (the mean possession-spell F1 over IoU thresholds `0.3, 0.4, 0.5, 0.6, 0.7`), with ties broken by **`macroF1`** over `{H, A, D}`. As demonstrated by the organizers in `EVALUATION.md`, per-frame classifiers and naive windowed smoothers suffer from fatal structural trade-offs: over-smoothing invents fictitious long possessions that lag transitions, while raw nearest-player heuristics fragment single possession sequences into hundreds of spurious 1–2 second micro-spells.

### 1.2 Our Architecture: Physical Bayesian State Machine with Dynamic Programming
Team Visca Barca developed a **Physically Grounded Bayesian State Machine with Viterbi Dynamic Programming and Football Morphology Filtering**. Rather than treating possession inference as independent per-frame image classification or black-box machine learning, our approach models the physical laws of association football:
1. **Regime-Aware Ball Trajectory Reconstruction:** 40.69% of graded frames lack raw ball coordinates. We recover ball kinematics by partitioning missing stretches into 4 distinct physical regimes (Observed, Micro-Gaps, Medium Live Play Gaps, and Stoppage Gaps), removing tracker teleportation glitches ($> 42.0\text{ m/s}$), and computing smoothed velocity and acceleration.
2. **55 Vectorized Multi-Scale Physical Channels:** Vectorized proximity metrics, spatial density, exponential soft-control envelopes, dead-ball physical discriminators (pitch boundary distances, stationary ball detectors, collective tempo percentiles), and tactical priors from Smart Tagging.
3. **A Priori Bayesian Likelihood Formulation:** Frame-level emission probabilities derived from first-principles kinematics and FIFA pitch geometry, with zero supervised tuning on ground truth.
4. **Three Domain Laws from `CHALLENGE.md` & `DATA.md`:**
   - **Law 1 (Causal Loose-Ball Retention & Physical Contact Gate):** During live-play turnover handovers ($H \leftrightarrow A$), transition boundaries are held with the prior possessing team while the ball is in flight until the incoming team's closest player comes within physical control reach ($d_{\text{incoming}} \le 2.2\text{m}$).
   - **Law 2 (Uncontested Control Verification for Sandwich Spells):** Absorbs 1.0s–2.5s aerial duel headers, clearances, and deflections without sustained control unless the contending team establishes uncontested control ($\ge 25\text{ frames}$ of $d_{T2} \le 1.8\text{m}$ and $d_{T1} \ge 2.8\text{m}$).
   - **Law 3 (Set-Piece Pre-Kick Stoppage Window):** Enforces dead-ball stoppage (`D`) throughout the pre-kick preparation window of Smart Tagging set pieces, transitioning cleanly to live play at the physical restart frame ($v_{\text{ball}} > 3.0\text{ m/s}$).
5. **Exact Boundary Alignment:** Zero abstentions (`X`) on graded frames; exact masking of pre-match, halftime, and post-match stretches as `X`; and guaranteed dead-ball classification (`D`) for the pre-2nd-half kickoff stretch (`110808..113741`).

---

## 2. Assets Used & Dropped

| Asset | File | Status | Usage / Reason for Dropping |
|:---|:---|:---:|:---|
| **Tactical ATD Tracking** | `BAR-ATH_tactical_atd.txt` / `.xml` | **USED** | Core positional and kinematic tracking for all 46 player track slots and the ball across all 199,319 frames. Transformed from 0–1 coordinates to metric pitch coordinates ($105.3\text{m} \times 68.0\text{m}$). |
| **Tactical Smart Tagging** | `BAR-ATH_tactical_smart_tagging.xml` | **USED** | Parsed all 638 tactical intervals (413 set pieces, 178 defensive blocks/transitions, 47 attacking transitions). Used for polarity-aligned tactical priors and set-piece restart boundaries. |
| **Public Ground Truth** | `BAR-ATH_possession_gt_public.csv` | **DIAGNOSTIC ONLY** | Shipped 5-minute slice (frames 11185..18684). Used strictly for read-only pipeline validation (`run_evaluation.py`). Never used for training, fitting, or hyperparameter sweeps. |
| **Broadcast ATD Tracking** | `BAR-ATH_broadcast_atd.txt` / `.xml` | **DROPPED** | Unaligned coordinate timeline with frequent panning cuts; redundant with continuous tactical video tracking. |
| **Broadcast Homography** | `BAR-ATH_broadcast_homography.csv` | **DROPPED** | Relates to broadcast camera pans, not needed for 2D tactical pitch coordinates. |
| **Tactical Homography** | `BAR-ATH_tactical_homography.csv` | **DROPPED** | ATD tracking already provided metric-normalised coordinates, making raw pixel projection unnecessary. |
| **Raw Tactical Video** | `BAR-ATH_tactical.mp4` | **DROPPED** | Per organizers' clarification, visual video inspection is prohibited for tuning. Decoding 199,319 H.264 video frames adds hours of GPU compute with zero accuracy gain over vector ATD coordinates. |

---

## 3. Training Data & Regulatory Compliance Statement

### 3.1 Certification of Compliance with `RULES.md`
Team Visca Barca certifies under the regulations in `RULES.md` and the organizers' clarification email that:
1. **Zero External Labeled Datasets:** No human-labeled or external event data (Opta, StatsBomb, Wyscout, etc.) was used.
2. **Zero Ground Truth Fitting:** Neither the 5-minute public ground truth nor any private ground truth was used to fit model parameters, train weights, or run hyperparameter search grids.
3. **Zero Output Tuning Against Video:** Predictions were never visually compared against the match video to tune constants or rules.
4. **All Parameters A Priori or Unsupervised:** Every parameter in `src/model.py` and `src/postprocessing.py` was derived either from first-principles football kinematics/geometry or from unsupervised match-wide percentiles.

### 3.2 Specification of Physical & Unsupervised Parameters

| Parameter | Value | Derivation Source | Physical Justification |
|:---|:---:|:---|:---|
| `PITCH_LENGTH_M` | $105.3\text{ m}$ | `games.py` / FIFA Regulations | Exact length of Camp Nou / Olympic Stadium pitch. |
| `PITCH_WIDTH_M` | $68.0\text{ m}$ | `games.py` / FIFA Regulations | Exact width of pitch. |
| `PHYSICAL_CONTROL_RADIUS_M` | $1.8\text{ m}$ | Human Biomechanics | Maximum human leg reach + reaction envelope for foot control. |
| `DUEL_PRESSURE_RADIUS_M` | $3.0\text{ m}$ | Football Tactical Theory | Envelope within which an opponent exerts direct contested pressing. |
| `PASS_FLIGHT_MIN_SPEED_MPS` | $5.0\text{ m/s}$ | Ball Kinematics | Velocity above which a kicked ball travels in flight rather than trapped at feet. |
| `BALL_STATIONARY_MAX_SPEED_MPS` | $1.0\text{ m/s}$ | Sensor Noise Floor | Velocity below which a tracked ball is physically stationary on the pitch. |
| `SET_PIECE_RESTART_SPEED` | $3.0\text{ m/s}$ | Ball Kinematics | Threshold indicating a physical restart kick into live play from a set piece. |
| `dead_tempo_p20` | $1.63\text{ m/s}$ | Unsupervised Percentile | 20th percentile of collective player speed across graded halves (walk/jog pace). |
| `MIN_CONTROL_DWELL_FRAMES` | $35\text{ frames}$ ($1.4\text{s}$) | `CHALLENGE.md` Domain Definition | Minimum duration for controlled possession vs momentary touch/deflection. |
| `MAX_SANDWICH_FRAMES` | $65\text{ frames}$ ($2.6\text{s}$) | `CHALLENGE.md` ("a few seconds") | Window for checking uncontested control in sandwiched duel touches. |
| `MIN_UNCONTESTED_FRAMES` | $25\text{ frames}$ ($1.0\text{s}$) | `CHALLENGE.md` | Minimum uncontested control ($d \le 1.8\text{m}, d_{\text{opp}} \ge 2.8\text{m}$) to validate turnover. |
| $A_{ii}$ (Self-Transition Prob) | $0.995$ | Football Possession Dynamics | $\frac{1}{1 - 0.995} = 200\text{ frames} = 8.0\text{s}$ expected base Markov dwell time. |

---

## 4. Possession Rules & Missing-Ball Recovery Implemented

### 4.1 Missing-Ball Recovery Engine (`src/ball_recovery.py`)
In raw ATD tracking, 40.69% of graded frames have unobserved ball coordinates. A naive assumption that missing ball implies dead ball severely degrades spell continuity. We partition all missing gaps into 4 physical regimes:
- **Regime 0 (Observed):** Ball is tracked. Glitch filter removes unphysical tracker teleportation spikes ($> 42.0\text{ m/s}$) and clamps velocities to $[0, 40]\text{ m/s}$ and accelerations to $[-50, 50]\text{ m/s}^2$.
- **Regime 1 (Micro-Gaps, $\le 30$ frames / $1.2\text{s}$):** Tracker dropouts during open play; smoothly interpolated linearly between valid endpoints.
- **Regime 2 (Medium Live-Play Gaps, $31..300$ frames / $1.2..12.0\text{s}$):** High-speed clearances, aerial crosses, or swarm occlusions; interpolated and supplemented by endpoint proximity diffs.
- **Regime 3 (Long Stoppage Gaps, $> 300$ frames / $> 12.0\text{s}$):** Out-of-play ball retrieval or medical stoppages; coupled with `gap_exit_oob` indicators to confirm dead ball.

### 4.2 Causal Loose-Ball Retention (Law 1)
In football, when Team H kicks a pass towards Team A, the ball remains in Team H's possession while in flight. Naive multi-frame centered smoothing leaks the eventual receiver backward in time, creating premature handovers. Our **Physical Contact Gate** delays any live-play handover until $d_{\text{incoming}} \le 2.2\text{m}$ (or ball occlusion), strictly carrying forward prior possession while $d > 2.2\text{m}$.

### 4.3 Deflection and Aerial-Duel Verification (Law 2)
Per `CHALLENGE.md`, deflections, headers won without control, and brief touches do not count as a change of possession. Any live spell $S$ of Team $T_2$ sandwiched between Team $T_1$ with duration $< 65$ frames ($2.6\text{s}$) is absorbed into $T_1$ unless $T_2$ establishes uncontested control ($d_{T2} \le 1.8\text{m}$ and $d_{T1} \ge 2.8\text{m}$ for $\ge 25$ frames).

### 4.4 Set-Piece Restart Alignment (Law 3)
Across all 59 Smart Tagging set-piece intervals, we compute the exact physical restart frame $k_f$ (speed spike $> 3.0\text{ m/s}$). We enforce dead ball (`D`) throughout the pre-kick setup window (including preceding stationary-ball periods), and transition crisply to the restarting team at $k_f$.

---

## 5. Limitations & Future Work

1. **Crowded Penalty-Box Scrambles During Ball Occlusions:** In multi-player goal-mouth scrambles where 8+ players from both teams converge within 2 meters of the ball while the camera loses the ball, 2D proximity differential is nearly zero. A 3D physical bounce model using optical camera calibration could better resolve close-range deflections.
2. **Dynamic Tactical Formations:** While Smart Tagging provides macro-phase intervals (defending blocks, transitions), incorporating dynamic team centroids and Voronoi pitch control partitioning could improve pass destination prediction during counter-attacks.

---

## 6. Exact Reproduction & UAT Instructions

### 6.1 Environment Setup
```bash
# Recommended: Python 3.11 with uv virtual environment
uv venv .venv
.venv\Scripts\activate      # Windows (or source .venv/bin/activate on Linux/macOS)
uv pip install -r requirements.txt -r requirements-starter.txt scipy scikit-learn pytest
```

### 6.2 Single-Command Pipeline Execution
To reproduce the exact submission file from raw data:
```bash
python run_pipeline.py
```
- **Runtime:** ~2.5 seconds on modern multi-core CPU (uses fast compressed NumPy caches).
- **Target Outputs:**
  - `Visca_Barca.csv` (repository root, for upload)
  - `evaluation/results/Visca_Barca.csv` (for evaluation suite)
- **Expected SHA-256 Checksum:**
  `63abd382d1a232a99be8a6d0906c14d1519fcb298f44b48adb315d993f5346b3`

### 6.3 Automated UAT & Scorer Validation
```bash
python tests/run_uat_suite.py
python evaluation/run_evaluation.py --bootstrap 1000
```
- Validates all contract invariants, physical football duration bounds, determinism, and scores against public ground truth.
