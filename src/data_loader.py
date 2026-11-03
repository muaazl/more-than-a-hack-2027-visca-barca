"""Fast NumPy data loader and caching pipeline for tactical match data.

Team: Visca Barca
Competition: More Than A Hack 2027 (FC Barcelona vs Athletic Club Ball Possession)

Design Principles:
- Strictly aligned to the tactical video timeline: 199,319 frames (0 to 199318).
- Coordinates converted from normalised [0, 1] to metres using pitch size (105.3m x 68.0m).
- 46 player track slots partitioned by team membership:
    * 24 Home slots (Barcelona) -> (199319, 24, 3) float32 [x_m, y_m, speed_mps]
    * 22 Away slots (Athletic Club) -> (199319, 22, 3) float32 [x_m, y_m, speed_mps]
    * Ball positions -> (199319, 2) float32 [ball_x_m, ball_y_m]
    * Ball visibility -> (199319,) bool
- All 638 Smart Tagging intervals parsed into structured NumPy array & list of dicts.
- Compressed cache (.npz) enabling sub-second loads across all subsequent phases.
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

# Root path resolution
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from games import GAME, GAMES, require, source_frames, half_video_blocks

# Official constants from games.py
N_VIDEO_FRAMES = source_frames("tactical", GAME)  # 199,319
FPS = GAMES[GAME]["fps"]                         # 25
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"] # 105.3, 68.0
PITCH_DIMS = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)

# Graded blocks: Half 1 (11185..90651), Half 2 (110808..190197)
GRADED_BLOCKS = half_video_blocks(GAME)
TOTAL_GRADED_FRAMES = sum(last - first + 1 for _, first, last in GRADED_BLOCKS) # 158,857

# Default cache directory and file
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
DEFAULT_CACHE_PATH = os.path.join(CACHE_DIR, "tactical_data.npz")
PUBLIC_GT_PATH = os.path.join(ROOT_DIR, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

# Structured dtype for Smart Tagging intervals
SMART_TAGGING_DTYPE = np.dtype([
    ("start_frame", np.int32),
    ("end_frame", np.int32),
    ("start_s", np.float32),
    ("end_s", np.float32),
    ("code", "U40"),
    ("team", "U4"),      # 'H', 'A', or 'N/A'
    ("type", "U30"),     # e.g., 'Corner Kick', 'Throw-in', etc.
    ("side", "U10"),     # 'Left', 'Right', 'Central'
])

# Team name mapping
TEAM_NORM_MAP = {
    "barcelona": "H",
    "atletic club": "A",
    "athletic club": "A",
    "n/a": "N/A",
    "": "N/A",
    "none": "N/A",
}


def get_team_slot_mapping(meta_path: str):
    """Extract team names, team IDs, and player slot partition from ATD metadata XML.

    Returns:
        home_name (str): e.g. 'Barcelona'
        away_name (str): e.g. 'Atletic Club'
        home_slots (list[int]): Slot indices belonging to Home team (0..23)
        away_slots (list[int]): Slot indices belonging to Away team (24..45)
    """
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"Expected 2 teams in {meta_path}, found {len(teams)}")

    home_id = teams[0].get("id")
    away_id = teams[1].get("id")
    home_name = teams[0].findtext("Name")
    away_name = teams[1].findtext("Name")

    home_slots = []
    away_slots = []
    for idx, player in enumerate(root.iter("Player")):
        team_id = player.get("teamId")
        if team_id == home_id:
            home_slots.append(idx)
        elif team_id == away_id:
            away_slots.append(idx)
        else:
            raise ValueError(f"Player slot {idx} has unknown teamId: {team_id}")

    return home_name, away_name, home_slots, away_slots


def parse_atd_tracking(raw_path: str, meta_path: str,
                       pitch: tuple = (PITCH_LENGTH, PITCH_WIDTH),
                       n_frames: int = N_VIDEO_FRAMES):
    """Fast streaming parser for raw ATD tracking text.

    Args:
        raw_path: Path to BAR-ATH_tactical_atd.txt
        meta_path: Path to BAR-ATH_tactical_atd.xml
        pitch: (length, width) in metres
        n_frames: Total video frames (199,319)

    Returns:
        home_players: np.ndarray (n_frames, n_home_slots, 3) float32 [x, y, speed] in metres & m/s
        away_players: np.ndarray (n_frames, n_away_slots, 3) float32 [x, y, speed] in metres & m/s
        ball_pos: np.ndarray (n_frames, 2) float32 [ball_x, ball_y] in metres
        ball_vis: np.ndarray (n_frames,) bool mask (~np.isnan(ball_pos[:, 0]))
        home_name: str
        away_name: str
    """
    home_name, away_name, home_slots, away_slots = get_team_slot_mapping(meta_path)
    n_home = len(home_slots)
    n_away = len(away_slots)
    pitch_len, pitch_wid = pitch

    # Pre-allocate arrays with NaN
    home_players = np.full((n_frames, n_home, 3), np.nan, dtype=np.float32)
    away_players = np.full((n_frames, n_away, 3), np.nan, dtype=np.float32)
    ball_pos = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            video_frame = frame_id - 1
            if video_frame < 0 or video_frame >= n_frames:
                continue

            # Parse ball
            ball_str = parts[2]
            if ball_str[0] != "N":  # Not 'NaN'
                c1 = ball_str.find(",")
                c2 = ball_str.find(",", c1 + 1)
                ball_pos[video_frame, 0] = float(ball_str[:c1]) * pitch_len
                ball_pos[video_frame, 1] = float(ball_str[c1 + 1:c2]) * pitch_wid

            # Parse players
            player_slots = parts[1].split(";")

            # Home players
            for out_idx, slot_idx in enumerate(home_slots):
                group = player_slots[slot_idx]
                if group[0] != "N":
                    c1 = group.find(",")
                    c2 = group.find(",", c1 + 1)
                    home_players[video_frame, out_idx, 0] = float(group[:c1]) * pitch_len
                    home_players[video_frame, out_idx, 1] = float(group[c1 + 1:c2]) * pitch_wid
                    home_players[video_frame, out_idx, 2] = float(group[c2 + 1:])

            # Away players
            for out_idx, slot_idx in enumerate(away_slots):
                group = player_slots[slot_idx]
                if group[0] != "N":
                    c1 = group.find(",")
                    c2 = group.find(",", c1 + 1)
                    away_players[video_frame, out_idx, 0] = float(group[:c1]) * pitch_len
                    away_players[video_frame, out_idx, 1] = float(group[c1 + 1:c2]) * pitch_wid
                    away_players[video_frame, out_idx, 2] = float(group[c2 + 1:])

    ball_vis = ~np.isnan(ball_pos[:, 0])
    return home_players, away_players, ball_pos, ball_vis, home_name, away_name


def parse_smart_tagging(xml_path: str, fps: int = FPS):
    """Parse BAR-ATH_tactical_smart_tagging.xml into structured records and array.

    Args:
        xml_path: Path to Smart Tagging XML
        fps: Frames per second (25)

    Returns:
        structured_array: np.ndarray of SMART_TAGGING_DTYPE (638 intervals)
        records: list[dict] with keys:
            start_frame, end_frame, start_s, end_s, code, team, type, side, tags
    """
    records = []
    instances = list(ET.parse(xml_path).iter("instance"))
    structured_arr = np.zeros(len(instances), dtype=SMART_TAGGING_DTYPE)

    for i, instance in enumerate(instances):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        code = instance.findtext("code") or ""
        tags = {lbl.findtext("group"): lbl.findtext("text") for lbl in instance.findall("label")}

        raw_team = tags.get("Team", "") or ""
        norm_team = TEAM_NORM_MAP.get(raw_team.strip().lower(), "N/A")
        tag_type = tags.get("Type") or ""
        tag_side = tags.get("Side") or ""

        start_frame = int(round(start_s * fps))
        end_frame = int(round(end_s * fps))

        record = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_s": start_s,
            "end_s": end_s,
            "code": code,
            "team": norm_team,
            "type": tag_type,
            "side": tag_side,
            "tags": tags,
        }
        records.append(record)

        structured_arr[i] = (
            start_frame,
            end_frame,
            np.float32(start_s),
            np.float32(end_s),
            code,
            norm_team,
            tag_type,
            tag_side,
        )

    return structured_arr, records


def load_public_ground_truth(gt_path: str = PUBLIC_GT_PATH):
    """Load the public 5-minute ground truth for diagnostic comparisons.

    Args:
        gt_path: Path to public ground truth CSV

    Returns:
        gt_labels: np.ndarray of shape (199319,) dtype '<U1' ('H', 'A', 'D', 'X')
        graded_window: tuple (first_frame, last_frame) -> (11185, 18684)
    """
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Public ground truth not found at {gt_path}")

    # Standard numpy load without heavy pandas
    raw = np.genfromtxt(gt_path, delimiter=",", skip_header=1, dtype=None, encoding="utf-8")
    gt_labels = np.array([row[1] for row in raw], dtype="<U1")
    return gt_labels, (11185, 18684)


def get_graded_frames_mask(n_frames: int = N_VIDEO_FRAMES, game: str = GAME) -> np.ndarray:
    """Return a boolean mask of shape (n_frames,) with True on all graded frames."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def save_tactical_cache(cache_path: str = DEFAULT_CACHE_PATH,
                        home_players: np.ndarray = None,
                        away_players: np.ndarray = None,
                        ball_pos: np.ndarray = None,
                        ball_vis: np.ndarray = None,
                        smart_tagging_arr: np.ndarray = None,
                        home_name: str = "Barcelona",
                        away_name: str = "Atletic Club") -> str:
    """Save parsed arrays to compressed .npz cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        home_players=home_players,
        away_players=away_players,
        ball_pos=ball_pos,
        ball_vis=ball_vis,
        smart_tagging=smart_tagging_arr,
        pitch_dims=PITCH_DIMS,
        meta=np.array([home_name, away_name], dtype="U50"),
    )
    return cache_path


def load_cached_tactical_data(cache_path: str = DEFAULT_CACHE_PATH):
    """Load cached tactical data from .npz in < 1 second.

    Returns:
        dict containing:
            'home_players': (199319, 24, 3) float32
            'away_players': (199319, 22, 3) float32
            'ball_pos': (199319, 2) float32
            'ball_vis': (199319,) bool
            'smart_tagging': (638,) structured np.ndarray
            'pitch_dims': (2,) float32 [105.3, 68.0]
            'home_name': str
            'away_name': str
    """
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Cache file not found at {cache_path}")

    with np.load(cache_path) as data:
        home_players = data["home_players"]
        away_players = data["away_players"]
        ball_pos = data["ball_pos"]
        ball_vis = data["ball_vis"]
        smart_tagging = data["smart_tagging"]
        pitch_dims = data["pitch_dims"]
        meta = data["meta"]

    return {
        "home_players": home_players,
        "away_players": away_players,
        "ball_pos": ball_pos,
        "ball_vis": ball_vis,
        "smart_tagging": smart_tagging,
        "pitch_dims": pitch_dims,
        "home_name": str(meta[0]),
        "away_name": str(meta[1]),
    }


def get_or_load_tactical_data(cache_path: str = DEFAULT_CACHE_PATH,
                              force_reload: bool = False):
    """Retrieve tactical dataset from cache if present; otherwise parse, cache, and return."""
    if os.path.isfile(cache_path) and not force_reload:
        return load_cached_tactical_data(cache_path)

    # Assets required
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    tagging_path = require(["smart_tagging"], GAME)[0]

    home_players, away_players, ball_pos, ball_vis, home_name, away_name = parse_atd_tracking(
        raw_path, meta_path, pitch=(PITCH_LENGTH, PITCH_WIDTH), n_frames=N_VIDEO_FRAMES
    )
    smart_tagging_arr, _ = parse_smart_tagging(tagging_path, fps=FPS)

    save_tactical_cache(
        cache_path,
        home_players=home_players,
        away_players=away_players,
        ball_pos=ball_pos,
        ball_vis=ball_vis,
        smart_tagging_arr=smart_tagging_arr,
        home_name=home_name,
        away_name=away_name,
    )

    return {
        "home_players": home_players,
        "away_players": away_players,
        "ball_pos": ball_pos,
        "ball_vis": ball_vis,
        "smart_tagging": smart_tagging_arr,
        "pitch_dims": PITCH_DIMS,
        "home_name": home_name,
        "away_name": away_name,
    }


if __name__ == "__main__":
    import time
    print(f"Building/Loading tactical cache for {GAME}...")
    t0 = time.time()
    data = get_or_load_tactical_data(force_reload=True)
    t1 = time.time()
    print(f"Parsed and cached in {t1 - t0:.2f}s")
    print(f"Home players shape: {data['home_players'].shape}, dtype={data['home_players'].dtype}")
    print(f"Away players shape: {data['away_players'].shape}, dtype={data['away_players'].dtype}")
    print(f"Ball position shape: {data['ball_pos'].shape}, dtype={data['ball_pos'].dtype}")
    print(f"Ball visible frames: {np.sum(data['ball_vis'])} / {len(data['ball_vis'])} ({np.mean(data['ball_vis'])*100:.1f}%)")
    print(f"Smart Tagging instances: {len(data['smart_tagging'])}")
