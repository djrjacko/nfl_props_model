"""Small things every script shares: the project folder, Arizona time, team names."""
from __future__ import annotations

import copy
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ARIZONA = ZoneInfo("America/Phoenix")  # UTC-7 all year, no daylight saving

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Personal settings live in config.json at the top of the repo so they can be edited without touching code.
CONFIG_FILE = ROOT / "config.json"
DEFAULT_CONFIG = {
    # teams the owner roots for: a leg that wins when one of them does badly is tagged "rooting conflict"
    "rooting_teams": [],
    # the boosts the owner usually has; "safest" picks the likeliest ticket, "best" the highest boosted value
    "boosts": [
        {"name": "50% boost", "boost": 0.5, "max_stake": 50, "legs": 2, "min_price": 150, "max_price": 250, "pick": "safest"},
        {"name": "100% boost", "boost": 1.0, "max_stake": 25, "legs": 3, "min_price": 300, "max_price": 400, "pick": "best"},
    ],
}


def load_config() -> dict:
    """config.json laid over the defaults above."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        config.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    return config


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def norm_name(name: str | None) -> str:
    """A name reduced for matching across sources: 'Kenneth Walker III' -> 'kenneth walker',
    'D.J. Moore' -> 'dj moore', 'Amon-Ra St. Brown' -> 'amon ra st brown'."""
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z ]", "", text.replace("-", " ")).split()
    return " ".join(word for word in words if word not in _NAME_SUFFIXES)


def az(moment: datetime) -> str:
    """A timezone-aware datetime as Arizona time, e.g. 'Tue Nov 24 7:37 PM MST'."""
    return moment.astimezone(ARIZONA).strftime("%a %b %d %I:%M %p").replace(" 0", " ") + " MST"


# Full team names as The Odds API writes them -> the abbreviations nflverse uses.
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
