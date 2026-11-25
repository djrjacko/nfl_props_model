"""Small things every script shares: the project folder, Arizona time, team names."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ARIZONA = ZoneInfo("America/Phoenix")  # UTC-7 all year, no daylight saving

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


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
