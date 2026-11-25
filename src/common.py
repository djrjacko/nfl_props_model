"""Small things every script shares: the project folder, Arizona time, team names."""
from __future__ import annotations

import copy
import json
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Arizona is UTC-7 all year with no daylight saving, so a fixed offset is exact and needs no time-zone
# database (plain Python on Windows has none, and the 15-minute pregame check installs nothing).
ARIZONA = timezone(timedelta(hours=-7), "MST")

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
    # Every boost and same-game ticket must start from an anchor leg: an RB rush attempts or rush yards over,
    # an anytime TD, or a team moneyline. false goes back to "any legs that share a game script".
    "require_anchor": True,
    # What the outside scheduler is set to do. Nothing here starts anything: the jobs live at the scheduler.
    # These lines describe them, so the board can list the next runs and the pregame check can say when it
    # will act. Change them when the jobs change.
    #   board_runs     times of day, Arizona time, at which it starts run.yml ("HH:MM", 24-hour clock)
    #   board_hours    the hours value it sends; a run started by the pregame check uses the same value
    #   check_minutes  minutes of every hour at which it starts pregame.yml
    "outside_schedule": {"board_runs": ["06:37", "15:37", "21:07"], "board_hours": 60, "check_minutes": [0, 15, 30, 45]},
}


def load_config() -> dict:
    """config.json laid over the defaults above."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        mine = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        # a config.json that sets only part of the outside schedule keeps the defaults for the rest
        mine["outside_schedule"] = {**config["outside_schedule"], **mine.get("outside_schedule", {})}
        config.update(mine)
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


def odds_api_key() -> tuple[str, str]:
    """(The Odds API key, where it came from). Raises LookupError if there is none. The key is never shown."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key, "the ODDS_API_KEY environment variable"
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.strip().partition("=")
            if name.strip() == "ODDS_API_KEY" and value.strip():
                return value.strip().strip("\"'"), ".env"
    raise LookupError("no API key found. Put it in .env as ODDS_API_KEY=... or set the ODDS_API_KEY environment variable.")


# --------------------------------------------------------- the outside schedule

WORKFLOWS = ROOT / ".github" / "workflows"
BOARD_WORKFLOW = WORKFLOWS / "run.yml"  # builds and publishes the board
PREGAME_WORKFLOW = WORKFLOWS / "pregame.yml"  # the check that starts a board run once inactives are due

# Neither workflow has a schedule of its own: GitHub's cron never started a run in this repository, so an
# outside scheduler starts both with GitHub's "workflow dispatch" call (see the README's Schedule section).


def outside_schedule() -> dict:
    """The outside scheduler's jobs, from config.json:
    {"board_runs": [(hour, minute), ...] in Arizona time, "board_hours": "60", "check_minutes": [0, 15, 30, 45]}.

    The pregame check runs through here, so a value that cannot be read falls back to its default instead of
    stopping the check: a typo in a description must not cost a run before a kickoff.
    """
    usual = DEFAULT_CONFIG["outside_schedule"]
    raw = {**usual, **(load_config().get("outside_schedule") or {})}

    def clock(text) -> tuple[int, int]:
        hour, minute = (int(part) for part in str(text).split(":"))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError(text)
        return hour, minute

    def read(key: str, convert):
        try:
            return convert(raw[key])
        except (TypeError, ValueError):
            return convert(usual[key])

    def hours(value) -> str:
        if not float(value) > 0:
            raise ValueError(value)
        return f"{float(value):g}"

    def minutes(values) -> list[int]:
        found = sorted({int(value) for value in values})
        if any(not 0 <= value < 60 for value in found):
            raise ValueError(values)
        return found

    return {"board_runs": read("board_runs", lambda values: sorted(clock(value) for value in values)),
            "board_hours": read("board_hours", hours), "check_minutes": read("check_minutes", minutes)}


def daily_board_runs(start: datetime, end: datetime) -> list[datetime]:
    """The outside scheduler's board runs after `start` and up to `end`."""
    found = []
    day = start.astimezone(ARIZONA).replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        for hour, minute in outside_schedule()["board_runs"]:
            moment = day.replace(hour=hour, minute=minute)
            if start < moment <= end:
                found.append(moment.astimezone(timezone.utc))
        day += timedelta(days=1)
    return sorted(found)


def first_check(moment: datetime) -> datetime | None:
    """The outside scheduler's first pregame check at or after `moment`, or None if it makes no checks.
    Arizona is a whole number of hours from UTC, so the minute of the hour is the same in both."""
    minutes = outside_schedule()["check_minutes"]
    if not minutes:
        return None
    tick = moment.replace(second=0, microsecond=0)
    if tick < moment:
        tick += timedelta(minutes=1)
    while tick.minute not in minutes:
        tick += timedelta(minutes=1)
    return tick
