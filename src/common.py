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


# ------------------------------------------------- reading the workflow schedules

WORKFLOWS = ROOT / ".github" / "workflows"
BOARD_WORKFLOW = WORKFLOWS / "run.yml"  # the fixed schedule of board runs
PREGAME_WORKFLOW = WORKFLOWS / "pregame.yml"  # the 15-minute check that adds runs before off-schedule kickoffs


def _cron_field(text: str) -> set[int] | None:
    """One cron field as a set of numbers, or None for '*'. Understands lists and ranges: '7,22' and '11-23'."""
    if text == "*":
        return None
    values: set[int] = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        values.update(range(int(low), int(high or low) + 1))
    return values


def scheduled_runs(start: datetime, end: datetime, workflow: Path = BOARD_WORKFLOW) -> list[datetime]:
    """Every moment after `start` and up to `end` at which a cron line in a workflow file fires (cron is UTC).

    The schedule is read from the workflow file itself, so it is written down in one place only.
    """
    if not workflow.exists():
        return []
    found: set[datetime] = set()
    for expression in re.findall(r"cron:\s*[\"']([^\"']+)[\"']", workflow.read_text(encoding="utf-8")):
        minutes, hours, days, months, weekdays = (_cron_field(field) for field in expression.split())
        day = start.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= end:
            cron_weekday = (day.weekday() + 1) % 7  # cron counts from Sunday = 0 (7 also means Sunday)
            on_date = days is None or day.day in days
            on_weekday = weekdays is None or cron_weekday in weekdays or (cron_weekday == 0 and 7 in weekdays)
            # cron's own rule: when both the date and the weekday are given, either one is enough
            day_matches = (on_date or on_weekday) if (days is not None and weekdays is not None) else (on_date and on_weekday)
            if day_matches and (months is None or day.month in months):
                for hour in sorted(hours if hours is not None else range(24)):
                    for minute in sorted(minutes if minutes is not None else range(60)):
                        moment = day.replace(hour=hour, minute=minute)
                        if start < moment <= end:
                            found.add(moment)
            day += timedelta(days=1)
    return sorted(found)
