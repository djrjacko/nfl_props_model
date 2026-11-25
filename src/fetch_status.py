"""Live player availability: who is out, doubtful, questionable or inactive.

    python src/fetch_status.py              # teams playing in the next 36 hours
    python src/fetch_status.py --hours 60   # a longer look-ahead
    python src/fetch_status.py --no-espn    # force the nflverse fallback (for testing)

Sources, in order:
  1. ESPN's public JSON (unofficial, no key):
       team roster        site.api.espn.com/.../teams/{id}/roster     every player's designation  <- the status source
       league injury feed .../nfl/injuries                            recent notes: injury, practice wording
       game summary       .../summary?event={id}                      game state, ESPN team ids, a cross-check
       team depth chart   .../teams/{id}/depthcharts                  who the starting QB is
       game-day roster    sports.core.api.espn.com/.../roster         OBSERVED ONLY, see below
  2. nflverse injuries and roster status, for any team ESPN cannot answer for.

What the live responses showed on 2026-11-24:
  * The league injury feed holds only the 25 newest notes per team, so older
    designations drop out of it (a QB on injured reserve was missing). The
    game summary lists only five players per team. The team roster endpoint
    carries a designation for every player, so status comes from there.
  * Feed athletes have no id field; the ESPN id is read from their links.
  * ESPN has no structured practice data. Practice participation appears only
    in the wording of a note, so it is read with a cautious text match and
    labelled as such. nflverse carries the official practice status, but only
    the latest one per week, and it can lag.
  * The game-day roster endpoint returned 404 for a game 22 hours away. For a
    finished game, didNotPlay == true looked like the inactive list, but what
    it shows before kickoff has not been seen yet, so this script records what
    it finds there and does not act on it. Until that is verified, "inactive"
    comes from an ESPN note that says so on game day.

Every run adds what it saw about practice, league-wide, to
logs/practice_log.csv. That log is what lets the board say "no practice seen
this week" rather than only "did not practice on the last report".

Output: data/status/status.json. It lists each team's starting QB and every
player who is NOT plainly active; a player who is absent from it is active.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

import fetch_nflverse as nfl
from common import ARIZONA, ROOT, TEAM_ABBR, az, norm_name, utc_now

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
STATUS_DIR = ROOT / "data" / "status"
STATUS_FILE = STATUS_DIR / "status.json"
PRACTICE_LOG = ROOT / "logs" / "practice_log.csv"
PRACTICE_FIELDS = ["season", "week", "team", "player_key", "player", "note_date", "level", "source", "logged_utc"]
DEFAULT_HOURS = 36.0
SKILL = ("QB", "RB", "WR", "TE")
SEVERITY = {"active": 0, "questionable": 1, "doubtful": 2, "out": 3, "inactive": 4}

# nflverse (official NFL report) wording -> board statuses and practice levels.
REPORT_STATUS = {"Questionable": "questionable", "Doubtful": "doubtful", "Out": "out"}
REPORT_PRACTICE = {
    "Did Not Participate In Practice": "dnp",
    "Limited Participation in Practice": "limited",
    "Full Participation in Practice": "full",
}
# nflverse roster codes treated as "not available" in fallback mode. RES is a
# reserve list and PUP/SUS are self-explanatory; RSN, RSR, NWT and EXE are my
# reading of reserve/non-football, reserve/retired, not-with-team and exempt.
RESERVE_CODES = {"RES", "PUP", "SUS", "RSN", "RSR", "NWT", "EXE"}

_PRACTICE_WORD = re.compile(r"practic|participa|walkthrough", re.I)
_DNP = re.compile(
    r"(did not|didn't|does not|doesn't|not|won't|unable to)\s+(practice|participate|take part)"
    r"|non-?participant|held out|sat out|sitting out|missed\s+(\S+\s+)?practice|\bDNP\b",
    re.I,
)
_LIMITED = re.compile(r"\blimited\b", re.I)
_FULL = re.compile(r"\bfull(y)?\b|without (any )?limitations|no limitations", re.I)
_INACTIVE = re.compile(r"\binactive\b|healthy scratch", re.I)


class EspnError(RuntimeError):
    pass


# --------------------------------------------------------------- small pieces


def http_json(url: str, tries: int = 3) -> dict:
    """GET a JSON document. LookupError on 404, EspnError after repeated failures."""
    problem = "no response"
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (compatible; nfl-props-board)"})
            if r.status_code == 404:
                raise LookupError(url)
            if r.status_code == 200:
                return r.json()
            problem = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as exc:
            problem = f"{type(exc).__name__}: {exc}"
        if attempt < tries:
            time.sleep(2 * attempt)
    raise EspnError(f"{url}: {problem}")


def espn_athlete_id(athlete: dict) -> str | None:
    """ESPN athlete id. The injury feed has no id field, so fall back to the id inside the links."""
    if athlete.get("id"):
        return str(athlete["id"])
    hrefs = [link.get("href", "") for link in athlete.get("links", [])]
    hrefs.append((athlete.get("headshot") or {}).get("href", ""))
    for href in hrefs:
        found = re.search(r"/(?:id|full)/(\d+)", href)
        if found:
            return found.group(1)
    return None


def status_from_text(text: str | None) -> str | None:
    """ESPN's designation wording -> a board status. None means the wording is not recognised."""
    t = (text or "").strip().lower()
    if t in ("", "active"):
        return "active"
    if "questionable" in t:
        return "questionable"
    if "doubtful" in t:
        return "doubtful"
    if t == "out" or "reserve" in t or "suspen" in t or "unable to perform" in t or "non-football" in t:
        return "out"
    return None


def practice_level(note: str | None) -> str | None:
    """Read 'dnp', 'limited' or 'full' from the wording of an ESPN note, or None if it does not say.

    This is a text match, not data. A note that mentions more than one level
    counts as the fuller one, so only a note that speaks solely of not
    practising can come back as 'dnp'.
    """
    if not note or not _PRACTICE_WORD.search(note):
        return None
    hits = {name for name, pattern in (("dnp", _DNP), ("limited", _LIMITED), ("full", _FULL)) if pattern.search(note)}
    if not hits:
        return None
    if hits == {"dnp"}:
        return "dnp"
    return "full" if "full" in hits else "limited"


def _parse_time(text: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat((text or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat(timespec="minutes") if moment else None


def _day(moment: datetime) -> str:
    return moment.astimezone(ARIZONA).strftime("%Y-%m-%d")


# ------------------------------------------------------------ nflverse context


class Context:
    """The nflverse facts this script leans on: schedule and the id crosswalk."""

    def __init__(self, now: datetime, hours: float):
        self.now = now
        self.season = nfl.run_info()["current_season"]
        games = nfl.load("schedules", ["game_id", "season", "week", "gameday", "gametime", "away_team", "home_team",
                                       "home_score", "espn"])
        games = games[games["season"] == self.season].copy()
        games["kick"] = nfl.kickoff_arizona(games)
        self.games = games.dropna(subset=["kick"]).sort_values("kick")
        self.window = self.games[(self.games["kick"] > now) & (self.games["kick"] <= now + timedelta(hours=hours))]

        players = nfl.load("players", ["gsis_id", "display_name", "position", "latest_team", "status", "last_season",
                                       "espn_id"])
        self.players = players[(players["last_season"] == self.season) & players["gsis_id"].notna()]
        self.by_espn = self.players.dropna(subset=["espn_id"]).drop_duplicates("espn_id").set_index("espn_id")
        self.by_name = {(norm_name(r.display_name), r.latest_team): r.gsis_id for r in self.players.itertuples()}
        self.by_gsis = self.players.drop_duplicates("gsis_id").set_index("gsis_id")

        # each team's upcoming game (the week its practice notes belong to) and its last kickoff
        self.next_game: dict[str, dict] = {}
        self.last_kick: dict[str, datetime] = {}
        for row in self.games.itertuples():
            for team in (row.away_team, row.home_team):
                if row.kick > now:
                    self.next_game.setdefault(team, {"week": int(row.week), "game_id": row.game_id, "kick": row.kick})
                else:
                    self.last_kick[team] = row.kick

    def match(self, espn_id: str | None, name: str, team: str) -> tuple[str | None, str | None]:
        """(gsis_id, nflverse position) for an ESPN athlete, by ESPN id and then by name within the team."""
        gsis = None
        if espn_id and espn_id in self.by_espn.index:
            gsis = self.by_espn.loc[espn_id, "gsis_id"]
        else:
            gsis = self.by_name.get((norm_name(name), team))
        if not gsis:
            return None, None
        position = self.by_gsis.loc[gsis, "position"]
        return gsis, position if isinstance(position, str) else None

    def in_game_week(self, team: str, moment: datetime | None) -> bool:
        """True if a note was written after the team's last game, so it is about the upcoming one."""
        if moment is None or team not in self.next_game:
            return False
        last = self.last_kick.get(team)
        return last is None or moment > last + timedelta(hours=4)

    def record(self, espn_id: str | None, name: str, team: str, espn_position: str | None, status: str,
               designation: str | None, noted: datetime | None, source: str) -> dict:
        gsis, position = self.match(espn_id, name, team)
        return {"player_key": gsis or f"espn:{espn_id}", "gsis_id": gsis, "espn_id": espn_id, "name": name, "team": team,
                "position": position or espn_position, "status": status, "designation": designation, "injury": None,
                "note_utc": _iso(noted), "source": source}


# -------------------------------------------------------------------- sources


def read_espn_notes(ctx: Context) -> tuple[dict[str, dict], list[dict], list[str]]:
    """League injury feed -> (newest note per ESPN athlete id, practice observations, problems).

    The feed covers every team, so practice wording is logged league-wide even
    though statuses are only needed for the teams in the window.
    """
    feed = http_json(f"{SITE}/injuries")
    notes: dict[str, dict] = {}
    practice: list[dict] = []
    problems: list[str] = []
    for block in feed.get("injuries", []):
        team = TEAM_ABBR.get(block.get("displayName", ""))
        if not team:
            problems.append(f"ESPN feed: unknown team name {block.get('displayName')!r}")
            continue
        for item in block.get("injuries", []):
            athlete = item.get("athlete") or {}
            espn_id = espn_athlete_id(athlete)
            noted = _parse_time(item.get("date"))
            if not espn_id or noted is None:
                continue
            text = item.get("shortComment") or ""
            this_week = ctx.in_game_week(team, noted)
            note = {
                "team": team, "name": athlete.get("displayName") or "", "noted": noted, "this_week": this_week,
                "position": (athlete.get("position") or {}).get("abbreviation"),
                "designation": item.get("status"), "status": status_from_text(item.get("status")),
                "injury": (item.get("details") or {}).get("type"), "says_inactive": bool(_INACTIVE.search(text)),
            }
            if espn_id not in notes or noted > notes[espn_id]["noted"]:
                notes[espn_id] = note
            level = practice_level(text) if this_week else None
            if level:
                gsis, _ = ctx.match(espn_id, note["name"], team)
                practice.append({"season": ctx.season, "week": ctx.next_game[team]["week"], "team": team,
                                 "player_key": gsis or f"espn:{espn_id}", "player": note["name"],
                                 "note_date": _day(noted), "level": level, "source": "espn_note"})
    return notes, practice, problems


def read_team_roster(ctx: Context, team: str, espn_team_id: str) -> list[dict]:
    """ESPN team roster -> one record for every player who carries a designation."""
    roster = http_json(f"{SITE}/teams/{espn_team_id}/roster")
    records = []
    for group in roster.get("athletes", []):
        for athlete in group.get("items", []):
            listed = sorted(athlete.get("injuries") or [], key=lambda entry: entry.get("date") or "")
            if not listed:
                continue
            designation = listed[-1].get("status")
            status = status_from_text(designation)
            rec = ctx.record(str(athlete.get("id")), athlete.get("displayName") or "", team,
                             (athlete.get("position") or {}).get("abbreviation"), status or "questionable",
                             designation, _parse_time(listed[-1].get("date")), "espn")
            rec["unrecognised"] = status is None
            records.append(rec)
    return records


def records_from_notes(ctx: Context, team: str, notes: dict[str, dict]) -> list[dict]:
    """Second best, when a team's roster cannot be read: whatever the feed's recent notes say."""
    records = []
    for espn_id, note in notes.items():
        if note["team"] == team and note["status"] != "active":
            rec = ctx.record(espn_id, note["name"], team, note["position"], note["status"] or "questionable",
                             note["designation"], note["noted"], "espn")
            rec["unrecognised"] = note["status"] is None
            records.append(rec)
    return records


def read_official_report(ctx: Context) -> tuple[list[dict], list[dict], str]:
    """nflverse injuries (the official NFL report) for each team's UPCOMING week only.

    Returns (player records, practice observations, a line about freshness).
    """
    inj = nfl.load("injuries", ["season", "week", "team", "gsis_id", "full_name", "position", "report_primary_injury",
                                "report_status", "practice_status"])
    inj = inj[inj["season"] == ctx.season]
    modified = nfl.last_modified("injuries")  # when nflverse last rebuilt the file
    as_of = parsedate_to_datetime(modified) if modified else ctx.now
    records, practice = [], []
    for row in inj.itertuples():
        upcoming = ctx.next_game.get(row.team)
        if not upcoming or int(row.week) != upcoming["week"] or pd.isna(row.gsis_id):
            continue
        status = REPORT_STATUS.get(row.report_status) if isinstance(row.report_status, str) else None
        if status:
            records.append({
                "player_key": row.gsis_id, "gsis_id": row.gsis_id, "espn_id": None, "name": row.full_name, "team": row.team,
                "position": row.position if isinstance(row.position, str) else None, "status": status,
                "designation": row.report_status,
                "injury": row.report_primary_injury if isinstance(row.report_primary_injury, str) else None,
                "note_utc": _iso(as_of), "source": "nflverse",
            })
        level = REPORT_PRACTICE.get(row.practice_status) if isinstance(row.practice_status, str) else None
        if level:
            practice.append({"season": ctx.season, "week": upcoming["week"], "team": row.team, "player_key": row.gsis_id,
                             "player": row.full_name, "note_date": _day(as_of), "level": level, "source": "nfl_report"})
    newest = int(inj["week"].max()) if len(inj) else 0
    weeks_wanted = sorted({g["week"] for g in ctx.next_game.values()})
    covered = sorted({int(week) for week in inj["week"].unique()} & set(weeks_wanted))
    line = (f"nflverse official report: newest week on file is {newest}; "
            + (f"covers upcoming week(s) {covered}" if covered else f"nothing yet for upcoming week(s) {weeks_wanted}"))
    return records, practice, line


def read_reserve_lists(ctx: Context, team: str) -> list[dict]:
    """Fallback only: players nflverse shows on a reserve list for this team."""
    rows = ctx.players[(ctx.players["latest_team"] == team) & ctx.players["status"].isin(RESERVE_CODES)]
    return [{
        "player_key": r.gsis_id, "gsis_id": r.gsis_id, "espn_id": r.espn_id if isinstance(r.espn_id, str) else None,
        "name": r.display_name, "team": team, "position": r.position if isinstance(r.position, str) else None,
        "status": "out", "designation": f"Reserve list ({r.status})", "injury": None, "note_utc": None, "source": "nflverse",
    } for r in rows.itertuples()]


def update_practice_log(observations: list[dict], now: datetime) -> tuple[pd.DataFrame, int]:
    """Add new practice observations to logs/practice_log.csv. Returns (whole log, rows added)."""
    old = pd.read_csv(PRACTICE_LOG, dtype=str) if PRACTICE_LOG.exists() else pd.DataFrame(columns=PRACTICE_FIELDS, dtype=str)
    both = old
    if observations:
        fresh = pd.DataFrame(observations, columns=PRACTICE_FIELDS[:-1]).astype(str)
        fresh["logged_utc"] = now.isoformat(timespec="seconds")
        # one observation per player, day and source; re-reading the same day later does not add a row
        both = pd.concat([old, fresh], ignore_index=True) if len(old) else fresh
        both = both.drop_duplicates(["season", "week", "player_key", "note_date", "source"])
    both = both.sort_values(["season", "week", "team", "player", "note_date", "source"]).reset_index(drop=True)
    PRACTICE_LOG.parent.mkdir(parents=True, exist_ok=True)
    both.to_csv(PRACTICE_LOG, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    return both, len(both) - len(old)


def starting_qb(ctx: Context, team: str, espn_team_id: str | None) -> dict:
    """The depth-chart QB1, plus whether he is also the QB who threw the most last game."""
    qb: dict = {"name": None, "gsis_id": None, "espn_id": None, "source": None}
    if espn_team_id:
        try:
            chart = http_json(f"{SITE}/teams/{espn_team_id}/depthcharts")
            for group in chart.get("depthchart", []):
                athletes = (group.get("positions", {}).get("qb") or {}).get("athletes") or []
                if athletes:
                    espn_id = str(athletes[0].get("id"))
                    gsis, _ = ctx.match(espn_id, athletes[0].get("displayName", ""), team)
                    qb = {"name": athletes[0].get("displayName"), "gsis_id": gsis, "espn_id": espn_id,
                          "source": "espn depth chart"}
                    break
        except (EspnError, LookupError):
            pass
    if qb["name"] is None:  # nflverse keeps a daily copy of the same ESPN chart
        chart = nfl.load("depth_charts", ["dt", "team", "player_name", "gsis_id", "espn_id", "pos_abb", "pos_rank"])
        top = chart[(chart["dt"] == chart["dt"].max()) & (chart["team"] == team) & (chart["pos_abb"] == "QB")
                    & (chart["pos_rank"] == 1)]
        if len(top):
            row = top.iloc[0]
            qb = {"name": str(row["player_name"]), "gsis_id": row["gsis_id"] if isinstance(row["gsis_id"], str) else None,
                  "espn_id": row["espn_id"] if isinstance(row["espn_id"], str) else None,
                  "source": "nflverse depth chart snapshot"}
    stats = nfl.load("player_week", ["player_id", "player_display_name", "season", "week", "team", "attempts"])
    stats = stats[(stats["season"] == ctx.season) & (stats["team"] == team)]
    if len(stats):
        last = stats[stats["week"] == stats["week"].max()].nlargest(1, "attempts").iloc[0]
        qb["last_game_passer"] = str(last["player_display_name"])
        qb["same_as_last_game"] = bool(qb["gsis_id"] and last["player_id"] == qb["gsis_id"])
    return qb


def read_game_summary(ctx: Context, game, info: dict, problems: list[str]) -> list[tuple[str, str, str]]:
    """Fill in the game's state and ESPN team ids. Returns the summary's own injury list for a cross-check."""
    try:
        summary = http_json(f"{SITE}/summary?event={game.espn}")
    except (EspnError, LookupError) as exc:
        problems.append(f"{info['away']}@{info['home']}: no ESPN game summary ({type(exc).__name__})")
        return []
    competition = summary["header"]["competitions"][0]
    info["state"] = competition["status"]["type"].get("state")
    for side in competition.get("competitors", []):
        abbr = TEAM_ABBR.get(side["team"].get("displayName", ""))
        if abbr:
            info["espn_team_ids"][abbr] = str(side["team"]["id"])
    listed = []
    for block in summary.get("injuries", []):
        for item in block.get("injuries", []):
            athlete = item.get("athlete") or {}
            status = status_from_text(item.get("status"))
            if espn_athlete_id(athlete) and status and status != "active":
                listed.append((espn_athlete_id(athlete), athlete.get("displayName") or "", status))
    return listed


def observe_gameday_roster(ctx: Context, game, info: dict) -> None:
    """Observed only: within 3 hours of kickoff, is the game-day roster published, and what do its flags say?"""
    if game.kick - ctx.now > timedelta(hours=3):
        return
    for abbr, team_id in info["espn_team_ids"].items():
        url = f"{CORE}/events/{game.espn}/competitions/{game.espn}/competitors/{team_id}/roster?limit=200"
        try:
            entries = http_json(url, tries=1).get("entries", [])
            info["gameday_roster"][abbr] = {
                "published": True, "entries": len(entries),
                "active_true": sum(1 for e in entries if e.get("active") is True),
                "valid_true": sum(1 for e in entries if e.get("valid") is True),
                "did_not_play_true": sum(1 for e in entries if e.get("didNotPlay") is True),
                "did_not_play_names": [e.get("displayName") for e in entries if e.get("didNotPlay") is True],
            }
        except (EspnError, LookupError):
            info["gameday_roster"][abbr] = {"published": False}


# ----------------------------------------------------------------- assembling


def build(hours: float, use_espn: bool) -> dict:
    now = utc_now()
    ctx = Context(now, hours)
    problems: list[str] = []
    notes: dict[str, dict] = {}
    practice: list[dict] = []
    feed_ok = False
    if use_espn:
        try:
            notes, practice, feed_problems = read_espn_notes(ctx)
            problems += feed_problems
            feed_ok = True
        except (EspnError, LookupError, KeyError, TypeError) as exc:
            problems.append(f"ESPN injury feed failed ({type(exc).__name__}: {str(exc)[:100]}): no notes or practice wording this run")
    official, official_practice, official_line = read_official_report(ctx)
    practice += official_practice
    log, added = update_practice_log(practice, now)

    games, records, team_source = [], [], {}
    summary_lists: list[tuple[str, list]] = []
    for game in ctx.window.itertuples():
        info = {"game_id": game.game_id, "espn_id": str(game.espn), "away": game.away_team, "home": game.home_team,
                "kickoff_utc": _iso(game.kick.tz_convert("UTC")), "state": None, "espn_team_ids": {}, "gameday_roster": {}}
        if use_espn:
            summary_lists.append((f"{info['away']}@{info['home']}", read_game_summary(ctx, game, info, problems)))
            observe_gameday_roster(ctx, game, info)
        for team in (info["away"], info["home"]):
            team_records = None
            if use_espn and team in info["espn_team_ids"]:
                try:
                    team_records = read_team_roster(ctx, team, info["espn_team_ids"][team])
                    team_source[team] = "espn roster"
                except (EspnError, LookupError, KeyError, TypeError) as exc:
                    problems.append(f"{team}: ESPN roster failed ({type(exc).__name__})")
            if team_records is None and feed_ok:
                team_records = records_from_notes(ctx, team, notes)
                team_source[team] = "espn recent notes only"
                problems.append(f"{team}: status comes from ESPN's recent notes only, so older designations may be missing")
            if team_records is None:
                team_records = [r for r in official if r["team"] == team] + read_reserve_lists(ctx, team)
                team_source[team] = "nflverse"
                if not any(r["team"] == team for r in official):
                    problems.append(f"{team}: nflverse fallback has no official report for the upcoming week yet, "
                                    "so only reserve lists are known")
            records += team_records
        games.append(info)

    # one record per player; ESPN's notes add the injury and can sharpen the status
    by_key: dict[str, dict] = {}
    for rec in records:
        by_key.setdefault(rec["player_key"], rec)
    teams = set(team_source)
    for espn_id, note in notes.items():
        if note["team"] not in teams:
            continue
        gsis, _ = ctx.match(espn_id, note["name"], note["team"])
        key = gsis or f"espn:{espn_id}"
        rec = by_key.get(key)
        if rec is None:
            if note["status"] in (None, "active"):
                continue
            # the feed lists someone the roster does not: keep him, he is at least worth a look
            rec = ctx.record(espn_id, note["name"], note["team"], note["position"], note["status"] or "questionable",
                             note["designation"], note["noted"], "espn")
            by_key[key] = rec
        rec["injury"] = rec["injury"] or note["injury"]
        newer = rec["note_utc"] is None or _iso(note["noted"]) > rec["note_utc"]
        if newer:
            rec["note_utc"] = _iso(note["noted"])
            if note["status"] and note["status"] != rec["status"] and note["status"] != "active":
                problems.append(f"{rec['name']} ({rec['team']}): roster says {rec['status']}, a newer note says {note['status']}; using the note")
                rec["status"], rec["designation"] = note["status"], note["designation"]
        kick = ctx.next_game.get(note["team"], {}).get("kick")
        if rec["status"] == "out" and note["says_inactive"] and note["this_week"] and kick is not None \
                and note["noted"] > kick - timedelta(hours=8):
            rec["status"] = "inactive"
    # the official report can list someone ESPN does not
    for rec in official:
        if rec["team"] in teams and rec["player_key"] not in by_key:
            by_key[rec["player_key"]] = rec
            problems.append(f"{rec['name']} ({rec['team']}): {rec['status']} on the official report but not listed by ESPN")

    # practice seen this week, per player
    seen: dict[str, list[dict]] = {}
    for row in log.itertuples():
        upcoming = ctx.next_game.get(row.team)
        if row.team in teams and upcoming and str(upcoming["week"]) == str(row.week) and str(ctx.season) == str(row.season):
            seen.setdefault(row.player_key, []).append({"date": row.note_date, "level": row.level, "source": row.source})

    players = []
    for rec in by_key.values():
        if rec["status"] == "active":
            continue
        if rec.pop("unrecognised", False):
            problems.append(f"{rec['name']} ({rec['team']}): unrecognised ESPN designation {rec['designation']!r}, treated as questionable")
        days = sorted(seen.get(rec["player_key"], []), key=lambda d: (d["date"], d["source"]))
        rec["practice"] = {
            "seen": days,
            "latest": days[-1]["level"] if days else None,
            # True only when there is evidence and none of it shows any participation
            "none_this_week": ({d["level"] for d in days} == {"dnp"}) if days else None,
        }
        players.append(rec)
    players.sort(key=lambda r: (r["team"], r["position"] not in SKILL, -SEVERITY[r["status"]], r["position"] or "", r["name"]))
    no_id = [f"{p['name']} ({p['team']} {p['position']})" for p in players if not p["gsis_id"]]
    if no_id:
        skill_no_id = [f"{p['name']} ({p['team']} {p['position']})" for p in players if not p["gsis_id"] and p["position"] in SKILL]
        problems.append(f"{len(no_id)} listed players have no nflverse id, so the model can only match them by name"
                        + (f"; skill positions: {', '.join(skill_no_id)}" if skill_no_id else "; none at a skill position"))

    for label, listed in summary_lists:
        for espn_id, name, status in listed:
            ours = next((p["status"] for p in players if p["espn_id"] == espn_id), "active")
            if ours != status and not (ours == "inactive" and status == "out"):
                problems.append(f"{label}: ESPN's game summary has {name} as {status}, the roster has him as {ours}")

    team_info = {}
    for info in games:
        for team in (info["away"], info["home"]):
            qb = starting_qb(ctx, team, info["espn_team_ids"].get(team))
            listed = by_key.get(qb.get("gsis_id") or f"espn:{qb.get('espn_id')}")
            qb["status"] = listed["status"] if listed else "active"
            qb["practice_none_this_week"] = listed["practice"]["none_this_week"] if listed and "practice" in listed else None
            if qb.get("same_as_last_game") is False:
                problems.append(f"{team}: depth-chart QB1 is {qb['name']} but {qb.get('last_game_passer')} threw the most passes last game")
            team_info[team] = {"starting_qb": qb, "status_source": team_source[team]}

    sources = set(team_source.values())
    return {
        "generated_utc": now.isoformat(timespec="seconds"),
        "source": "espn" if sources <= {"espn roster"} else ("nflverse" if sources == {"nflverse"} else "mixed"),
        "window_hours": hours, "official_report": official_line, "practice_log_rows": len(log),
        "practice_log_added": added, "games": games, "teams": team_info, "players": players, "problems": problems,
    }


def load() -> dict:
    """The last status file. Players not listed in it are active."""
    if not STATUS_FILE.exists():
        raise FileNotFoundError("data/status/status.json is missing. Run: python src/fetch_status.py")
    return json.loads(STATUS_FILE.read_text(encoding="utf-8"))


def _practice_text(practice: dict) -> str:
    if not practice["seen"]:
        return "nothing seen"
    days = [f"{d['level']} {datetime.strptime(d['date'], '%Y-%m-%d').strftime('%a')}"
            + ("" if d["source"] == "nfl_report" else "*") for d in practice["seen"]]
    return ", ".join(days) + ("  -> NO PRACTICE SEEN" if practice["none_this_week"] else "")


def report(status: dict) -> None:
    label = {"espn": "ESPN", "nflverse": "nflverse FALLBACK (ESPN not used)", "mixed": "MIXED (see below)"}[status["source"]]
    print(f"\nPLAYER STATUS | {az(utc_now())} | source: {label} | window: next {status['window_hours']:g} hours")
    print(status["official_report"])
    print(f"practice log: {status['practice_log_added']} new observations, {status['practice_log_rows']} in total "
          f"({PRACTICE_LOG.relative_to(ROOT)})")
    if not status["games"]:
        print("\nNo games kick off in this window.")
    for game in status["games"]:
        kick = datetime.fromisoformat(game["kickoff_utc"])
        print(f"\n{game['away']} @ {game['home']} | {az(kick)} | ESPN game {game['espn_id']}, state: {game['state'] or 'unknown'}")
        for team in (game["away"], game["home"]):
            qb = status["teams"][team]["starting_qb"]
            same = {True: "also threw the most passes last game", False: f"last game's passer was {qb.get('last_game_passer')}",
                    None: "no last-game check"}[qb.get("same_as_last_game")]
            print(f"  {team} starting QB: {qb['name']} - {qb['status'].upper()} ({qb['source']}; {same})")
        listed = [p for p in status["players"] if p["team"] in (game["away"], game["home"])]
        reserve = [p for p in listed if "reserve" in (p["designation"] or "").lower()]
        shown = [p for p in listed if p["position"] in SKILL and p not in reserve]
        print(f"  skill-position players with a designation ({len(shown)}):")
        if shown:
            print(f"    {'team':<5}{'player':<24}{'pos':<4}{'status':<14}{'injury':<16}{'noted':<25}practice seen this week")
        for p in shown:
            noted = az(datetime.fromisoformat(p["note_utc"])) if p["note_utc"] else "-"
            print(f"    {p['team']:<5}{p['name'][:23]:<24}{p['position']:<4}{p['status']:<14}{str(p['injury'] or '-')[:15]:<16}"
                  f"{noted:<25}{_practice_text(p['practice'])}")
        for team in (game["away"], game["home"]):
            ir = [f"{p['name']} ({p['position']})" for p in reserve if p["team"] == team and p["position"] in SKILL]
            rest = [p for p in listed if p["team"] == team and p["position"] not in SKILL]
            counts = pd.Series([p["status"] for p in rest]).value_counts().to_dict() if rest else {}
            print(f"  {team} [{status['teams'][team]['status_source']}] on a reserve list (skill positions): {', '.join(ir) or 'none'}"
                  f" | other positions with a designation: {len(rest)} {counts or ''}")
        for team, seen in game["gameday_roster"].items():
            if seen.get("published"):
                print(f"  {team} game-day roster is published (observed only): {seen['entries']} entries, active {seen['active_true']}, "
                      f"valid {seen['valid_true']}, didNotPlay {seen['did_not_play_true']} {seen['did_not_play_names']}")
            else:
                print(f"  {team} game-day roster: not published yet")
    print("\n  * = read from the wording of an ESPN note, not from structured data")
    if status["problems"]:
        print("\nthings to know:")
        for problem in status["problems"]:
            print(f"  - {problem}")
    print(f"\nsaved to {STATUS_FILE.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Player availability from ESPN, with nflverse as the fallback.")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="look-ahead window in hours (default 36)")
    parser.add_argument("--no-espn", action="store_true", help="skip ESPN and use the nflverse fallback")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    status = build(args.hours, use_espn=not args.no_espn)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=1), encoding="utf-8", newline="
")
    report(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
