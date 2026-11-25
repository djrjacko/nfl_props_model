"""Start a board run once inactives are due for a game.

Inactives are announced 90 minutes before kickoff. An outside scheduler starts pregame.yml every 15 minutes
and pregame.yml runs this script. The first check that falls 85 to 25 minutes before a kickoff, with no
board run started since that game's inactives were due, starts "Run the prop board". Later checks see that
run and start nothing, so each kickoff time gets one run. A board run started some other way inside that
window (one of the scheduler's daily runs, or a tap on the site) counts as that run too.

    python src/pregame.py            # show what it would do; starts nothing
    python src/pregame.py --trigger  # start the board run if one is due (what pregame.yml does)

The games come from The Odds API's events feed, which costs no credits. Only the standard library is used,
so a check installs nothing. config.json "outside_schedule" describes the scheduler's jobs. It supplies the
look-ahead a started run asks for and the times shown for runs still to come; it decides nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from common import BOARD_WORKFLOW, ROOT, TEAM_ABBR, az, daily_board_runs, first_check, odds_api_key, outside_schedule, utc_now

EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
LEAD = timedelta(minutes=85)  # a run is useful from here on: five minutes after inactives are due
LATEST = timedelta(minutes=25)  # closer to kickoff than this, a run would land too late to act on
LOOK_AHEAD = timedelta(hours=40)  # how far ahead to list games when showing the plan


def fetch_events(key: str, now: datetime) -> list[dict]:
    """Upcoming games from The Odds API (free endpoint): id, teams and kickoff."""
    query = urllib.parse.urlencode({
        "apiKey": key, "dateFormat": "iso",
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": (now + LOOK_AHEAD).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    try:
        with urllib.request.urlopen(f"{EVENTS_URL}?{query}", timeout=30) as response:
            return json.load(response)
    except urllib.error.URLError as exc:  # the message can carry the address, so keep the key out of it
        raise RuntimeError(f"The Odds API events feed failed: {str(exc).replace(key, '***')}") from None


def kickoff(event: dict) -> datetime:
    return datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))


def label(event: dict) -> str:
    return f"{TEAM_ABBR.get(event['away_team'], event['away_team'])}@{TEAM_ABBR.get(event['home_team'], event['home_team'])}"


def expected_run(opens: datetime, closes: datetime) -> tuple[datetime | None, str | None]:
    """When the run for a game is expected, and what starts it: ("check" or "daily"). It is the first pregame
    check between `opens` and `closes`, unless one of the scheduler's daily board runs lands in that window
    before the check does, in which case the check will find that run and start nothing."""
    check = first_check(opens)
    daily = daily_board_runs(opens - timedelta(seconds=1), closes)
    if daily and (check is None or daily[0] < check):
        return daily[0], "daily"
    if check is not None and check <= closes:
        return check, "check"
    return None, None


def plan(now: datetime, events: list[dict], recent_runs: list[datetime] | None = None) -> list[dict]:
    """What to do about each upcoming game. `recent_runs` are the start times of recent board runs, however
    they were started.

    The rule: a game is due from 85 minutes before kickoff (five minutes after its inactives are due) until
    25 minutes before, unless a board run has already started since then. One run is enough for every game
    that shares that kickoff time.
    """
    out = []
    for event in sorted(events, key=kickoff):
        kick = kickoff(event)
        opens, closes = kick - LEAD, kick - LATEST
        entry = {"game": label(event), "kickoff": kick, "state": None, "at": None, "by": None, "why": ""}
        done = [run for run in (recent_runs or []) if opens <= run <= kick]
        if done:
            entry.update(state="done", at=done[0], why=f"a board run already started at {az(done[0])}")
        elif now > closes:
            entry.update(state="late", why="less than 25 minutes to kickoff")
        elif now >= opens:
            entry.update(state="due", at=now, why="no board run since inactives were due")
        else:
            at, by = expected_run(opens, closes)
            entry.update(state="waiting", at=at, by=by, why="inactives are not due yet")
        out.append(entry)
    return out


def describe(entries: list[dict]) -> list[str]:
    lines = []
    for entry in entries:
        if entry["state"] == "waiting" and entry["by"] == "check":
            what = f"board run at the {az(entry['at'])} check, the first after inactives are due"
        elif entry["state"] == "waiting" and entry["by"] == "daily":
            what = f"covered by the daily run at {az(entry['at'])}, which comes after inactives are due"
        elif entry["state"] == "waiting":
            what = "no check or daily run falls between inactives and kickoff"
        else:
            what = {"done": f"done: {entry['why']}", "late": f"skipped: {entry['why']}",
                    "due": "DUE NOW: starting a board run"}[entry["state"]]
        lines.append(f"  {entry['game']:<9}{az(entry['kickoff']):<25}{what}")
    return lines


# ------------------------------------------------------------------ GitHub


def repo_name() -> str | None:
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    try:
        url = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "remote.origin.url"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    tail = url.split("github.com")[-1].lstrip(":/") if "github.com" in url else ""
    return (tail[:-4] if tail.endswith(".git") else tail) or None


def github(method: str, path: str, token: str | None, body: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(f"https://api.github.com{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
    request.add_header("Accept", "application/vnd.github+json")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        return exc.code, {"message": exc.read().decode(errors="replace")[:300]}


def recent_board_runs(repo: str, token: str | None, since: datetime) -> list[datetime]:
    """Start times of "Run the prop board" runs created since `since`, whoever or whatever started them."""
    created = urllib.parse.quote(">=" + since.strftime("%Y-%m-%dT%H:%M:%SZ"))
    status, data = github("GET", f"/repos/{repo}/actions/workflows/{BOARD_WORKFLOW.name}/runs?per_page=30&created={created}", token)
    if status != 200:
        raise RuntimeError(f"could not list recent board runs (HTTP {status}): {data.get('message', '')}")
    return [datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) for run in data.get("workflow_runs", [])]


def start_board_run(repo: str, token: str) -> None:
    """Start "Run the prop board" with the same look-ahead the scheduler's own runs use."""
    status, data = github("POST", f"/repos/{repo}/actions/workflows/{BOARD_WORKFLOW.name}/dispatches", token,
                          {"ref": "main", "inputs": {"hours": outside_schedule()["board_hours"]}})
    if status not in (200, 204):
        raise RuntimeError(f"could not start the board run (HTTP {status}): {data.get('message', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start a board run once inactives are due for a game.")
    parser.add_argument("--trigger", action="store_true", help="actually start the board run when a game is due")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    now = utc_now()
    try:
        key, _ = odds_api_key()
        events = fetch_events(key, now)
        repo, token = repo_name(), os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        runs = recent_board_runs(repo, token, now - timedelta(hours=2)) if repo else []
    except (LookupError, RuntimeError) as exc:
        print(f"Pregame check failed: {exc}")
        return 1
    entries = plan(now, events, runs)
    print(f"PREGAME CHECK | {az(now)} | {len(events)} game(s) in the next {LOOK_AHEAD.total_seconds() / 3600:g} hours")
    print("\n".join(describe(entries)) or "  no games coming up")
    due = [entry for entry in entries if entry["state"] == "due"]
    if not due:
        print("Nothing is due.")
        return 0
    games = ", ".join(entry["game"] for entry in due)
    if not args.trigger:
        print(f"A board run is due for {games}. Run with --trigger to start it.")
        return 0
    if not (repo and token):
        print("A board run is due but there is no GitHub token to start it with.")
        return 1
    try:
        start_board_run(repo, token)
    except RuntimeError as exc:
        print(f"Pregame check failed: {exc}")
        return 1
    print(f"Started the board run for {games}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
