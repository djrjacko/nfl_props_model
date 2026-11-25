"""Start a board run shortly before any kickoff that the fixed schedule does not reach in time.

Inactives are announced 90 minutes before kickoff. The fixed schedule in run.yml has a run soon after that
for the usual Thursday-night, Sunday and Monday-night kickoffs. Other games have none: a Wednesday game,
Thanksgiving afternoon, a Saturday late in the season, an early kickoff abroad. A small workflow
(pregame.yml) runs this script every 15 minutes, and when such a game is 85 to 25 minutes from kickoff
with no board run since inactives were due, it starts "Run the prop board" itself.

    python src/pregame.py            # show what it would do; starts nothing
    python src/pregame.py --trigger  # start the board run if one is due (what pregame.yml does)

A game counts as covered when a fixed run in run.yml falls 85 to 40 minutes before its kickoff. The games
come from The Odds API's events feed, which costs no credits. Only the standard library is used, so the
15-minute check installs nothing.
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

from common import BOARD_WORKFLOW, PREGAME_WORKFLOW, ROOT, TEAM_ABBR, az, odds_api_key, scheduled_runs, utc_now

EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
LEAD = timedelta(minutes=85)  # earliest a pre-kickoff run is useful: five minutes after inactives are due
LATEST = timedelta(minutes=25)  # closer to kickoff than this, a run would land too late to act on
COVERED_UNTIL = timedelta(minutes=40)  # a fixed run between LEAD and this before kickoff covers the game
NO_SHOW = timedelta(minutes=20)  # how overdue a fixed run can be before this script stops waiting for it
LOOK_AHEAD = timedelta(hours=40)  # how far ahead to list games when showing the plan
BOARD_HOURS = "36"


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


def plan(now: datetime, events: list[dict], recent_runs: list[datetime] | None = None) -> list[dict]:
    """What to do about each upcoming game. `recent_runs` are the start times of recent board runs."""
    out = []
    for event in sorted(events, key=kickoff):
        kick = kickoff(event)
        opens, closes = kick - LEAD, kick - LATEST
        entry = {"game": label(event), "kickoff": kick, "state": None, "at": None, "why": ""}
        fixed = [run for run in scheduled_runs(opens - timedelta(minutes=1), kick - COVERED_UNTIL) if run >= opens]
        done = [run for run in (recent_runs or []) if opens <= run <= kick]
        # GitHub occasionally drops a scheduled run. If the fixed one is 20 minutes overdue with nothing
        # started, stop waiting for it.
        no_show = bool(fixed) and not done and fixed[0] + NO_SHOW < now <= closes
        if done:
            entry.update(state="done", at=done[0], why=f"a board run already started at {az(done[0])}")
        elif no_show:
            entry.update(state="due", at=now, why=f"the fixed run at {az(fixed[0])} never started")
        elif fixed:
            entry.update(state="covered", at=fixed[0], why=f"the fixed schedule has a run at {az(fixed[0])}")
        elif now > closes:
            entry.update(state="late", why="less than 25 minutes to kickoff")
        elif now >= opens:
            entry.update(state="due", at=now, why="no run since inactives were due")
        else:
            ticks = scheduled_runs(opens - timedelta(minutes=1), closes, PREGAME_WORKFLOW)
            ticks = [tick for tick in ticks if tick >= opens]
            entry.update(state="waiting", at=ticks[0] if ticks else None,
                         why="its own run comes at the first check after inactives are due")
        out.append(entry)
    return out


def describe(entries: list[dict]) -> list[str]:
    lines = []
    for entry in entries:
        what = {"covered": f"covered: {entry['why']}", "done": f"done: {entry['why']}", "late": f"skipped: {entry['why']}",
                "due": "DUE NOW: starting a board run",
                "waiting": f"own run about {az(entry['at'])}" if entry["at"] else "own run, but no check is scheduled in time"}[entry["state"]]
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
    status, data = github("POST", f"/repos/{repo}/actions/workflows/{BOARD_WORKFLOW.name}/dispatches", token,
                          {"ref": "main", "inputs": {"hours": BOARD_HOURS}})
    if status not in (200, 204):
        raise RuntimeError(f"could not start the board run (HTTP {status}): {data.get('message', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start a board run before kickoffs the fixed schedule misses.")
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
