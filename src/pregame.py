"""Start a board run once inactives are due for a game.

Inactives are announced 90 minutes before kickoff. An outside scheduler starts pregame.yml every 15 minutes
and pregame.yml runs this script. The first check that falls 85 to 25 minutes before a kickoff starts "Run
the prop board". Once a board run that started since that game's inactives were due has finished
successfully, later checks start nothing, so each kickoff time gets one good run. A board run started some
other way inside that window (one of the scheduler's daily runs, or a tap on the site) counts as well.

A run that fails or is cancelled does not count. The next check inside the window starts another one, up
to two retries per kickoff. After that, or when there is no time left, the check stops and writes a notice
to docs/notices.json, which the site shows with the board's notes and the next boards repeat in theirs.

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

from common import ARIZONA, BOARD_WORKFLOW, ROOT, TEAM_ABBR, az, daily_board_runs, first_check, odds_api_key, outside_schedule, utc_now

EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
LEAD = timedelta(minutes=85)  # a run is useful from here on: five minutes after inactives are due
LATEST = timedelta(minutes=25)  # closer to kickoff than this, a run would land too late to act on
LOOK_AHEAD = timedelta(hours=40)  # how far ahead to list games when showing the plan
RETRIES = 2  # after a run fails or is cancelled: this many more tries for that kickoff, then stop
NOTICES_FILE = ROOT / "docs" / "notices.json"  # what the site shows when a run after inactives was given up on
NOTICE_LIFE = timedelta(hours=12)  # how long after kickoff such a notice stays


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


def clock(moment: datetime) -> str:
    """A time of day in Arizona, like '4:45 PM'."""
    return moment.astimezone(ARIZONA).strftime("%I:%M %p").lstrip("0")


def listed(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def plan(now: datetime, events: list[dict], recent_runs: list[dict] | None = None) -> list[dict]:
    """What to do about each upcoming game. `recent_runs` are recent board runs, however they were started:
    {"at": start time, "status": GitHub's status, "conclusion": GitHub's conclusion}.

    A game's window runs from 85 minutes before kickoff (five minutes after its inactives are due) to 25
    minutes before. Looking only at board runs that started since the window opened:
      done     one finished successfully. Nothing more is started for this kickoff.
      running  one is still going. This check waits for it.
      due      none has succeeded and the window is open: start one. It is the first run, or a retry after
               a run that failed or was cancelled.
      failed   three runs failed (the first and RETRIES retries), or one failed and the window has closed.
               Nothing more is started, and a notice is published.
      late     no run was started and the window has closed.
      waiting  the window has not opened yet.
    One run serves every game that shares the kickoff time.
    """
    out = []
    for event in sorted(events, key=kickoff):
        kick = kickoff(event)
        opens, closes = kick - LEAD, kick - LATEST
        entry = {"game": label(event), "kickoff": kick, "state": None, "at": None, "by": None, "why": "", "failed_at": []}
        inside = sorted((run for run in (recent_runs or []) if opens <= run["at"] <= kick), key=lambda run: run["at"])
        good = [run["at"] for run in inside if run["status"] == "completed" and run["conclusion"] == "success"]
        going = [run["at"] for run in inside if run["status"] != "completed"]
        failed = [run["at"] for run in inside if run["status"] == "completed" and run["conclusion"] != "success"]
        entry["failed_at"] = failed
        count = f"{len(failed)} time{'' if len(failed) == 1 else 's'}"
        if good:
            entry.update(state="done", at=good[0], why=f"a board run that started at {az(good[0])} finished successfully")
        elif going:
            entry.update(state="running", at=going[0], why=f"a board run that started at {az(going[0])} is still going")
        elif len(failed) > RETRIES:
            entry.update(state="failed", why=f"the run failed {count}, at {listed([clock(at) for at in failed])}, "
                                             f"and after {RETRIES} retries it was not tried again")
        elif now > closes and failed:
            entry.update(state="failed", why=f"the run failed {count}, at {listed([clock(at) for at in failed])}, "
                                             "and there was no time left before kickoff to try again")
        elif now > closes:
            entry.update(state="late", why="less than 25 minutes to kickoff")
        elif now >= opens and failed:
            entry.update(state="due", at=now, why=f"retry {len(failed)} of {RETRIES}: the run at {clock(failed[-1])} failed")
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
        elif entry["state"] == "due":
            what = "DUE NOW: starting a board run" + (f" ({entry['why']})" if entry["failed_at"] else "")
        else:
            what = {"done": "done", "running": "waiting", "late": "skipped", "failed": "GAVE UP"}[entry["state"]] + f": {entry['why']}"
        lines.append(f"  {entry['game']:<9}{az(entry['kickoff']):<25}{what}")
    return lines


# ----------------------------------------------------------------- notices


def load_notices() -> list[dict]:
    """The notices now published. A missing or unreadable file is the same as none."""
    try:
        found = json.loads(NOTICES_FILE.read_text(encoding="utf-8")).get("notices", [])
        return [notice for notice in found if isinstance(notice, dict) and "kickoff_utc" in notice and "text" in notice]
    except (OSError, ValueError, AttributeError):
        return []


def updated_notices(now: datetime, entries: list[dict], recent_runs: list[dict], existing: list[dict]) -> list[dict]:
    """The notices that should be published after this check: one for each kickoff time whose run after
    inactives was given up on. A notice comes down when a run in that window succeeds after all, and
    otherwise NOTICE_LIFE after kickoff."""
    by_kick: dict[datetime, list[dict]] = {}
    for entry in entries:
        by_kick.setdefault(entry["kickoff"], []).append(entry)
    old = {notice["kickoff_utc"]: notice for notice in existing}
    keep = []
    for kick, group in by_kick.items():
        key = kick.isoformat(timespec="minutes")
        state = group[0]["state"]  # games with one kickoff time share a window, so they share a state
        if state == "failed":
            games = [entry["game"] for entry in group]
            text = f"The board was not rebuilt after inactives for {', '.join(games)} (kickoff {az(kick)}): {group[0]['why']}."
            posted = old[key] if key in old else {"posted_utc": now.isoformat(timespec="seconds"), "posted_arizona": az(now)}
            keep.append({"kickoff_utc": key, "games": games, "posted_utc": posted["posted_utc"],
                         "posted_arizona": posted["posted_arizona"], "text": text})
        elif state != "done" and key in old:
            keep.append(old[key])  # a new run is going or about to start: the notice stays until one succeeds
    listed_now = {kick.isoformat(timespec="minutes") for kick in by_kick}
    for key, notice in old.items():
        if key in listed_now:
            continue
        kick = datetime.fromisoformat(key)  # no longer in the feed, so the game has kicked off
        rebuilt = any(run["status"] == "completed" and run["conclusion"] == "success" and kick - LEAD <= run["at"] <= kick
                      for run in recent_runs)
        if not rebuilt and now <= kick + NOTICE_LIFE:
            keep.append(notice)
    return sorted(keep, key=lambda notice: notice["kickoff_utc"])


def save_notices(notices: list[dict]) -> bool:
    """Write docs/notices.json if it would change. Returns True when it was written, so the workflow has
    something to commit. Most checks change nothing and write nothing."""
    if notices == load_notices() and NOTICES_FILE.exists():
        return False
    NOTICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTICES_FILE, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"notices": notices}, indent=1) + "\n")
    return True


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


def recent_board_runs(repo: str, token: str | None, since: datetime) -> list[dict]:
    """"Run the prop board" runs created since `since`, whoever or whatever started them: when each started,
    GitHub's status ("completed" once it is over) and its conclusion ("success", "failure", "cancelled", ...)."""
    created = urllib.parse.quote(">=" + since.strftime("%Y-%m-%dT%H:%M:%SZ"))
    status, data = github("GET", f"/repos/{repo}/actions/workflows/{BOARD_WORKFLOW.name}/runs?per_page=30&created={created}", token)
    if status != 200:
        raise RuntimeError(f"could not list recent board runs (HTTP {status}): {data.get('message', '')}")
    return [{"at": datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")), "status": run.get("status"),
             "conclusion": run.get("conclusion")} for run in data.get("workflow_runs", [])]


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
    code = start_if_due(entries, repo, token, args.trigger)
    if args.trigger:  # a problem with the notice file must never stop a run from being started, so it comes last
        try:
            if save_notices(updated_notices(now, entries, runs, load_notices())):
                print("The notice file changed. The workflow publishes it in its next step.")
        except OSError as exc:
            print(f"The notice file could not be written: {exc}")
    return code


def start_if_due(entries: list[dict], repo: str | None, token: str | None, trigger: bool) -> int:
    """Start one board run if any game is due. Returns the exit code for the check."""
    due = [entry for entry in entries if entry["state"] == "due"]
    if not due:
        print("Nothing is due.")
        return 0
    games = ", ".join(entry["game"] for entry in due)
    if not trigger:
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
