"""Pull NFL game lines and player props from The Odds API.

By default this is a DRY RUN: one free call lists the games kicking off in the
look-ahead window, and the script reports what a real pull would cost. Credits
are only spent with --pull.

    python src/fetch_odds.py                     # dry run, free
    python src/fetch_odds.py --pull              # game lines, then props for each game
    python src/fetch_odds.py --hours 60          # look further ahead than 36 hours
    python src/fetch_odds.py --pull --max-age 0  # ignore cached responses

Credit rules (the-odds-api.com/liveapi/guides/v4, read 2026-11-24):
    /events              free
    /odds                markets asked for x bookmaker groups  -> spreads, totals = 2
    /events/{id}/odds    markets returned  x bookmaker groups  -> up to 5 per game, 7 with alt lines
Up to 10 bookmakers count as one group; only DraftKings and FanDuel are requested.

What real responses look like (first pull, GB@LA, 2026-11-24):
    * Over/Under markets carry both sides for each player: name is "Over" or
      "Under", description is the player's name, point is the line, price is
      American odds.
    * player_anytime_td has "Yes" prices only (no "No" side and no point), and
      its list includes team defenses ("Green Bay Packers D/ST", "... Defense").
    * FanDuel returned no player_rush_attempts market; DraftKings did.
    * The two books often post different lines for the same player.

The key comes from the ODDS_API_KEY environment variable (the GitHub Actions
secret) or from .env in the project folder. It is never printed, logged or
cached. Raw responses are cached in data/odds/, which is gitignored because the
API's terms do not allow republishing its data as downloadable files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from common import ROOT, TEAM_ABBR, az, utc_now

API = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
BOOKMAKERS = ("draftkings", "fanduel")
GAME_MARKETS = ("spreads", "totals")
PROP_MARKETS = (
    "player_rush_attempts",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",
    "player_anytime_td",
)
# Blowout rule: when the spread is 10 or more the board recommends a lower alt
# yardage line, so those games also need the alternate markets. (The Odds API
# has no first-half player props, so alt lines are the only priced option.)
ALT_MARKETS = ("player_rush_yds_alternate", "player_reception_yds_alternate")
BLOWOUT_SPREAD = 10.0
DEFAULT_HOURS = 36.0
DEFAULT_MAX_AGE_MIN = 10.0

CACHE_DIR = ROOT / "data" / "odds"
PROPS_DIR = CACHE_DIR / "props"
EVENTS_FILE = CACHE_DIR / "events.json"
GAME_LINES_FILE = CACHE_DIR / "game_lines.json"
LAST_PULL_FILE = CACHE_DIR / "last_pull.json"
CREDIT_LOG = ROOT / "logs" / "odds_credits.csv"
LOG_FIELDS = ["time_utc", "time_arizona", "call", "event_id", "matchup", "markets_asked", "markets_returned",
              "cost", "used", "remaining"]

GAME_LINE_COLUMNS = ["event_id", "commence_time", "away", "home", "book", "home_spread", "home_spread_price",
                     "away_spread_price", "total", "over_price", "under_price"]
PROP_COLUMNS = ["event_id", "commence_time", "away", "home", "book", "market", "player", "side", "line", "price",
                "last_update", "fetched_utc"]


class OddsApiError(RuntimeError):
    """A problem talking to The Odds API. The message never contains the key."""


# ------------------------------------------------------------------- the key


def load_key() -> tuple[str, str]:
    """Return (key, where it came from). The key itself is never shown anywhere."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key, "the ODDS_API_KEY environment variable"
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.strip().partition("=")
            if name.strip() == "ODDS_API_KEY" and value.strip():
                return value.strip().strip("\"'"), ".env"
    raise OddsApiError("no API key found. Put it in .env as ODDS_API_KEY=... or set the ODDS_API_KEY environment variable.")


# ------------------------------------------------------------------ API calls


def _int(value: str | None) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def api_get(path: str, params: dict[str, str], key: str) -> tuple[Any, dict]:
    """GET one endpoint. Returns (parsed JSON, {"cost", "used", "remaining"} from the usage headers)."""
    problem = "no response"
    for attempt in range(1, 5):
        try:
            r = requests.get(API + path, params={**params, "apiKey": key}, timeout=30)
        except requests.RequestException as exc:  # the message can contain the URL, so strip the key
            problem = f"{type(exc).__name__}: {exc}".replace(key, "***")
            time.sleep(2 * attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:  # rate limited or server trouble: space the calls out
            problem = f"HTTP {r.status_code}"
            time.sleep(2 * attempt)
            continue
        credits = {
            "cost": _int(r.headers.get("x-requests-last")),
            "used": _int(r.headers.get("x-requests-used")),
            "remaining": _int(r.headers.get("x-requests-remaining")),
        }
        if r.status_code != 200:
            try:
                message = r.json().get("message", "")
            except ValueError:
                message = r.text[:200]
            raise OddsApiError(f"{path}: HTTP {r.status_code} {message}".replace(key, "***"))
        return r.json(), credits
    raise OddsApiError(f"{path}: gave up after 4 tries ({problem})")


def _z(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _abbr(team_name: str) -> str:
    if team_name not in TEAM_ABBR:
        raise OddsApiError(f"unknown team name from The Odds API: {team_name!r} (add it to TEAM_ABBR in common.py)")
    return TEAM_ABBR[team_name]


def _matchup(event: dict) -> str:
    return f"{_abbr(event['away_team'])}@{_abbr(event['home_team'])}"


def _kickoff(event: dict) -> datetime:
    return datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))


def list_events(key: str, hours: float) -> tuple[list[dict], dict, dict]:
    """Free call: games kicking off between now and now + hours, earliest first."""
    start = utc_now().replace(microsecond=0)
    end = start + timedelta(hours=hours)
    params = {"dateFormat": "iso", "commenceTimeFrom": _z(start), "commenceTimeTo": _z(end)}
    data, credits = api_get(f"/sports/{SPORT}/events", params, key)
    events = sorted(data, key=lambda e: e["commence_time"])
    for event in events:
        _matchup(event)  # fail now, before any credits are spent, if a team name is not recognised
    return events, credits, {"hours": hours, "start": start, "end": end, "params": params}


# ---------------------------------------------------------- cache and logging


def _save(path: Path, endpoint: str, params: dict, credits: dict, data: Any) -> dict:
    record = {"fetched_utc": utc_now().isoformat(timespec="seconds"), "endpoint": endpoint, "params": params,
              "credits": credits, "data": data}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} is missing. Run: python src/fetch_odds.py --pull")
    return json.loads(path.read_text(encoding="utf-8"))


def _fresh(path: Path, max_age_min: float) -> dict | None:
    """The cached response, if there is one younger than max_age_min minutes."""
    if max_age_min <= 0 or not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    age = utc_now() - datetime.fromisoformat(record["fetched_utc"])
    return record if age <= timedelta(minutes=max_age_min) else None


def _log(call: str, credits: dict, event: dict | None = None, asked: int = 0, returned: int = 0) -> None:
    """Append one row per paid-endpoint call to logs/odds_credits.csv."""
    CREDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CREDIT_LOG.exists()
    now = utc_now()
    with open(CREDIT_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "time_utc": now.isoformat(timespec="seconds"), "time_arizona": az(now), "call": call,
            "event_id": event["id"] if event else "", "matchup": _matchup(event) if event else "",
            "markets_asked": asked, "markets_returned": returned,
            "cost": credits.get("cost"), "used": credits.get("used"), "remaining": credits.get("remaining"),
        })


# ------------------------------------------------- tidy tables for the model


def _game_line_rows(events: list[dict]) -> list[dict]:
    rows = []
    for ev in events:
        for book in ev.get("bookmakers", []):
            row = dict.fromkeys(GAME_LINE_COLUMNS)
            row.update(event_id=ev["id"], commence_time=ev["commence_time"], away=_abbr(ev["away_team"]),
                       home=_abbr(ev["home_team"]), book=book["key"])
            for market in book.get("markets", []):
                for out in market.get("outcomes", []):
                    if market["key"] == "spreads" and out["name"] == ev["home_team"]:
                        row["home_spread"], row["home_spread_price"] = out.get("point"), out.get("price")
                    elif market["key"] == "spreads" and out["name"] == ev["away_team"]:
                        row["away_spread_price"] = out.get("price")
                    elif market["key"] == "totals" and out["name"] == "Over":
                        row["total"], row["over_price"] = out.get("point"), out.get("price")
                    elif market["key"] == "totals" and out["name"] == "Under":
                        row["under_price"] = out.get("price")
            rows.append(row)
    return rows


def game_lines() -> pd.DataFrame:
    """Latest pull: one row per game per bookmaker. home_spread is negative when the home team is favoured."""
    return pd.DataFrame(_game_line_rows(_read(GAME_LINES_FILE)["data"]), columns=GAME_LINE_COLUMNS)


def _prop_rows(record: dict) -> list[dict]:
    ev = record["data"]
    base = {"event_id": ev["id"], "commence_time": ev["commence_time"], "away": _abbr(ev["away_team"]),
            "home": _abbr(ev["home_team"]), "fetched_utc": record["fetched_utc"]}
    rows = []
    for book in ev.get("bookmakers", []):
        for market in book.get("markets", []):
            for out in market.get("outcomes", []):
                rows.append({**base, "book": book["key"], "market": market["key"], "player": out.get("description"),
                             "side": out.get("name"), "line": out.get("point"), "price": out.get("price"),
                             "last_update": market.get("last_update")})
    return rows


def props() -> pd.DataFrame:
    """Latest pull: one row per game, bookmaker, market, player, side (Over/Under/Yes) and line."""
    rows: list[dict] = []
    for event in _read(EVENTS_FILE)["data"]:
        path = PROPS_DIR / f"{event['id']}.json"
        if path.exists():
            rows.extend(_prop_rows(json.loads(path.read_text(encoding="utf-8"))))
    return pd.DataFrame(rows, columns=PROP_COLUMNS)


# ------------------------------------------------------------- dry run / pull


def _header(title: str, source: str, window: dict) -> None:
    print(f"\n{title} | {az(utc_now())} | key loaded from {source} (not shown)")
    print(f"window: next {window['hours']:g} hours, to {az(window['end'])}\n")


def _credit_line(credits: dict) -> str:
    used, remaining = credits.get("used"), credits.get("remaining")
    if used is None or remaining is None:
        return "credits: the API did not report usage on this call"
    return f"credits this period: {used:,} used, {remaining:,} remaining"


def dry_run(key: str, source: str, hours: float, use_alts: bool) -> int:
    events, credits, window = list_events(key, hours)
    _header("ODDS API DRY RUN", source, window)
    if not events:
        print("No games kick off in this window, so a real pull would cost 0 credits.")
        print(_credit_line(credits))
        return 0
    print(f"games kicking off in the window ({len(events)}, from the free events call):")
    for event in events:
        print(f"  {_matchup(event):<9}{az(_kickoff(event)):<25}{event['away_team']} at {event['home_team']}")
    n = len(events)
    base = len(GAME_MARKETS) + n * len(PROP_MARKETS)
    extra = n * len(ALT_MARKETS) if use_alts else 0

    def cost_row(label: str, what: str, amount: str, note: str = "") -> None:
        print(f"  {label:<12}{what:<46}{amount:>9} credits{note}")

    print("\na real pull would cost:")
    cost_row("game lines", "spreads + totals, DraftKings + FanDuel", str(len(GAME_MARKETS)))
    cost_row("props", f"{n} game{'s' if n != 1 else ''} x {len(PROP_MARKETS)} markets", str(n * len(PROP_MARKETS)),
             "   (billed only for markets the books have posted)")
    if use_alts:
        cost_row("alt lines", f"+{len(ALT_MARKETS)} for each game with a spread of {BLOWOUT_SPREAD:g} or more",
                 f"0 to {extra}", "   (known once game lines are in)")
    cost_row("total", "", f"{base} to {base + extra}" if extra else str(base))
    print("\n" + _credit_line(credits))
    if credits.get("remaining") is not None:
        print(f"after the pull: at least {credits['remaining'] - base - extra:,} remaining")
    print("\nNothing was spent. To pull for real: python src/fetch_odds.py --pull")
    return 0


def _consensus_home_spread(game_line_events: list[dict]) -> dict[str, float]:
    """event id -> home spread averaged over the books that posted one."""
    lines = pd.DataFrame(_game_line_rows(game_line_events), columns=GAME_LINE_COLUMNS)
    lines = lines.dropna(subset=["home_spread"])
    return lines.groupby("event_id")["home_spread"].mean().to_dict()


def pull(key: str, source: str, hours: float, max_age_min: float, use_alts: bool) -> int:
    events, credits, window = list_events(key, hours)
    _header("ODDS API PULL", source, window)
    _save(EVENTS_FILE, f"/sports/{SPORT}/events", window["params"], credits, events)
    spent = 0
    summary: dict[str, Any] = {
        "pulled_utc": utc_now().isoformat(timespec="seconds"), "window_hours": hours,
        "window_start_utc": _z(window["start"]), "window_end_utc": _z(window["end"]), "games": [],
    }
    if not events:
        print("No games kick off in this window. Nothing pulled, nothing spent.")
        print(_credit_line(credits))
        summary.update(cost=0, used=credits.get("used"), remaining=credits.get("remaining"))
        LAST_PULL_FILE.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return 0

    # ---- game lines: one call for every game in the window
    wanted_ids = {e["id"] for e in events}
    record = _fresh(GAME_LINES_FILE, max_age_min)
    if record and wanted_ids <= {e["id"] for e in record["data"]}:
        note = "reused from cache, 0 credits"
    else:
        params = {"bookmakers": ",".join(BOOKMAKERS), "markets": ",".join(GAME_MARKETS), "oddsFormat": "american",
                  **window["params"]}
        data, credits = api_get(f"/sports/{SPORT}/odds", params, key)
        record = _save(GAME_LINES_FILE, f"/sports/{SPORT}/odds", params, credits, data)
        _log("game_lines", credits, asked=len(GAME_MARKETS), returned=len(GAME_MARKETS))
        spent += credits.get("cost") or 0
        note = f"{credits.get('cost')} credits"
    print(f"game lines ({note}):")
    lines = pd.DataFrame(_game_line_rows(record["data"]), columns=GAME_LINE_COLUMNS)
    for event in events:
        quotes = []
        for row in lines[lines["event_id"] == event["id"]].itertuples():
            spread = "no spread" if pd.isna(row.home_spread) else f"{row.home} {row.home_spread:+g}"
            total = "no total" if pd.isna(row.total) else f"total {row.total:g}"
            quotes.append(f"{row.book}: {spread}, {total}")
        print(f"  {_matchup(event):<9}{az(_kickoff(event)):<25}" + (" | ".join(quotes) or "no lines posted yet"))

    # ---- props: one call per game
    spreads = _consensus_home_spread(record["data"])
    print("\nprops (players priced per market, by book):")
    for event in events:
        blowout = abs(spreads.get(event["id"], 0.0)) >= BLOWOUT_SPREAD
        markets = PROP_MARKETS + (ALT_MARKETS if (use_alts and blowout) else ())
        path = PROPS_DIR / f"{event['id']}.json"
        cached = _fresh(path, max_age_min)
        if cached and set(markets) <= set(cached["params"]["markets"].split(",")):
            prop_record, cost_note, cost = cached, "reused from cache, 0 credits", 0
        else:
            params = {"bookmakers": ",".join(BOOKMAKERS), "markets": ",".join(markets), "oddsFormat": "american",
                      "dateFormat": "iso"}
            data, credits = api_get(f"/sports/{SPORT}/events/{event['id']}/odds", params, key)
            prop_record = _save(path, f"/sports/{SPORT}/events/{event['id']}/odds", params, credits, data)
            returned = {m["key"] for b in data.get("bookmakers", []) for m in b.get("markets", [])}
            _log("props", credits, event, asked=len(markets), returned=len(returned))
            cost = credits.get("cost") or 0
            spent += cost
            cost_note = f"{cost} credits"
            time.sleep(0.4)  # stay well inside the rate limit
        table = pd.DataFrame(_prop_rows(prop_record), columns=PROP_COLUMNS)
        returned_markets = sorted(table["market"].unique())
        print(f"  {_matchup(event)}: asked for {len(markets)} markets{' (blowout: alt lines added)' if len(markets) > len(PROP_MARKETS) else ''},"
              f" got {len(returned_markets)}, {cost_note}")
        if table.empty:
            print("      no props posted yet")
        else:
            counts = table.groupby(["market", "book"])["player"].nunique().unstack(fill_value=0)
            for line in counts.to_string().splitlines():
                print("      " + line)
        summary["games"].append({"event_id": event["id"], "matchup": _matchup(event), "commence_time": event["commence_time"],
                                 "markets_asked": list(markets), "markets_returned": returned_markets, "cost": cost})

    summary.update(cost=spent, used=credits.get("used"), remaining=credits.get("remaining"))
    LAST_PULL_FILE.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nthis run spent {spent} credits. " + _credit_line(credits))
    print(f"saved to {CACHE_DIR.relative_to(ROOT)} (not committed); credit log: {CREDIT_LOG.relative_to(ROOT)}")

    everything = props()
    if not everything.empty:
        print("\nfirst rows of the tidy props table the model will read:")
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            shown = everything.drop(columns=["event_id", "commence_time", "fetched_utc"]).head(8)
            print("\n".join("  " + line for line in shown.to_string(index=False).splitlines()))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Odds API: free dry run by default, --pull to spend credits.")
    parser.add_argument("--pull", action="store_true", help="spend credits: pull game lines and props")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="look-ahead window in hours (default 36)")
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_MIN,
                        help="reuse cached responses younger than this many minutes (default 10, 0 = always re-pull)")
    parser.add_argument("--no-alts", action="store_true", help="never request alt yardage lines for blowout games")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        key, source = load_key()
        if args.pull:
            return pull(key, source, args.hours, args.max_age, not args.no_alts)
        return dry_run(key, source, args.hours, not args.no_alts)
    except OddsApiError as exc:
        print(f"Odds API problem: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
