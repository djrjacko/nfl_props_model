"""Build the board the website shows.

    python src/board.py              # games kicking off in the next 36 hours
    python src/board.py --hours 60   # what the automatic runs use
    python src/board.py --literal    # price with the rules exactly as written

It prices the cached odds (fetch_odds.py --pull) with the cached player status (fetch_status.py) and the
nflverse data on disk, then writes:

    docs/picks.json     everything the Board tab shows
    docs/board.txt      the same board as plain text, for pasting into Notepad
    docs/changes.json   what changed since the previous run, in plain words
    logs/picks_log.csv  one row per recommended leg per run, so results can be graded later

Same-game tickets. Legs are grouped by game script (see model.py): everything in a group is helped by the
same team being ahead. A ticket takes one leg per player from one group, all priced at the same book. The
price shown is the legs multiplied together; a book prices a same-game parlay shorter than that when the
legs move together, so check the real price before using a boost on it.

Two rules decide which legs a ticket may use:

  - A leg the calibrated model and the rules as written disagree on (model.FLIP_FLAG) is never used.
  - The anchor rule (config.json "require_anchor", on by default): every ticket starts from an anchor leg,
    which is a running back's rush attempts or rush yards over, an anytime TD, or a team moneyline, and
    the other legs come from the same game script. So no ticket is made of unders alone. The model has no
    view on who wins a game, so a moneyline is carried at the book's own probability with the vig removed
    and adds no edge of its own. With the rule off, moneylines are not used and any same-script legs can
    be combined, as before.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta

import fetch_nflverse as nfl
import fetch_odds
import fetch_status
import model
import pregame
from common import ROOT, az, daily_board_runs, load_config, outside_schedule, utc_now

DOCS = ROOT / "docs"
PICKS_FILE, TEXT_FILE, CHANGES_FILE = DOCS / "picks.json", DOCS / "board.txt", DOCS / "changes.json"
PICKS_LOG = ROOT / "logs" / "picks_log.csv"
DEFAULT_HOURS = 36.0
UNDER_WARNING_SHARE = 0.75  # warn when more than this share of the legs are unders
SUGGESTED_PER_SIZE = 4  # same-game tickets listed for each size
PRICE_MOVE = 15  # cents of American odds before a price change is worth mentioning
STALE_HOURS = 3.0  # odds or status older than this get a note on the board
LOG_FIELDS = ["run_utc", "run_arizona", "game_id", "game", "kickoff_utc", "player", "player_id", "team", "opponent",
              "position", "market", "side", "line", "book", "price", "model_prob", "literal_prob", "implied_prob", "edge",
              "tier", "flags", "tags", "status", "correlation_group", "projection"]
LEG_FIELDS = ["id", "game", "game_id", "kickoff_utc", "player", "player_id", "position", "team", "opponent", "market",
              "market_key", "side", "line", "book", "price", "books", "projection", "model_prob", "implied_prob", "edge",
              "literal_prob", "literal_agrees", "tier", "flags", "tags", "rooting_conflict", "status", "correlation_group",
              "correlation_role", "reasoning"]
MONEYLINE = "moneyline"
ANCHOR_RUSH_MARKETS = ("player_rush_attempts", "player_rush_yds")  # only running backs are priced in these
ANCHOR_RULE = "an RB rush attempts or rush yards over, an anytime TD or a team moneyline"


# ---------------------------------------------------------------- small pieces


def american_to_decimal(price: float) -> float:
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / -price)


def decimal_to_american(decimal: float) -> int:
    return round((decimal - 1.0) * 100.0) if decimal >= 2.0 else round(-100.0 / (decimal - 1.0))


def bet_text(leg: dict) -> str:
    """'under 12.5 rush attempts', 'anytime TD' or 'moneyline'."""
    if leg["line"] is None:
        return leg["market"]
    return f"{leg['side']} {leg['line']:g} {leg['market']}"


def leg_text(leg: dict) -> str:
    return f"{leg['player']} {bet_text(leg)} ({leg['book']} {leg['price']:+d})"


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def repo_name() -> str | None:
    """'owner/name' from GitHub Actions or the git remote, so nothing about the account is written in code."""
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    try:
        url = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "remote.origin.url"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if "github.com" not in url:
        return None
    tail = url.split("github.com")[-1].lstrip(":/")
    return tail[:-4] if tail.endswith(".git") else tail or None


def next_runs(now: datetime, count: int = 5) -> list[dict]:
    """The next runs to expect: the outside scheduler's daily runs (config.json describes them), and the run
    the pregame check starts once inactives are due for each kickoff time in the odds cache."""
    runs: dict[datetime, list[str]] = {moment: ["daily run"] for moment in daily_board_runs(now, now + timedelta(days=3))}
    if fetch_odds.EVENTS_FILE.exists():
        events = json.loads(fetch_odds.EVENTS_FILE.read_text(encoding="utf-8"))["data"]
        waiting: dict[datetime, list[str]] = {}
        for entry in pregame.plan(now, events):
            if entry["state"] == "waiting" and entry["at"]:
                waiting.setdefault(entry["at"], []).append(entry["game"])
        for moment, games in waiting.items():
            runs.setdefault(moment, []).append(f"after inactives for {', '.join(games)}")
    return [{"utc": moment.isoformat(timespec="minutes"), "arizona": az(moment), "why": ", ".join(runs[moment])}
            for moment in sorted(runs)[:count]]


def schedule_summary() -> dict:
    """The outside scheduler's jobs in a form the site can show: clock times in Arizona time, the look-ahead
    its runs ask for, and the minutes of the hour at which it starts the pregame check."""
    jobs = outside_schedule()
    return {"board_runs": [datetime(2000, 1, 1, hour, minute).strftime("%I:%M %p").lstrip("0") for hour, minute in jobs["board_runs"]],
            "board_hours": jobs["board_hours"], "check_minutes": jobs["check_minutes"]}


# ------------------------------------------------------------- the under lean


def summarise(legs: list[dict]) -> dict:
    """The one-line count at the top of every board, and a warning when unders dominate."""
    overs = sum(1 for leg in legs if leg["side"] == "over")
    unders = sum(1 for leg in legs if leg["side"] == "under")
    tds = sum(1 for leg in legs if leg["side"] == "yes")
    share = unders / len(legs) if legs else 0.0
    return {
        "legs": len(legs), "overs": overs, "unders": unders, "tds": tds, "under_share": round(share, 3),
        "line": f"{plural(len(legs), 'leg')}: {plural(overs, 'over')}, {plural(unders, 'under')}, {plural(tds, 'TD')}",
        "warning": (f"{share:.0%} of this board is unders. The calibration may be shrinking too hard for this slate."
                    if share > UNDER_WARNING_SHARE else None),
    }


# --------------------------------------------------------- same-game tickets


def is_anchor(leg: dict) -> bool:
    """A leg a ticket may be built on: a running back's rush attempts or rush yards over, an anytime TD, or a
    team moneyline. An under is never an anchor."""
    if leg.get("market_key") == MONEYLINE or leg["side"] == "yes":
        return True
    return leg["side"] == "over" and leg.get("market_key") in ANCHOR_RUSH_MARKETS


def moneyline_legs(slate: list[dict], rooting: set[str] | frozenset[str] = frozenset()) -> list[dict]:
    """One leg-shaped entry for each team with a posted moneyline, for anchoring tickets only.

    They are never picks. The model has no view on who wins, so each one carries the book's own probability
    with the vig removed and an edge of zero. A moneyline sits on the script "that team ahead". It is a
    rooting conflict when the other team is one the owner roots for.
    """
    out = []
    for game in slate:
        for team, books in (game.get("moneylines") or {}).items():
            opponent = game["home"] if team == game["away"] else game["away"]
            conflict = opponent in rooting
            out.append({
                "id": f"{game['game']}|{MONEYLINE}|{team}", "game": game["game"], "player": team, "player_id": f"team:{team}",
                "team": team, "opponent": opponent, "market": "moneyline", "market_key": MONEYLINE, "side": "win",
                "line": None, "tier": None, "flags": [], "tags": ["rooting conflict"] if conflict else [],
                "rooting_conflict": conflict, "literal_agrees": True, "correlation_group": f"{game['game']}: {team} ahead",
                "offers": [{"book": book, "line": None, "price": offer["price"], "model_prob": offer["fair"],
                            "implied_prob": offer["fair"], "edge": 0.0} for book, offer in sorted(books.items())],
            })
    return out


def ticket_candidates(legs: list[dict], size: int, moneylines: list[dict] | tuple = (),
                      require_anchor: bool = True) -> list[dict]:
    """Every ticket of `size` legs that shares a game script, uses one leg per player and sits at one book.

    A leg whose calibrated and as-written probabilities disagree is never used. With `require_anchor`, a
    ticket must hold an anchor leg (see is_anchor), so none is made of unders alone, and `moneylines` are
    offered as anchors; the anchor recorded is the likeliest of the ticket's anchor legs. Without it the
    moneylines are ignored, any same-script legs can be combined and the anchor is simply the likeliest leg.
    """
    groups: dict[str, list[dict]] = {}
    for leg in list(legs) + (list(moneylines) if require_anchor else []):
        if leg.get("correlation_group") and leg.get("literal_agrees", True):
            groups.setdefault(leg["correlation_group"], []).append(leg)
    tickets = []
    for group, members in groups.items():
        for book in ("DK", "FD"):
            # a moneyline has no model edge to clear; every other leg needs the minimum edge at this book
            priced = [(leg, offer) for leg in members for offer in leg["offers"]
                      if offer["book"] == book and (offer["edge"] >= model.MIN_EDGE or leg.get("market_key") == MONEYLINE)]
            for combo in itertools.combinations(priced, size):
                if len({leg["player_id"] for leg, _ in combo}) < size:
                    continue
                anchors = [pair for pair in combo if is_anchor(pair[0])] if require_anchor else list(combo)
                if not anchors:
                    continue  # the anchor rule: nothing to build on, for example a ticket of unders alone
                decimal = model_prob = implied = 1.0
                for _, offer in combo:
                    decimal *= american_to_decimal(offer["price"])
                    model_prob *= offer["model_prob"]
                    implied *= offer["implied_prob"]
                anchor = max(anchors, key=lambda pair: pair[1]["model_prob"])[0]
                tickets.append({
                    "size": size, "group": group, "game": combo[0][0]["game"], "book": book,
                    "legs": [{"id": leg["id"], "player": leg["player"], "team": leg["team"], "bet": bet_text({**leg, "line": offer["line"]}),
                              "price": offer["price"], "model_prob": offer["model_prob"], "edge": offer["edge"],
                              "tier": leg["tier"], "flagged": bool(leg["flags"]), "anchor": leg["id"] == anchor["id"],
                              "moneyline": leg.get("market_key") == MONEYLINE} for leg, offer in combo],
                    "leg_ids": sorted(leg["id"] for leg, _ in combo), "anchor": anchor["id"], "anchor_player": anchor["player"],
                    "decimal": round(decimal, 3), "price": decimal_to_american(decimal),
                    # the legs are treated as independent here; legs that share a script hit together more often
                    # than that, so this understates the chance and the book's parlay price will be shorter too
                    "model_prob": round(model_prob, 4), "implied_prob": round(implied, 4),
                    "value": round(model_prob * decimal - 1.0, 4),
                    "rooting_conflict": any(leg["rooting_conflict"] for leg, _ in combo),
                    "flagged_legs": sum(1 for leg, _ in combo if leg["flags"]),
                })
    return tickets


def suggest(tickets: list[dict], limit: int = SUGGESTED_PER_SIZE) -> list[dict]:
    """The best few tickets by value, at most two per game script and without leaning on one leg more than twice."""
    chosen, uses, per_group = [], {}, {}
    for ticket in sorted(tickets, key=lambda t: -t["value"]):
        if per_group.get(ticket["group"], 0) >= 2 or any(uses.get(leg_id, 0) >= 2 for leg_id in ticket["leg_ids"]):
            continue
        chosen.append(ticket)
        per_group[ticket["group"]] = per_group.get(ticket["group"], 0) + 1
        for leg_id in ticket["leg_ids"]:
            uses[leg_id] = uses.get(leg_id, 0) + 1
        if len(chosen) == limit:
            break
    return chosen


def allocate_boosts(tickets_by_size: dict[int, list[dict]], rules: list[dict], require_anchor: bool = False) -> list[dict]:
    """One ticket per boost. The bigger boost chooses first, and no ticket's anchor leg appears on another.
    `require_anchor` only changes the wording when nothing can be offered."""
    kind = "anchored same-script" if require_anchor else "same-script"
    taken_anchors: set[str] = set()
    taken_legs: set[str] = set()
    out = []
    for rule in sorted(rules, key=lambda r: -r["boost"]):
        pool = tickets_by_size.get(rule["legs"], [])
        in_range = [t for t in pool if rule["min_price"] <= t["price"] <= rule["max_price"]]
        free = [t for t in in_range if t["anchor"] not in taken_legs and not (set(t["leg_ids"]) & taken_anchors)]

        def boosted_value(ticket: dict) -> float:
            return ticket["model_prob"] * (1.0 + (1.0 + rule["boost"]) * (ticket["decimal"] - 1.0)) - 1.0

        key = (lambda t: t["model_prob"]) if rule["pick"] == "safest" else boosted_value
        entry = {"name": rule["name"], "boost": rule["boost"], "max_stake": rule["max_stake"], "legs": rule["legs"],
                 "range": f"{rule['min_price']:+d} to {rule['max_price']:+d}", "pick": rule["pick"], "ticket": None,
                 "alternative": None, "note": None}

        def with_boost(ticket: dict) -> dict:
            boosted_decimal = 1.0 + (1.0 + rule["boost"]) * (ticket["decimal"] - 1.0)
            return {**ticket, "boosted_price": decimal_to_american(boosted_decimal), "boosted_value": round(boosted_value(ticket), 4)}

        if free:
            best = max(free, key=key)
            entry["ticket"] = with_boost(best)
            # the pick stays even if it goes against a team the owner roots for; the next-best clean ticket is
            # listed beside it so there is something to use if he skips it
            clean = [t for t in free if not t["rooting_conflict"]]
            if best["rooting_conflict"] and clean:
                entry["alternative"] = with_boost(max(clean, key=key))
            taken_anchors.add(best["anchor"])
            taken_legs.update(best["leg_ids"])
        elif in_range:
            entry["note"] = (f"Every {rule['legs']}-leg ticket in the {entry['range']} range shares its anchor leg with the "
                             "other boost's ticket, so none is offered.")
        elif pool:
            nearest = min(pool, key=lambda t: min(abs(t["price"] - rule["min_price"]), abs(t["price"] - rule["max_price"])))
            entry["note"] = (f"No {rule['legs']}-leg {kind} ticket is priced {entry['range']} on this slate. "
                             f"The nearest is {nearest['price']:+d} at {nearest['book']}: "
                             + " + ".join(f"{leg['player']} {leg['bet']}" for leg in nearest["legs"]) + ".")
        else:
            entry["note"] = (f"No {rule['legs']}-leg {kind} ticket can be built from this board's legs."
                             + (f" A ticket has to start from {ANCHOR_RULE}." if require_anchor else ""))
        out.append(entry)
    return sorted(out, key=lambda e: e["boost"])


# -------------------------------------------------------------- what changed


def diff(previous: dict | None, board: dict, priced: list[dict], now: datetime) -> dict:
    """Compare this board with the previous one and say what happened in plain words."""
    changes = {"generated_utc": board["generated_utc"], "generated_arizona": board["generated_arizona"],
               "previous_utc": None, "previous_arizona": None, "headline": "", "added": [], "removed": [],
               "line_moved": [], "price_moved": [], "status_changed": [], "tier_changed": [], "kicked_off": 0}
    if not previous or "legs" not in previous:
        changes["headline"] = "First run: there is no earlier board to compare with."
        return changes
    changes["previous_utc"] = previous.get("generated_utc")
    changes["previous_arizona"] = previous.get("generated_arizona")
    old = {leg["id"]: leg for leg in previous["legs"]}
    new = {leg["id"]: leg for leg in board["legs"]}
    old_games = {game["game"] for game in previous.get("slate", [])}
    new_games = {game["game"] for game in board["slate"]}
    gated = {g["id"]: g["reason"] for g in board["gated"]}
    gated_players = {g["player"]: g["reason"] for g in board["gated"]}
    skipped = {s["id"]: s for s in board["skips"]}
    now_priced = {(p["player_id"], p["market_key"]): p for p in priced}

    for leg_id, leg in new.items():
        if leg_id in old:
            continue
        why = ("its game is now inside the window" if leg["game"] not in old_games
               else f"the edge is now {leg['edge']:.1%}")
        changes["added"].append({"id": leg_id, "text": f"Added: {leg_text(leg)}, {leg['tier']} tier, because {why}."})

    for leg_id, leg in old.items():
        if leg_id in new:
            continue
        if datetime.fromisoformat(leg["kickoff_utc"]) <= now:
            changes["kicked_off"] += 1
            continue
        current = now_priced.get((leg["player_id"], leg["market_key"]))
        if leg_id in gated or leg["player"] in gated_players:
            why = f"ruled out: {gated.get(leg_id) or gated_players[leg['player']]}"
        elif leg_id in skipped and skipped[leg_id]["kind"] == "not trusted":
            why = f"the model's edge grew to {skipped[leg_id]['edge']:.0%}, which is too large to trust"
        elif leg_id in skipped and skipped[leg_id]["kind"] == "blowout":
            why = "the blowout rule now applies to this game"
        elif leg["game"] not in new_games:
            why = "its game is outside this run's window"
        elif leg.get("position") == "QB" and leg.get("market_key") == "player_anytime_td":
            why = "quarterback anytime TDs are no longer priced"
        elif current is None:
            why = "the books no longer offer it"
        elif leg["line"] is not None and current["line"] is not None and current["line"] != leg["line"]:
            why = f"line moved {leg['line']:g} to {current['line']:g} and the edge is now {current['edge']:.1%}"
        else:
            why = f"the edge fell from {leg['edge']:.1%} to {current['edge']:.1%}"
        changes["removed"].append({"id": leg_id, "text": f"Removed: {leg_text(leg)}, {why}."})

    for leg_id in old.keys() & new.keys():
        before, after = old[leg_id], new[leg_id]
        name = f"{after['player']} {after['market']}"
        if before["line"] != after["line"] and before["line"] is not None and after["line"] is not None:
            changes["line_moved"].append({"id": leg_id, "text": (
                f"Line moved: {name} {before['line']:g} to {after['line']:g} ({after['book']}); "
                f"still {after['side']}, edge {after['edge']:.1%}.")})
        elif before["book"] != after["book"] or abs(before["price"] - after["price"]) >= PRICE_MOVE:
            changes["price_moved"].append({"id": leg_id, "text": (
                f"Price moved: {after['player']} {bet_text(after)} was {before['book']} {before['price']:+d}, "
                f"now {after['book']} {after['price']:+d}; edge {after['edge']:.1%}.")})
        if before.get("status") != after.get("status"):
            changes["status_changed"].append({"id": leg_id, "text": (
                f"Status changed: {after['player']} was {before.get('status')}, now {after.get('status')}.")})
        if before["tier"] != after["tier"]:
            fresh = [flag for flag in after["flags"] if flag not in before["flags"]]
            why = f"new flag: {'; '.join(fresh)}" if fresh else f"edge {before['edge']:.1%} to {after['edge']:.1%}"
            changes["tier_changed"].append({"id": leg_id, "text": (
                f"Tier changed: {after['player']} {bet_text(after)} went from {before['tier']} to {after['tier']} ({why}).")})

    was_gated = {g["reason"] for g in previous.get("gated", [])}
    for reason in sorted({g["reason"] for g in board["gated"]} - was_gated):
        changes["status_changed"].append({"id": None, "text": f"Ruled out: {reason}. Legs that depend on it are off the board."})

    counts = [(len(changes["added"]), "added"), (len(changes["removed"]), "removed"),
              (len(changes["line_moved"]), "line moved"), (len(changes["price_moved"]), "price moved"),
              (len(changes["status_changed"]), "status changed"), (len(changes["tier_changed"]), "tier changed")]
    said = ", ".join(f"{count} {label}" for count, label in counts if count)
    changes["headline"] = (said or "Nothing changed") + f" since {changes['previous_arizona'] or 'the previous run'}."
    if changes["kicked_off"]:
        changes["headline"] += f" {plural(changes['kicked_off'], 'earlier leg')} dropped off because the game kicked off."
    return changes


# --------------------------------------------------------------- the outputs


def build(hours: float) -> tuple[dict, dict]:
    """Returns (board for picks.json, model result). Reads caches only."""
    now = utc_now()
    config = load_config()
    result = model.run()
    until = now + timedelta(hours=hours)
    slate = [g for g in result["games"] if now < datetime.fromisoformat(g["kickoff_utc"]) <= until]
    games = {g["game"] for g in slate}
    teams = {team for g in slate for team in (g["away"], g["home"])}
    legs = [leg for leg in result["legs"] if leg["game"] in games]
    skips = [skip for skip in result["skips"] if skip["game"] in games]
    gated = [g for g in result["gated"] if g["team"] in teams]
    for game in slate:
        game["kickoff_arizona"] = az(datetime.fromisoformat(game["kickoff_utc"]))

    require_anchor = bool(config.get("require_anchor", True))
    anchors = moneyline_legs(slate, set(config["rooting_teams"])) if require_anchor else []
    tickets = {size: ticket_candidates(legs, size, anchors, require_anchor) for size in (2, 3)}
    notes = list(result["notes"])
    flipped = sum(1 for leg in legs if leg.get("literal_agrees") is False)
    if flipped:
        notes.append(f"{plural(flipped, 'leg')} on this board carry the flag \"{model.FLIP_FLAG}\": the calibrated model and "
                     "the rules as written point opposite ways, so they sit in B tier and are left out of every ticket.")
    sources = {"odds_pulled_utc": None, "status_generated_utc": None, "status_source": None,
               "nflverse_fetched_utc": nfl.run_info()["fetched_utc"], "last_completed_week": nfl.run_info()["last_completed_week"]}
    credits = {"this_run": None, "used": None, "remaining": None}
    if fetch_odds.LAST_PULL_FILE.exists():
        pull = json.loads(fetch_odds.LAST_PULL_FILE.read_text(encoding="utf-8"))
        sources["odds_pulled_utc"] = pull["pulled_utc"]
        credits = {"this_run": pull.get("cost"), "used": pull.get("used"), "remaining": pull.get("remaining")}
        age = (now - datetime.fromisoformat(pull["pulled_utc"])).total_seconds() / 3600
        if age > STALE_HOURS:
            notes.append(f"The odds on this board were pulled {age:.0f} hours ago.")
    try:
        status = fetch_status.load()
        sources.update(status_generated_utc=status["generated_utc"], status_source=status["source"])
        age = (now - datetime.fromisoformat(status["generated_utc"])).total_seconds() / 3600
        if age > STALE_HOURS:
            notes.append(f"Player status was last checked {age:.0f} hours ago.")
        if status["source"] != "espn":
            notes.append("ESPN could not be reached for every team, so some statuses come from nflverse and may be incomplete.")
    except FileNotFoundError:
        notes.append("No player status file was found, so nobody could be checked against the injury gate.")
    if not slate:
        notes.append(f"No games with posted odds kick off in the next {hours:g} hours.")

    board = {
        "generated_utc": now.isoformat(timespec="seconds"), "generated_arizona": az(now), "repo": repo_name(),
        "window_hours": hours, "season": result["season"], "slate": slate, "summary": summarise(legs),
        "legs": [{k: leg.get(k) for k in LEG_FIELDS} for leg in legs], "skips": skips, "gated": gated,
        "sgps": {"two_leg": suggest(tickets[2]), "three_leg": suggest(tickets[3])},
        "boosts": allocate_boosts(tickets, config["boosts"], require_anchor),
        "anchor_rule": require_anchor,
        "credits": credits, "sources": sources, "notes": notes, "next_runs": next_runs(now),
        "schedule": schedule_summary(),
        "calibration": "off (rules as written)" if model.CALIBRATION["rush_att"] == (0.0, 1.0) else "on",
        "rooting_teams": config["rooting_teams"],
    }
    return board, result


def ticket_lines(ticket: dict, indent: str = "    ") -> list[str]:
    head = (f"{ticket['book']} about {ticket['price']:+d}: model {ticket['model_prob']:.0%} against {ticket['implied_prob']:.0%} "
            f"implied [{ticket['group']}]" + ("  [rooting conflict]" if ticket["rooting_conflict"] else ""))
    def mark(leg: dict) -> str:
        if leg.get("moneyline"):
            return "  [anchor: the book's own odds, no model edge]" if leg.get("anchor") else "  [the book's own odds, no model edge]"
        return ("  [anchor]" if leg.get("anchor") else "") + ("  (flagged)" if leg["flagged"] else "")

    return [indent + head] + [f"{indent}  - {leg['player']} {leg['bet']} ({leg['price']:+d}){mark(leg)}" for leg in ticket["legs"]]


def render_text(board: dict, changes: dict) -> str:
    """The board as plain ASCII text."""
    out: list[str] = []
    wrap = lambda text, indent: textwrap.wrap(text, width=96, initial_indent=indent, subsequent_indent=indent)  # noqa: E731
    credits = board["credits"]
    spent = (f"Odds API credits: {credits['this_run']} this run, {credits['used']:,} used, {credits['remaining']:,} left"
             if credits["remaining"] is not None else "Odds API credits: not recorded")
    out += ["NFL PROP BOARD", f"Updated {board['generated_arizona']} | {spent}",
            f"Window: games kicking off in the next {board['window_hours']:g} hours | calibration {board['calibration']}", ""]
    for game in board["slate"]:
        out.append(f"  {game['game']:<9}{game['kickoff_arizona']:<25}{game['home']} {game['home_spread']:+g}, total {game['total']:g}"
                   + ("  BLOWOUT RULE ON" if game["blowout"] else ""))
    out += ["", board["summary"]["line"]]
    if board["summary"]["warning"]:
        out.append(f"WARNING: {board['summary']['warning']}")

    anchored = board.get("anchor_rule", False)
    out += ["", "BOOSTS"]
    out += wrap(f"Anchor rule on: every ticket starts from {ANCHOR_RULE}, then adds legs on the same game script."
                if anchored else "Anchor rule off: a ticket can be any legs that share a game script.", "  ")
    for boost in board["boosts"]:
        out.append(f"  {boost['name']} (${boost['max_stake']} max, {boost['legs']} legs, {boost['range']}, {boost['pick']})")
        if boost["ticket"]:
            out += ticket_lines(boost["ticket"])
            out.append(f"      with the boost about {boost['ticket']['boosted_price']:+d}")
            if boost["alternative"]:
                out.append("    if you skip that one for the rooting conflict:")
                out += ticket_lines(boost["alternative"])
                out.append(f"      with the boost about {boost['alternative']['boosted_price']:+d}")
        else:
            out += wrap(boost["note"], "    ")
    out += ["  Ticket prices are the legs multiplied together. A book prices a same-game parlay shorter than",
            "  that when the legs move together, so check the real price is still in range before boosting."]

    for label, key in (("SAME GAME PARLAYS, 2 LEGS", "two_leg"), ("SAME GAME PARLAYS, 3 LEGS", "three_leg")):
        out += ["", label]
        if not board["sgps"][key]:
            out.append("  none: no anchor leg shares a game script with enough other legs at the same book" if anchored
                       else "  none: no legs share a game script at the same book")
        for ticket in board["sgps"][key]:
            out += ticket_lines(ticket, "  ")

    for tier, title in (("A", "A TIER (edge 8% or more, no flags)"), ("B", "B TIER (edge 5% to 8%, or any flag)")):
        rows = [leg for leg in board["legs"] if leg["tier"] == tier]
        out += ["", f"{title}: {len(rows)}"]
        for number, leg in enumerate(rows, 1):
            tags = "".join(f"  [{tag}]" for tag in leg["tags"])
            out.append(f"  {number}. {leg['player']} ({leg['team']} {leg['position']}, {leg['game']}) {bet_text(leg)}  "
                       f"{leg['book']} {leg['price']:+d}{tags}")
            out.append(f"     model {leg['model_prob']:.1%} | book {leg['implied_prob']:.1%} | edge {leg['edge']:+.1%} | "
                       f"rules as written {leg['literal_prob']:.1%}" + (f" | {leg['status'].upper()}" if leg["status"] != "active" else ""))
            out += wrap(leg["reasoning"], "     ")
            for flag in leg["flags"]:
                out += wrap(f"flag: {flag}", "     ")

    out += ["", f"SKIPS: {len(board['skips'])}"]
    for skip in board["skips"]:
        out += wrap(f"[{skip['kind']}] {leg_text(skip)}: {skip['reason']}", "  ")
    out += ["", "WHAT CHANGED", f"  {changes['headline']}"]
    for key in ("status_changed", "removed", "added", "line_moved", "price_moved", "tier_changed"):
        for item in changes[key]:
            out += wrap(item["text"], "  ")
    if board["notes"]:
        out += ["", "NOTES"]
        for note in board["notes"]:
            out += wrap(f"- {note}", "  ")
    return "\n".join(out) + "\n"


def append_log(board: dict) -> int:
    """One row per recommended leg per run."""
    PICKS_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not PICKS_LOG.exists()
    with open(PICKS_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        for leg in board["legs"]:
            row = {field: leg.get(field) for field in LOG_FIELDS}
            row.update(run_utc=board["generated_utc"], run_arizona=board["generated_arizona"],
                       flags="; ".join(leg["flags"]), tags="; ".join(leg["tags"]))
            writer.writerow(row)
    return len(board["legs"])


def write_json(path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Price the cached odds and write the board files.")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="only games kicking off within this many hours (default 36)")
    parser.add_argument("--literal", action="store_true", help="use the rules exactly as written, with no calibration")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.literal:
        model.use_literal_rules()
    board, result = build(args.hours)
    previous = json.loads(PICKS_FILE.read_text(encoding="utf-8")) if PICKS_FILE.exists() else None
    changes = diff(previous, board, result["priced"], datetime.fromisoformat(board["generated_utc"]))
    text = render_text(board, changes)

    DOCS.mkdir(parents=True, exist_ok=True)
    write_json(PICKS_FILE, board)
    write_json(CHANGES_FILE, changes)
    TEXT_FILE.write_text(text, encoding="utf-8", newline="\n")
    logged = append_log(board)
    print(text)
    print(f"wrote {PICKS_FILE.relative_to(ROOT)}, {TEXT_FILE.relative_to(ROOT)}, {CHANGES_FILE.relative_to(ROOT)}; "
          f"{plural(logged, 'row')} added to {PICKS_LOG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
