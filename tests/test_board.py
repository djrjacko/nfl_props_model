"""Unit tests for board.py: the under-lean summary, same-game tickets and their anchor rule, the boost rules
and the change log.

    python -m unittest discover -s tests -v

These use small made-up legs so every rule can be checked exactly.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import board  # noqa: E402
import common  # noqa: E402

NOW = datetime(2026, 11, 29, 16, 0, tzinfo=timezone.utc)
KICK = (NOW + timedelta(hours=3)).isoformat(timespec="minutes")
RULES = [
    {"name": "50% boost", "boost": 0.5, "max_stake": 50, "legs": 2, "min_price": 150, "max_price": 250, "pick": "safest"},
    {"name": "100% boost", "boost": 1.0, "max_stake": 25, "legs": 3, "min_price": 300, "max_price": 400, "pick": "best"},
]


MARKET_KEYS = {"rush attempts": "player_rush_attempts", "rush yards": "player_rush_yds", "receptions": "player_receptions",
               "receiving yards": "player_reception_yds", "anytime TD": "player_anytime_td"}


def leg(player: str, side: str = "over", price: int = -150, prob: float = 0.70, group: str = "AAA@BBB: BBB ahead",
        team: str = "BBB", line: float | None = 20.5, market: str = "rush yards", book: str = "DK", tier: str = "A",
        flags: list[str] | None = None, conflict: bool = False, status: str = "active", agrees: bool = True) -> dict:
    """A made-up leg. The default is a running back's rush yards over, which is an anchor."""
    implied = round(prob - 0.10, 4)
    if side == "yes":
        market = "anytime TD"
    return {
        "id": f"{player}|{market}|{side}", "game": "AAA@BBB", "game_id": "2026_13_AAA_BBB", "kickoff_utc": KICK,
        "player": player, "player_id": player, "position": "RB", "team": team, "opponent": "AAA" if team == "BBB" else "BBB",
        "market": market, "market_key": MARKET_KEYS[market], "side": side, "line": None if side == "yes" else line,
        "book": book, "price": price, "books": {}, "projection": 25.0, "model_prob": prob, "implied_prob": implied,
        "edge": round(prob - implied, 4), "literal_prob": prob, "literal_agrees": agrees, "tier": tier, "flags": flags or [],
        "tags": ["rooting conflict"] if conflict else [], "rooting_conflict": conflict, "status": status,
        "correlation_group": group, "correlation_role": "rb_volume", "reasoning": "made up",
        "offers": [{"book": book, "line": None if side == "yes" else line, "price": price, "model_prob": prob,
                    "implied_prob": implied, "edge": round(prob - implied, 4)}],
    }


def slate(lines: dict | None = None) -> list[dict]:
    """One game, AAA at BBB, with the moneylines given as {team: {book: (price, fair probability)}}."""
    lines = lines if lines is not None else {"BBB": {"DK": (-200, 0.64), "FD": (-210, 0.65)}, "AAA": {"DK": (170, 0.36), "FD": (176, 0.35)}}
    return [{"game": "AAA@BBB", "away": "AAA", "home": "BBB", "moneylines": {
        team: {book: {"price": price, "fair": fair} for book, (price, fair) in books.items()} for team, books in lines.items()}}]


def make_board(legs: list[dict], gated: list[dict] | None = None, skips: list[dict] | None = None, when: datetime = NOW) -> dict:
    return {"generated_utc": when.isoformat(timespec="seconds"), "generated_arizona": "test time", "legs": legs,
            "slate": [{"game": "AAA@BBB"}], "gated": gated or [], "skips": skips or []}


class OddsMathTests(unittest.TestCase):
    def test_american_and_decimal_odds(self):
        self.assertAlmostEqual(board.american_to_decimal(-150), 1 + 100 / 150)
        self.assertAlmostEqual(board.american_to_decimal(120), 2.2)
        self.assertEqual(board.decimal_to_american(3.0), 200)
        self.assertEqual(board.decimal_to_american(1.5), -200)


class SummaryTests(unittest.TestCase):
    def test_counts_overs_unders_and_touchdowns(self):
        legs = [leg("a", "over"), leg("b", "under"), leg("c", "under"), leg("d", "yes")]
        summary = board.summarise(legs)
        self.assertEqual((summary["legs"], summary["overs"], summary["unders"], summary["tds"]), (4, 1, 2, 1))
        self.assertEqual(summary["line"], "4 legs: 1 over, 2 unders, 1 TD")
        self.assertIsNone(summary["warning"])  # 50% unders

    def test_warning_only_above_three_quarters_unders(self):
        exactly = board.summarise([leg("a", "under"), leg("b", "under"), leg("c", "under"), leg("d", "over")])
        self.assertIsNone(exactly["warning"])  # 75% is not more than 75%
        over = board.summarise([leg(str(i), "under") for i in range(4)] + [leg("x", "over")])
        self.assertIn("80% of this board is unders", over["warning"])
        self.assertIn("calibration may be shrinking too hard", over["warning"])

    def test_empty_board(self):
        summary = board.summarise([])
        self.assertEqual(summary["line"], "0 legs: 0 overs, 0 unders, 0 TDs")
        self.assertIsNone(summary["warning"])


class TicketTests(unittest.TestCase):
    def test_a_ticket_needs_one_script_one_book_and_different_players(self):
        legs = [leg("a"), leg("b"), leg("c", group="AAA@BBB: AAA ahead"), leg("d", book="FD"),
                {**leg("a", market="rush attempts"), "player_id": "a"}]
        pairs = board.ticket_candidates(legs, 2)
        self.assertEqual({tuple(t["leg_ids"]) for t in pairs},
                         {("a|rush yards|over", "b|rush yards|over"), ("a|rush attempts|over", "b|rush yards|over")})
        for ticket in pairs:
            self.assertEqual(ticket["book"], "DK")
            self.assertEqual(len({l["player"] for l in ticket["legs"]}), 2)

    def test_ticket_price_and_probability_are_the_legs_multiplied(self):
        ticket = board.ticket_candidates([leg("a", price=-150, prob=0.7), leg("b", price=-120, prob=0.6)], 2)[0]
        decimal = (1 + 100 / 150) * (1 + 100 / 120)
        self.assertAlmostEqual(ticket["decimal"], decimal, places=3)
        self.assertEqual(ticket["price"], round((decimal - 1) * 100))
        self.assertAlmostEqual(ticket["model_prob"], 0.42, places=4)
        self.assertEqual(ticket["anchor"], "a|rush yards|over")  # the likeliest leg anchors the ticket

    def test_a_leg_is_only_used_at_a_book_where_it_keeps_a_five_percent_edge(self):
        thin = leg("b")
        thin["offers"][0]["edge"] = 0.03
        self.assertEqual(board.ticket_candidates([leg("a"), thin], 2), [])

    def test_rooting_conflict_carries_to_the_ticket(self):
        ticket = board.ticket_candidates([leg("a"), leg("b", conflict=True)], 2)[0]
        self.assertTrue(ticket["rooting_conflict"])

    def test_a_leg_the_two_models_disagree_on_is_never_in_a_ticket(self):
        legs = [leg("a"), leg("b"), leg("flip", agrees=False, flags=["calibration flips the literal lean"], tier="B")]
        for rule_on in (True, False):
            for size in (2, 3):
                tickets = board.ticket_candidates(legs, size, require_anchor=rule_on)
                self.assertFalse([t for t in tickets if "flip|rush yards|over" in t["leg_ids"]])
        self.assertEqual(len(board.ticket_candidates(legs, 2)), 1)
        self.assertEqual(board.ticket_candidates(legs, 3), [])


class AnchorTests(unittest.TestCase):
    def test_what_counts_as_an_anchor(self):
        self.assertTrue(board.is_anchor(leg("a", "over", market="rush attempts")))
        self.assertTrue(board.is_anchor(leg("a", "over", market="rush yards")))
        self.assertTrue(board.is_anchor(leg("a", "yes")))
        self.assertTrue(all(board.is_anchor(line) for line in board.moneyline_legs(slate())))
        for side, market in (("under", "rush attempts"), ("under", "rush yards"), ("over", "receptions"),
                             ("under", "receptions"), ("over", "receiving yards"), ("under", "receiving yards")):
            self.assertFalse(board.is_anchor(leg("a", side, market=market)), f"{side} {market}")

    def test_unders_alone_never_make_a_ticket(self):
        unders = [leg(name, "under", market="receiving yards") for name in ("a", "b", "c", "d")]
        self.assertEqual(board.ticket_candidates(unders, 2), [])
        self.assertEqual(board.ticket_candidates(unders, 3), [])
        overs = [leg(name, "over", market="receptions") for name in ("a", "b", "c")]  # not anchors either
        self.assertEqual(board.ticket_candidates(overs, 2), [])

    def test_the_rule_can_be_switched_off(self):
        unders = [leg(name, "under", market="receiving yards") for name in ("a", "b", "c")]
        self.assertEqual(len(board.ticket_candidates(unders, 2, require_anchor=False)), 3)
        self.assertEqual(len(board.ticket_candidates(unders, 3, require_anchor=False)), 1)
        # and the moneylines are left alone when it is off
        with_lines = board.ticket_candidates(unders, 2, board.moneyline_legs(slate()), require_anchor=False)
        self.assertEqual(len(with_lines), 3)
        self.assertFalse([t for t in with_lines if any(l["moneyline"] for l in t["legs"])])

    def test_every_ticket_holds_an_anchor_and_stays_on_one_script(self):
        legs = [leg("back", "over", market="rush attempts", prob=0.60), leg("scorer", "yes", prob=0.45, price=150),
                leg("wr1", "under", market="receiving yards", prob=0.66), leg("wr2", "under", market="receptions", prob=0.64),
                leg("other", "over", market="rush yards", group="AAA@BBB: AAA ahead", team="AAA"),
                leg("other_wr", "under", market="receptions", group="AAA@BBB: AAA ahead", team="AAA")]
        for size in (2, 3):
            tickets = board.ticket_candidates(legs, size)
            self.assertTrue(tickets)
            for ticket in tickets:
                members = [l for l in legs if l["id"] in ticket["leg_ids"]]
                self.assertEqual(len({l["correlation_group"] for l in members}), 1)
                self.assertTrue(any(board.is_anchor(l) for l in members))
                self.assertFalse(all(l["side"] == "under" for l in members))
                anchor = next(l for l in members if l["id"] == ticket["anchor"])
                self.assertTrue(board.is_anchor(anchor))
                self.assertEqual([l["id"] for l in ticket["legs"] if l["anchor"]], [ticket["anchor"]])
        pairs = {tuple(t["leg_ids"]) for t in board.ticket_candidates(legs, 2)}
        self.assertNotIn(("wr1|receiving yards|under", "wr2|receptions|under"), pairs)
        self.assertIn(("back|rush attempts|over", "wr1|receiving yards|under"), pairs)
        self.assertIn(("other_wr|receptions|under", "other|rush yards|over"), pairs)

    def test_the_anchor_is_the_likeliest_anchor_leg_not_the_likeliest_leg(self):
        legs = [leg("back", "over", market="rush attempts", prob=0.58), leg("scorer", "yes", prob=0.45, price=150),
                leg("wr", "under", market="receiving yards", prob=0.80)]
        ticket = board.ticket_candidates(legs, 3)[0]
        self.assertEqual(ticket["anchor"], "back|rush attempts|over")
        self.assertEqual(ticket["anchor_player"], "back")
        loose = board.ticket_candidates(legs, 3, require_anchor=False)[0]
        self.assertEqual(loose["anchor"], "wr|receiving yards|under")  # with the rule off it is simply the likeliest leg

    def test_a_moneyline_can_anchor_legs_on_its_own_script(self):
        unders = [leg("wr1", "under", market="receiving yards", prob=0.66, price=-115),
                  leg("wr2", "under", market="receptions", prob=0.62, price=-110)]
        lines = board.moneyline_legs(slate())
        pairs = board.ticket_candidates(unders, 2, lines)
        self.assertEqual({tuple(t["leg_ids"]) for t in pairs},
                         {("AAA@BBB|moneyline|BBB", "wr1|receiving yards|under"), ("AAA@BBB|moneyline|BBB", "wr2|receptions|under")})
        ticket = next(t for t in pairs if "wr1|receiving yards|under" in t["leg_ids"])
        self.assertEqual(ticket["anchor"], "AAA@BBB|moneyline|BBB")
        line = next(l for l in ticket["legs"] if l["moneyline"])
        self.assertEqual((line["player"], line["bet"], line["price"], line["edge"]), ("BBB", "moneyline", -200, 0.0))
        self.assertAlmostEqual(line["model_prob"], 0.64)  # the book's own probability: no edge is claimed
        self.assertAlmostEqual(ticket["model_prob"], round(0.64 * 0.66, 4), places=4)
        self.assertAlmostEqual(ticket["decimal"], 1.5 * (1 + 100 / 115), places=3)
        triple = board.ticket_candidates(unders, 3, lines)
        self.assertEqual(len(triple), 1)
        self.assertEqual(len(triple[0]["legs"]), 3)
        # AAA's moneyline is on the other script, where there are no legs to go with it
        self.assertFalse([t for t in pairs + triple if "AAA@BBB|moneyline|AAA" in t["leg_ids"]])

    def test_a_moneyline_is_priced_at_the_ticket_book_only(self):
        unders = [leg("wr1", "under", market="receiving yards", book="FD"), leg("wr2", "under", market="receptions", book="FD")]
        only_dk = board.moneyline_legs(slate({"BBB": {"DK": (-200, 0.64)}, "AAA": {"DK": (170, 0.36)}}))
        self.assertEqual(board.ticket_candidates(unders, 2, only_dk), [])
        both = board.ticket_candidates(unders, 2, board.moneyline_legs(slate()))
        self.assertEqual({t["book"] for t in both}, {"FD"})
        self.assertEqual({l["price"] for t in both for l in t["legs"] if l["moneyline"]}, {-210})

    def test_a_moneyline_against_a_rooted_team_is_a_rooting_conflict(self):
        lines = {l["team"]: l for l in board.moneyline_legs(slate(), {"AAA"})}
        self.assertTrue(lines["BBB"]["rooting_conflict"])  # BBB winning means AAA losing
        self.assertEqual(lines["BBB"]["tags"], ["rooting conflict"])
        self.assertFalse(lines["AAA"]["rooting_conflict"])
        self.assertEqual(lines["BBB"]["correlation_group"], "AAA@BBB: BBB ahead")
        ticket = board.ticket_candidates([leg("wr1", "under", market="receiving yards")], 2, list(lines.values()))[0]
        self.assertTrue(ticket["rooting_conflict"])
        self.assertFalse(any(l["rooting_conflict"] for l in board.moneyline_legs(slate())))

    def test_a_slate_without_moneylines_still_builds_from_prop_anchors(self):
        self.assertEqual(board.moneyline_legs([{"game": "AAA@BBB", "away": "AAA", "home": "BBB"}]), [])
        legs = [leg("back", "over", market="rush attempts"), leg("wr", "under", market="receiving yards")]
        self.assertEqual(len(board.ticket_candidates(legs, 2, [])), 1)


class BoostTests(unittest.TestCase):
    def setUp(self):
        # five legs in one script at one book: prices chosen so pairs and triples land in and out of range
        self.legs = [leg("a", price=-160, prob=0.72), leg("b", price=-150, prob=0.70), leg("c", price=-140, prob=0.66),
                     leg("d", price=-130, prob=0.64), leg("e", price=110, prob=0.55)]
        self.tickets = {size: board.ticket_candidates(self.legs, size) for size in (2, 3)}

    def test_each_boost_gets_a_ticket_in_its_price_range(self):
        fifty, hundred = board.allocate_boosts(self.tickets, RULES)
        self.assertEqual((fifty["name"], hundred["name"]), ("50% boost", "100% boost"))
        self.assertEqual(len(fifty["ticket"]["legs"]), 2)
        self.assertEqual(len(hundred["ticket"]["legs"]), 3)
        self.assertTrue(150 <= fifty["ticket"]["price"] <= 250)
        self.assertTrue(300 <= hundred["ticket"]["price"] <= 400)

    def test_the_same_anchor_leg_is_never_on_both_tickets(self):
        fifty, hundred = board.allocate_boosts(self.tickets, RULES)
        self.assertNotIn(hundred["ticket"]["anchor"], fifty["ticket"]["leg_ids"])
        self.assertNotIn(fifty["ticket"]["anchor"], hundred["ticket"]["leg_ids"])

    def test_safest_means_the_likeliest_ticket_left(self):
        fifty, hundred = board.allocate_boosts(self.tickets, RULES)
        blocked = set(hundred["ticket"]["leg_ids"])
        allowed = [t for t in self.tickets[2] if 150 <= t["price"] <= 250 and t["anchor"] not in blocked
                   and hundred["ticket"]["anchor"] not in t["leg_ids"]]
        self.assertEqual(fifty["ticket"]["model_prob"], max(t["model_prob"] for t in allowed))

    def test_boosted_price_applies_the_boost_to_the_winnings(self):
        fifty, _ = board.allocate_boosts(self.tickets, RULES)
        ticket = fifty["ticket"]
        self.assertEqual(ticket["boosted_price"], board.decimal_to_american(1 + 1.5 * (ticket["decimal"] - 1)))

    def test_no_ticket_in_range_is_explained(self):
        long_shots = [leg("a", price=200, prob=0.45), leg("b", price=220, prob=0.42), leg("c", price=250, prob=0.40)]
        tickets = {size: board.ticket_candidates(long_shots, size) for size in (2, 3)}
        fifty, hundred = board.allocate_boosts(tickets, RULES)
        self.assertIsNone(fifty["ticket"])
        self.assertIn("No 2-leg same-script ticket is priced +150 to +250", fifty["note"])
        self.assertIn("The nearest is", fifty["note"])
        self.assertIsNone(hundred["ticket"])

    def test_with_the_anchor_rule_a_board_of_unders_gets_no_boost_and_says_why(self):
        unders = [leg(name, "under", market="receiving yards", price=-110, prob=0.6) for name in ("a", "b", "c")]
        tickets = {size: board.ticket_candidates(unders, size) for size in (2, 3)}
        self.assertEqual(tickets, {2: [], 3: []})
        fifty, hundred = board.allocate_boosts(tickets, RULES, require_anchor=True)
        self.assertIsNone(fifty["ticket"])
        self.assertIn("No 2-leg anchored same-script ticket can be built", fifty["note"])
        self.assertIn("an RB rush attempts or rush yards over, an anytime TD or a team moneyline", fifty["note"])
        self.assertIsNone(hundred["ticket"])

    def test_one_moneyline_cannot_anchor_both_boosts(self):
        unders = [leg(name, "under", market="receiving yards", price=-110, prob=0.6) for name in ("a", "b", "c")]
        lines = board.moneyline_legs(slate())
        tickets = {size: board.ticket_candidates(unders, size, lines) for size in (2, 3)}
        self.assertTrue(tickets[2] and tickets[3])
        fifty, hundred = board.allocate_boosts(tickets, RULES, require_anchor=True)
        picked = [boost["ticket"] for boost in (fifty, hundred) if boost["ticket"]]
        self.assertEqual(len(picked), 1)  # every ticket here starts from BBB's moneyline, so only one boost is used
        self.assertTrue(any(l["moneyline"] and l["anchor"] for l in picked[0]["legs"]))

    def test_a_rooting_conflict_pick_comes_with_a_clean_alternative(self):
        legs = [leg("a", price=-160, prob=0.74, conflict=True), leg("b", price=-150, prob=0.72),
                leg("c", price=-140, prob=0.66), leg("d", price=-130, prob=0.64)]
        tickets = {2: board.ticket_candidates(legs, 2), 3: []}
        fifty = board.allocate_boosts(tickets, RULES)[0]
        self.assertTrue(fifty["ticket"]["rooting_conflict"])  # the pick is not removed
        self.assertIsNotNone(fifty["alternative"])
        self.assertFalse(fifty["alternative"]["rooting_conflict"])


class ChangeTests(unittest.TestCase):
    def test_first_run(self):
        changes = board.diff(None, make_board([leg("a")]), [], NOW)
        self.assertIn("First run", changes["headline"])

    def test_added_removed_and_moved(self):
        before = make_board([leg("stays"), leg("moves", line=44.5), leg("fades"), leg("gone")], when=NOW - timedelta(hours=3))
        after = make_board([leg("stays"), leg("moves", line=49.5), leg("new")])
        priced = [{"player_id": "fades", "market_key": "player_rush_yds", "line": 20.5, "edge": 0.021},
                  {"player_id": "moves", "market_key": "player_rush_yds", "line": 49.5, "edge": 0.10}]
        changes = board.diff(before, after, priced, NOW)
        self.assertEqual(len(changes["added"]), 1)
        self.assertIn("Added: new over 20.5 rush yards (DK -150)", changes["added"][0]["text"])
        self.assertEqual(len(changes["line_moved"]), 1)
        self.assertIn("44.5 to 49.5", changes["line_moved"][0]["text"])
        removed = {item["id"]: item["text"] for item in changes["removed"]}
        self.assertIn("the edge fell from 10.0% to 2.1%", removed["fades|rush yards|over"])
        self.assertIn("the books no longer offer it", removed["gone|rush yards|over"])
        self.assertIn("1 added, 2 removed, 1 line moved", changes["headline"])

    def test_a_line_move_that_kills_the_edge_says_so(self):
        before = make_board([leg("x", line=44.5)], when=NOW - timedelta(hours=3))
        changes = board.diff(before, make_board([]), [{"player_id": "x", "market_key": "player_rush_yds", "line": 49.5, "edge": 0.01}], NOW)
        self.assertIn("line moved 44.5 to 49.5", changes["removed"][0]["text"])

    def test_ruled_out_is_the_reason_when_the_injury_gate_drops_a_leg(self):
        before = make_board([leg("hurt")], when=NOW - timedelta(hours=3))
        after = make_board([], gated=[{"id": "hurt|rush yards|over", "player": "hurt", "team": "BBB", "market": "rush yards",
                                       "reason": "hurt is out"}])
        changes = board.diff(before, after, [], NOW)
        self.assertIn("ruled out: hurt is out", changes["removed"][0]["text"])
        self.assertTrue(any("Ruled out: hurt is out" in item["text"] for item in changes["status_changed"]))

    def test_a_dropped_quarterback_touchdown_says_why(self):
        old = {**leg("qb", "yes"), "position": "QB"}
        changes = board.diff(make_board([old], when=NOW - timedelta(hours=3)), make_board([]), [], NOW)
        self.assertIn("quarterback anytime TDs are no longer priced", changes["removed"][0]["text"])
        back = leg("rb", "yes")  # a running back's touchdown that the books took down is still described as that
        changes = board.diff(make_board([back], when=NOW - timedelta(hours=3)), make_board([]), [], NOW)
        self.assertIn("the books no longer offer it", changes["removed"][0]["text"])

    def test_status_and_tier_changes_are_reported(self):
        before = make_board([leg("q")], when=NOW - timedelta(hours=3))
        after = make_board([leg("q", status="questionable", tier="B", flags=["questionable (Ankle; practice: limited Fri)"])])
        changes = board.diff(before, after, [], NOW)
        self.assertIn("was active, now questionable", changes["status_changed"][0]["text"])
        self.assertIn("went from A to B", changes["tier_changed"][0]["text"])

    def test_a_price_move_needs_fifteen_cents(self):
        before = make_board([leg("a", price=-110), leg("b", price=-110)], when=NOW - timedelta(hours=3))
        after = make_board([leg("a", price=-120), leg("b", price=-130)])
        changes = board.diff(before, after, [], NOW)
        self.assertEqual([item["id"] for item in changes["price_moved"]], ["b|rush yards|over"])

    def test_legs_from_games_that_kicked_off_are_not_called_removed(self):
        old = leg("played")
        old["kickoff_utc"] = (NOW - timedelta(hours=1)).isoformat(timespec="minutes")
        changes = board.diff(make_board([old], when=NOW - timedelta(hours=5)), make_board([]), [], NOW)
        self.assertEqual(changes["removed"], [])
        self.assertEqual(changes["kicked_off"], 1)


class NextRunsTests(unittest.TestCase):
    """The Re-run tab lists the runs to expect. The jobs are set here, so config.json can change freely."""

    JOBS = {"outside_schedule": {"board_runs": ["06:37", "15:37", "21:07"], "board_hours": 60, "check_minutes": [0, 15, 30, 45]}}
    EVENTS = {"data": [
        {"id": "a", "away_team": "Green Bay Packers", "home_team": "Los Angeles Rams", "commence_time": "2026-11-26T01:00:00Z"},
        {"id": "b", "away_team": "Chicago Bears", "home_team": "Detroit Lions", "commence_time": "2026-11-26T18:00:00Z"},
        {"id": "c", "away_team": "Philadelphia Eagles", "home_team": "Dallas Cowboys", "commence_time": "2026-11-26T21:30:00Z"},
        {"id": "d", "away_team": "Kansas City Chiefs", "home_team": "Buffalo Bills", "commence_time": "2026-11-27T01:20:00Z"}]}

    def next_runs(self, now: datetime, events: dict | None) -> list[tuple[str, str]]:
        folder = Path(tempfile.mkdtemp())
        if events is not None:
            (folder / "events.json").write_text(json.dumps(events), encoding="utf-8")
        with mock.patch.object(board.fetch_odds, "EVENTS_FILE", folder / "events.json"), \
                mock.patch.object(common, "load_config", return_value=self.JOBS):
            return [(run["arizona"], run["why"]) for run in board.next_runs(now)]

    def test_daily_runs_and_the_runs_after_inactives_come_in_time_order(self):
        four_pm = datetime(2026, 11, 25, 23, 0, tzinfo=timezone.utc)  # Wednesday 4:00 PM in Arizona
        self.assertEqual(self.next_runs(four_pm, self.EVENTS), [
            ("Wed Nov 25 4:45 PM MST", "after inactives for GB@LA"),
            ("Wed Nov 25 9:07 PM MST", "daily run"),
            ("Thu Nov 26 6:37 AM MST", "daily run"),
            ("Thu Nov 26 9:45 AM MST", "after inactives for CHI@DET"),
            ("Thu Nov 26 1:15 PM MST", "after inactives for PHI@DAL")])

    def test_without_cached_games_only_the_daily_runs_are_known(self):
        four_pm = datetime(2026, 11, 25, 23, 0, tzinfo=timezone.utc)
        runs = self.next_runs(four_pm, None)
        self.assertEqual(runs[:2], [("Wed Nov 25 9:07 PM MST", "daily run"), ("Thu Nov 26 6:37 AM MST", "daily run")])
        self.assertEqual({why for _, why in runs}, {"daily run"})

    def test_games_with_one_kickoff_time_share_a_line(self):
        twins = {"data": [dict(self.EVENTS["data"][1]), {**self.EVENTS["data"][2], "commence_time": "2026-11-26T18:00:00Z"}]}
        runs = self.next_runs(datetime(2026, 11, 26, 14, 0, tzinfo=timezone.utc), twins)  # Thursday 7:00 AM in Arizona
        self.assertEqual(runs[0], ("Thu Nov 26 9:45 AM MST", "after inactives for CHI@DET, PHI@DAL"))

    def test_the_site_is_told_what_the_jobs_are(self):
        with mock.patch.object(common, "load_config", return_value=self.JOBS):
            self.assertEqual(board.schedule_summary(), {"board_runs": ["6:37 AM", "3:37 PM", "9:07 PM"], "board_hours": "60",
                                                        "check_minutes": [0, 15, 30, 45]})


class MissedRunNoteTests(unittest.TestCase):
    """The pregame check writes a notice when a run after inactives was given up on. Boards built after that
    game has kicked off repeat it in their notes for twelve hours."""

    NOTICE = {"kickoff_utc": "2026-11-26T01:00+00:00", "games": ["GB@LA"], "posted_utc": "2026-11-26T00:30:20+00:00",
              "posted_arizona": "Wed Nov 25 5:30 PM MST", "text": "The board was not rebuilt after inactives for GB@LA (made-up notice)."}

    def notes(self, now: datetime, content: str | None) -> list[str]:
        folder = Path(tempfile.mkdtemp())
        if content is not None:
            (folder / "notices.json").write_text(content, encoding="utf-8")
        with mock.patch.object(board.pregame, "NOTICES_FILE", folder / "notices.json"):
            return board.missed_run_notes(now)

    def test_a_notice_is_repeated_once_its_game_has_kicked_off(self):
        content = json.dumps({"notices": [self.NOTICE]})
        kick = datetime(2026, 11, 26, 1, 0, tzinfo=timezone.utc)
        # before kickoff a board being built is the rebuild itself, so it must not say there was none
        self.assertEqual(self.notes(kick - timedelta(minutes=20), content), [])
        self.assertEqual(self.notes(kick + timedelta(hours=3, minutes=7), content), [self.NOTICE["text"]])  # the 9:07 PM daily run
        self.assertEqual(self.notes(kick + timedelta(hours=12, minutes=1), content), [])

    def test_no_file_or_a_broken_one_adds_nothing(self):
        now = datetime(2026, 11, 26, 4, 7, tzinfo=timezone.utc)
        self.assertEqual(self.notes(now, None), [])
        self.assertEqual(self.notes(now, "not json"), [])
        self.assertEqual(self.notes(now, json.dumps({"notices": [{"kickoff_utc": "not a time", "text": "x"}, {"text": "no kickoff"}]})), [])
        self.assertEqual(self.notes(now, json.dumps({"notices": []})), [])


class TextTests(unittest.TestCase):
    def board_text(self, legs: list[dict], rule_on: bool) -> str:
        lines = board.moneyline_legs(slate()) if rule_on else []
        tickets = {size: board.ticket_candidates(legs, size, lines, rule_on) for size in (2, 3)}
        made = {**make_board(legs), "window_hours": 36.0, "calibration": "on", "summary": board.summarise(legs),
                "slate": [{"game": "AAA@BBB", "kickoff_arizona": "Sun Nov 29 11:00 AM MST", "home": "BBB", "home_spread": -3.0,
                           "total": 44.0, "blowout": False}],
                "sgps": {"two_leg": board.suggest(tickets[2]), "three_leg": board.suggest(tickets[3])},
                "boosts": board.allocate_boosts(tickets, RULES, rule_on), "anchor_rule": rule_on,
                "credits": {"this_run": 7, "used": 7, "remaining": 19993}, "notes": []}
        text = board.render_text(made, board.diff(None, made, [], NOW))
        text.encode("ascii")  # raises if anything would not paste cleanly into Notepad
        return text

    def test_board_text_is_plain_ascii_with_the_summary_on_top(self):
        legs = [leg(str(i), "under", market="receiving yards", conflict=(i == 0)) for i in range(5)]
        text = self.board_text(legs, rule_on=True)
        self.assertEqual(text.splitlines()[0], "NFL PROP BOARD")
        self.assertIn("5 legs: 0 overs, 5 unders, 0 TDs", text)
        self.assertIn("WARNING: 100% of this board is unders", text)
        self.assertIn("[rooting conflict]", text)

    def test_board_text_marks_the_anchor_and_says_which_rule_is_on(self):
        legs = [leg(str(i), "under", market="receiving yards") for i in range(4)]
        on = self.board_text(legs, rule_on=True)
        self.assertIn("Anchor rule on: every ticket starts from an RB rush attempts or rush yards over", on)
        self.assertIn("- BBB moneyline (-200)  [anchor: the book's own odds, no model edge]", on)
        off = self.board_text(legs, rule_on=False)
        self.assertIn("Anchor rule off", off)
        self.assertNotIn("moneyline", off)
        with_back = self.board_text(legs + [leg("back", "over", market="rush attempts", prob=0.9)], rule_on=True)
        self.assertIn("- back over 20.5 rush attempts (-150)  [anchor]", with_back)


if __name__ == "__main__":
    unittest.main()
