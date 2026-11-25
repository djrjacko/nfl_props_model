"""Unit tests for board.py: the under-lean summary, same-game tickets, the boost rules and the change log.

    python -m unittest discover -s tests -v

These use small made-up legs so every rule can be checked exactly.
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import board  # noqa: E402

NOW = datetime(2026, 11, 29, 16, 0, tzinfo=timezone.utc)
KICK = (NOW + timedelta(hours=3)).isoformat(timespec="minutes")
RULES = [
    {"name": "50% boost", "boost": 0.5, "max_stake": 50, "legs": 2, "min_price": 150, "max_price": 250, "pick": "safest"},
    {"name": "100% boost", "boost": 1.0, "max_stake": 25, "legs": 3, "min_price": 300, "max_price": 400, "pick": "best"},
]


def leg(player: str, side: str = "over", price: int = -150, prob: float = 0.70, group: str = "AAA@BBB: BBB ahead",
        team: str = "BBB", line: float | None = 20.5, market: str = "rush yards", book: str = "DK", tier: str = "A",
        flags: list[str] | None = None, conflict: bool = False, status: str = "active") -> dict:
    implied = round(prob - 0.10, 4)
    return {
        "id": f"{player}|{market}|{side}", "game": "AAA@BBB", "game_id": "2026_13_AAA_BBB", "kickoff_utc": KICK,
        "player": player, "player_id": player, "position": "RB", "team": team, "opponent": "AAA" if team == "BBB" else "BBB",
        "market": market, "market_key": market.replace(" ", "_"), "side": side, "line": None if side == "yes" else line,
        "book": book, "price": price, "books": {}, "projection": 25.0, "model_prob": prob, "implied_prob": implied,
        "edge": round(prob - implied, 4), "literal_prob": prob, "tier": tier, "flags": flags or [],
        "tags": ["rooting conflict"] if conflict else [], "rooting_conflict": conflict, "status": status,
        "correlation_group": group, "correlation_role": "rb_volume", "reasoning": "made up",
        "offers": [{"book": book, "line": None if side == "yes" else line, "price": price, "model_prob": prob,
                    "implied_prob": implied, "edge": round(prob - implied, 4)}],
    }


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
        priced = [{"player_id": "fades", "market_key": "rush_yards", "line": 20.5, "edge": 0.021},
                  {"player_id": "moves", "market_key": "rush_yards", "line": 49.5, "edge": 0.10}]
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
        changes = board.diff(before, make_board([]), [{"player_id": "x", "market_key": "rush_yards", "line": 49.5, "edge": 0.01}], NOW)
        self.assertIn("line moved 44.5 to 49.5", changes["removed"][0]["text"])

    def test_ruled_out_is_the_reason_when_the_injury_gate_drops_a_leg(self):
        before = make_board([leg("hurt")], when=NOW - timedelta(hours=3))
        after = make_board([], gated=[{"id": "hurt|rush yards|over", "player": "hurt", "team": "BBB", "market": "rush yards",
                                       "reason": "hurt is out"}])
        changes = board.diff(before, after, [], NOW)
        self.assertIn("ruled out: hurt is out", changes["removed"][0]["text"])
        self.assertTrue(any("Ruled out: hurt is out" in item["text"] for item in changes["status_changed"]))

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


class TextTests(unittest.TestCase):
    def test_board_text_is_plain_ascii_with_the_summary_on_top(self):
        legs = [leg(str(i), "under", conflict=(i == 0)) for i in range(5)]
        tickets = {size: board.ticket_candidates(legs, size) for size in (2, 3)}
        made = {**make_board(legs), "window_hours": 36.0, "calibration": "on", "summary": board.summarise(legs),
                "slate": [{"game": "AAA@BBB", "kickoff_arizona": "Sun Nov 29 11:00 AM MST", "home": "BBB", "home_spread": -3.0,
                           "total": 44.0, "blowout": False}],
                "sgps": {"two_leg": board.suggest(tickets[2]), "three_leg": board.suggest(tickets[3])},
                "boosts": board.allocate_boosts(tickets, RULES), "credits": {"this_run": 7, "used": 7, "remaining": 19993},
                "notes": []}
        text = board.render_text(made, board.diff(None, made, [], NOW))
        text.encode("ascii")  # raises if anything would not paste cleanly into Notepad
        top = text.splitlines()
        self.assertEqual(top[0], "NFL PROP BOARD")
        self.assertIn("5 legs: 0 overs, 5 unders, 0 TDs", text)
        self.assertIn("WARNING: 100% of this board is unders", text)
        self.assertIn("[rooting conflict]", text)


if __name__ == "__main__":
    unittest.main()
