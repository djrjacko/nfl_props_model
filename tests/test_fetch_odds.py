"""Unit tests for the part of fetch_odds.py that turns a game-lines response into rows. No network.

    python -m unittest discover -s tests -v

The response below has the shape The Odds API returned on 2026-11-25 for spreads, totals and h2h (the
moneyline). The numbers are that day's Green Bay at Los Angeles Rams lines.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fetch_odds  # noqa: E402

EVENT = {
    "id": "5b612899d6dcdf998107f2c6c6e49a58", "sport_key": "americanfootball_nfl", "commence_time": "2026-11-26T01:00:00Z",
    "home_team": "Los Angeles Rams", "away_team": "Green Bay Packers",
    "bookmakers": [
        {"key": "draftkings", "title": "DraftKings", "markets": [
            {"key": "spreads", "outcomes": [{"name": "Green Bay Packers", "price": -110, "point": 2.5},
                                            {"name": "Los Angeles Rams", "price": -109, "point": -2.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -121, "point": 43.5},
                                           {"name": "Under", "price": 101, "point": 43.5}]},
            {"key": "h2h", "outcomes": [{"name": "Green Bay Packers", "price": 121},
                                        {"name": "Los Angeles Rams", "price": -143}]}]},
        {"key": "fanduel", "title": "FanDuel", "markets": [
            {"key": "spreads", "outcomes": [{"name": "Green Bay Packers", "price": -108, "point": 2.5},
                                            {"name": "Los Angeles Rams", "price": -112, "point": -2.5}]}]},
    ],
}


class GameLineTests(unittest.TestCase):
    def test_the_moneyline_is_asked_for_and_has_its_columns(self):
        self.assertEqual(set(fetch_odds.GAME_MARKETS), {"spreads", "totals", "h2h"})
        self.assertTrue({"home_ml", "away_ml"} <= set(fetch_odds.GAME_LINE_COLUMNS))

    def test_one_row_per_book_with_spread_total_and_moneyline(self):
        rows = pd.DataFrame(fetch_odds._game_line_rows([EVENT]), columns=fetch_odds.GAME_LINE_COLUMNS)
        self.assertEqual(list(rows["book"]), ["draftkings", "fanduel"])
        dk = rows.iloc[0]
        self.assertEqual((dk["away"], dk["home"]), ("GB", "LA"))
        self.assertEqual((dk["home_spread"], dk["home_spread_price"], dk["away_spread_price"]), (-2.5, -109, -110))
        self.assertEqual((dk["total"], dk["over_price"], dk["under_price"]), (43.5, -121, 101))
        self.assertEqual((dk["home_ml"], dk["away_ml"]), (-143, 121))  # home is the Rams, away the Packers

    def test_a_book_without_a_market_leaves_it_empty(self):
        fd = pd.DataFrame(fetch_odds._game_line_rows([EVENT]), columns=fetch_odds.GAME_LINE_COLUMNS).iloc[1]
        self.assertEqual(fd["home_spread"], -2.5)
        self.assertTrue(pd.isna(fd["home_ml"]) and pd.isna(fd["away_ml"]) and pd.isna(fd["total"]))


if __name__ == "__main__":
    unittest.main()
