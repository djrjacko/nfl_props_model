"""Unit tests for the schedule helpers in fetch_nflverse.py. No network.

    python -m unittest discover -s tests -v

nflverse stored the schedule's game date as text until 2026-11-25 and as a date from then on. The first
board run after that change stopped at the download step, so both forms are tested here.
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fetch_nflverse as nfl  # noqa: E402

ROWS = [  # season, week, game date, kickoff (Eastern), home score
    (2025, 18, "2026-01-04", "13:00", 24.0),
    (2026, 11, "2026-11-22", "13:00", 27.0),
    (2026, 12, "2026-11-25", "20:00", None),
    (2026, 12, "2026-11-26", "16:30", None),
    (2026, 18, "2027-01-10", "13:00", None),
]


def schedule(as_dates: bool) -> pd.DataFrame:
    frame = pd.DataFrame(ROWS, columns=["season", "week", "gameday", "gametime", "home_score"])
    if as_dates:
        frame["gameday"] = [date.fromisoformat(day) for day in frame["gameday"]]
    return frame


class GamedayTests(unittest.TestCase):
    def test_the_game_date_reads_the_same_as_text_or_as_a_date(self):
        text, dates = schedule(False), schedule(True)
        self.assertIsInstance(dates["gameday"].iloc[0], date)
        self.assertEqual(list(nfl.gameday_text(text)), [row[2] for row in ROWS])
        self.assertEqual(list(nfl.gameday_text(dates)), [row[2] for row in ROWS])

    def test_kickoff_is_the_same_either_way_and_in_arizona_time(self):
        from_text, from_dates = nfl.kickoff_arizona(schedule(False)), nfl.kickoff_arizona(schedule(True))
        self.assertTrue((from_text == from_dates).all())
        wednesday = from_dates.iloc[2]  # 8:00 PM Eastern standard time is 6:00 PM in Arizona
        self.assertEqual((wednesday.year, wednesday.month, wednesday.day, wednesday.hour, wednesday.minute), (2026, 11, 25, 18, 0))
        self.assertEqual(wednesday.utcoffset().total_seconds(), -7 * 3600)

    def test_a_missing_date_or_time_gives_no_kickoff_instead_of_an_error(self):
        frame = schedule(True)
        frame.loc[1, "gameday"] = None
        frame.loc[3, "gametime"] = None
        kicks = nfl.kickoff_arizona(frame)
        self.assertTrue(pd.isna(kicks.iloc[1]) and pd.isna(kicks.iloc[3]))
        self.assertFalse(pd.isna(kicks.iloc[2]))

    def test_the_current_season_is_found_either_way(self):
        for as_dates in (False, True):
            with self.subTest(as_dates=as_dates):
                current, history, last_week = nfl.resolve_seasons(schedule(as_dates), None, date(2026, 11, 25))
                self.assertEqual((current, history, last_week), (2026, [2026], 11))


if __name__ == "__main__":
    unittest.main()
