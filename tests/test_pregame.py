"""Unit tests for the pre-kickoff scheduler: reading cron lines, and deciding which games need their own run.

    python -m unittest discover -s tests -v

The decisions are checked against the real workflow files in .github/workflows, using Thanksgiving week
2026 (week 12), whose kickoffs do not fit the usual Thursday / Sunday / Monday pattern.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import common  # noqa: E402
import pregame  # noqa: E402

UTC = timezone.utc


def arizona(month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """An Arizona wall-clock time in 2026 as a UTC datetime (Arizona is UTC-7 all year)."""
    return datetime(2026, month, day, hour, minute, tzinfo=UTC) + timedelta(hours=7)


def event(away: str, home: str, kick: datetime) -> dict:
    return {"id": f"{away}{home}", "away_team": away, "home_team": home, "commence_time": kick.strftime("%Y-%m-%dT%H:%M:%SZ")}


WEDNESDAY = event("Green Bay Packers", "Los Angeles Rams", arizona(11, 25, 18, 0))
THANKSGIVING_EARLY = event("Chicago Bears", "Detroit Lions", arizona(11, 26, 11, 0))
THANKSGIVING_AFTERNOON = event("Philadelphia Eagles", "Dallas Cowboys", arizona(11, 26, 14, 30))
THANKSGIVING_NIGHT = event("Kansas City Chiefs", "Buffalo Bills", arizona(11, 26, 18, 20))
BLACK_FRIDAY = event("Denver Broncos", "Pittsburgh Steelers", arizona(11, 27, 13, 0))
SUNDAY_EARLY = event("New Orleans Saints", "Cincinnati Bengals", arizona(11, 29, 11, 0))
SUNDAY_LATE = event("Seattle Seahawks", "San Francisco 49ers", arizona(11, 29, 14, 25))
SUNDAY_NIGHT = event("Las Vegas Raiders", "Cleveland Browns", arizona(11, 29, 18, 20))
MONDAY_NIGHT = event("Carolina Panthers", "Tampa Bay Buccaneers", arizona(11, 30, 18, 15))
NEXT_THURSDAY = event("Kansas City Chiefs", "Los Angeles Rams", arizona(12, 3, 18, 15))


class CronTests(unittest.TestCase):
    def workflow(self, *crons: str) -> Path:
        folder = Path(tempfile.mkdtemp())
        path = folder / "wf.yml"
        path.write_text("on:\n  schedule:\n" + "".join(f'    - cron: "{c}"\n' for c in crons), encoding="utf-8")
        return path

    def test_weekday_cron(self):
        # 37 22 * * 5 is Friday 22:37 UTC, which is Friday 3:37 PM in Arizona
        runs = common.scheduled_runs(datetime(2026, 11, 23, tzinfo=UTC), datetime(2026, 11, 30, tzinfo=UTC), self.workflow("37 22 * * 5"))
        self.assertEqual(runs, [datetime(2026, 11, 27, 22, 37, tzinfo=UTC)])
        self.assertEqual(common.az(runs[0]), "Fri Nov 27 3:37 PM MST")

    def test_sunday_is_zero_and_the_utc_day_can_differ_from_the_arizona_day(self):
        # 7 4 * * 0 is Sunday 04:07 UTC, which is Saturday 9:07 PM in Arizona
        runs = common.scheduled_runs(datetime(2026, 11, 23, tzinfo=UTC), datetime(2026, 11, 30, tzinfo=UTC), self.workflow("7 4 * * 0"))
        self.assertEqual([common.az(r) for r in runs], ["Sat Nov 28 9:07 PM MST"])

    def test_dated_cron_fires_once_a_year(self):
        runs = common.scheduled_runs(datetime(2026, 11, 1, tzinfo=UTC), datetime(2027, 12, 1, tzinfo=UTC), self.workflow("37 23 25 11 *"))
        self.assertEqual(runs, [datetime(2026, 11, 25, 23, 37, tzinfo=UTC), datetime(2027, 11, 25, 23, 37, tzinfo=UTC)])

    def test_lists_and_ranges(self):
        runs = common.scheduled_runs(datetime(2026, 11, 25, 10, 59, tzinfo=UTC), datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
                                     self.workflow("7,22,37,52 11-23,0-2 * 1,2,9-12 *"))
        self.assertEqual([r.strftime("%H:%M") for r in runs], ["11:07", "11:22", "11:37", "11:52"])

    def test_month_filter_keeps_the_check_quiet_in_the_off_season(self):
        runs = common.scheduled_runs(datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 6, 8, tzinfo=UTC),
                                     self.workflow("7,22,37,52 11-23,0-2 * 1,2,9-12 *"))
        self.assertEqual(runs, [])

    def test_start_is_exclusive_and_end_inclusive(self):
        at = datetime(2026, 11, 27, 22, 37, tzinfo=UTC)
        wf = self.workflow("37 22 * * 5")
        self.assertEqual(common.scheduled_runs(at, at + timedelta(hours=1), wf), [])
        self.assertEqual(common.scheduled_runs(at - timedelta(hours=1), at, wf), [at])

    def test_the_real_board_workflow_has_the_thirteen_weekly_runs(self):
        runs = common.scheduled_runs(datetime(2026, 12, 7, tzinfo=UTC), datetime(2026, 12, 14, tzinfo=UTC))
        self.assertEqual(len(runs), 13)


class PlanTests(unittest.TestCase):
    def state(self, game: dict, now: datetime, recent: list[datetime] | None = None) -> dict:
        return pregame.plan(now, [game], recent)[0]

    def test_usual_slots_are_covered_by_the_fixed_schedule(self):
        early = arizona(11, 25, 12)
        for game, when in ((SUNDAY_EARLY, "Sun Nov 29 9:47 AM MST"), (SUNDAY_LATE, "Sun Nov 29 1:07 PM MST"),
                           (SUNDAY_NIGHT, "Sun Nov 29 5:07 PM MST"), (MONDAY_NIGHT, "Mon Nov 30 4:52 PM MST"),
                           (NEXT_THURSDAY, "Thu Dec 03 4:52 PM MST".replace(" 03", " 3"))):
            with self.subTest(game=pregame.label(game)):
                entry = self.state(game, early)
                self.assertEqual(entry["state"], "covered")
                self.assertEqual(common.az(entry["at"]), when)

    def test_the_wednesday_game_gets_a_run_at_437_either_way(self):
        # While the one-off cron line is in run.yml the fixed schedule covers it; once that line is deleted
        # the automatic rule lands on the same time. This test must pass in both cases.
        entry = self.state(WEDNESDAY, arizona(11, 25, 12))
        self.assertIn(entry["state"], ("covered", "waiting"))
        self.assertEqual(common.az(entry["at"]), "Wed Nov 25 4:37 PM MST")

    def test_holiday_games_get_their_own_run_after_inactives_are_due(self):
        early = arizona(11, 25, 20)
        for game, when in ((THANKSGIVING_EARLY, "Thu Nov 26 9:37 AM MST"), (THANKSGIVING_AFTERNOON, "Thu Nov 26 1:07 PM MST"),
                           (THANKSGIVING_NIGHT, "Thu Nov 26 5:07 PM MST"), (BLACK_FRIDAY, "Fri Nov 27 11:37 AM MST")):
            with self.subTest(game=pregame.label(game)):
                entry = self.state(game, early)
                self.assertEqual(entry["state"], "waiting")
                self.assertEqual(common.az(entry["at"]), when)
                lead = pregame.kickoff(game) - entry["at"]
                self.assertTrue(timedelta(minutes=70) <= lead <= timedelta(minutes=85))  # 5 to 20 minutes after inactives

    def test_due_only_inside_the_window(self):
        kick = pregame.kickoff(THANKSGIVING_EARLY)
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=86))["state"], "waiting")
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=85))["state"], "due")
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=25))["state"], "due")
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=24))["state"], "late")

    def test_one_run_per_game(self):
        kick = pregame.kickoff(THANKSGIVING_EARLY)
        started = kick - timedelta(minutes=83)
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=68), [started])["state"], "done")
        # a run from before inactives were due does not count
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=68), [kick - timedelta(minutes=100)])["state"], "due")

    def test_a_fixed_run_that_never_started_is_replaced(self):
        kick = pregame.kickoff(SUNDAY_EARLY)  # fixed run at 9:47 AM, kickoff 11:00 AM
        fixed = arizona(11, 29, 9, 47)
        self.assertEqual(self.state(SUNDAY_EARLY, fixed + timedelta(minutes=15))["state"], "covered")
        self.assertEqual(self.state(SUNDAY_EARLY, fixed + timedelta(minutes=21))["state"], "due")
        self.assertEqual(self.state(SUNDAY_EARLY, fixed + timedelta(minutes=21), [fixed + timedelta(minutes=9)])["state"], "done")
        self.assertEqual(self.state(SUNDAY_EARLY, kick - timedelta(minutes=10))["state"], "covered")  # too late to replace

    def test_games_are_listed_in_kickoff_order_with_readable_lines(self):
        entries = pregame.plan(arizona(11, 25, 20), [BLACK_FRIDAY, THANKSGIVING_EARLY])
        self.assertEqual([e["game"] for e in entries], ["CHI@DET", "DEN@PIT"])
        lines = pregame.describe(entries)
        self.assertIn("own run about Thu Nov 26 9:37 AM MST", lines[0])


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "a README edit must never stop a board run")
class OutsideSchedulerTests(unittest.TestCase):
    """The README tells an outside scheduler when to start run.yml and with which `hours`. That table has to
    say what the workflow file says.

    These run on a PC, where the files are edited. They are skipped inside GitHub Actions, because the board
    workflow runs the tests first and a wording change in the README is no reason to miss a board.
    """

    @classmethod
    def setUpClass(cls):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.section = readme.split("### Starting runs from an outside scheduler")[1].split("\n## ")[0]
        cls.workflow = common.BOARD_WORKFLOW.read_text(encoding="utf-8")

    def test_the_readme_times_are_the_workflows_times_in_arizona(self):
        listed = set()
        for line in self.section.splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) == 3 and cells[0] in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"):
                listed.update((cells[0], moment.strip(), cells[2]) for moment in cells[1].split(","))
        # the workflow gives one run 60 hours: the cron line it names in its "Decide how far ahead to look" step
        long_minute, long_hour, _, _, long_weekday = re.search(r'"\$SCHEDULE" = "([^"]+)"', self.workflow).group(1).split()
        # a week with no dated line in it, from midnight Monday in Arizona
        start = arizona(12, 7, 0)
        expected = set()
        for run in common.scheduled_runs(start - timedelta(seconds=1), start + timedelta(days=7) - timedelta(seconds=1)):
            is_long = (run.minute, run.hour, (run.weekday() + 1) % 7) == (int(long_minute), int(long_hour), int(long_weekday))
            local = run.astimezone(common.ARIZONA)
            expected.add((local.strftime("%a"), local.strftime("%I:%M %p").lstrip("0"), "60" if is_long else "36"))
        self.assertEqual(len(expected), 13)
        self.assertEqual(listed, expected)

    def test_the_readme_call_names_both_workflows_and_a_valid_body(self):
        self.assertIn("/actions/workflows/run.yml/dispatches", self.section)
        self.assertIn("pregame.yml/dispatches", self.section)
        self.assertTrue(common.BOARD_WORKFLOW.exists() and common.PREGAME_WORKFLOW.exists())
        body = re.search(r'^\{"ref".*\}$', self.section, flags=re.MULTILINE).group(0)
        self.assertEqual(json.loads(body), {"ref": "main", "inputs": {"hours": pregame.BOARD_HOURS}})
        # run.yml declares the input the body sends; pregame.yml declares none, so its body must not send one
        self.assertRegex(self.workflow, r"workflow_dispatch:\s*\n\s+inputs:\s*\n\s+hours:")
        self.assertNotIn("inputs:", common.PREGAME_WORKFLOW.read_text(encoding="utf-8"))
        self.assertIn('`{"ref":"main"}`', self.section)

    def test_the_readme_pregame_times_match_its_cron(self):
        ticks = common.scheduled_runs(arizona(12, 7, 0) - timedelta(seconds=1), arizona(12, 8, 0) - timedelta(seconds=1),
                                      common.PREGAME_WORKFLOW)
        local = [tick.astimezone(common.ARIZONA) for tick in ticks]
        self.assertEqual({moment.minute for moment in local}, {7, 22, 37, 52})
        first, last = min(local), max(local)
        self.assertEqual((first.strftime("%I:%M %p").lstrip("0"), last.strftime("%I:%M %p").lstrip("0")), ("4:07 AM", "7:52 PM"))
        self.assertIn("at :07, :22, :37 and :52, from 4:07 AM to 7:52 PM, September to", self.section)
        self.assertEqual(common.scheduled_runs(datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 6, 8, tzinfo=UTC),
                                               common.PREGAME_WORKFLOW), [])  # nothing in the off-season


if __name__ == "__main__":
    unittest.main()
