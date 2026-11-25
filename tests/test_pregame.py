"""Unit tests for the pre-kickoff check and for the description of the outside scheduler's jobs.

    python -m unittest discover -s tests -v

GitHub's own cron never started a run in this repository, so neither workflow has a schedule. An outside
scheduler starts the pregame check every 15 minutes and the board a few times a day. The decisions are
checked with Thanksgiving week 2026 (week 12), whose kickoffs come at many different times of day.

Every test that needs the scheduler's jobs sets them itself, so editing config.json cannot fail a test and
stop a board run. Only the README checks at the bottom read the real files, and they are skipped inside
GitHub Actions.
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
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import common  # noqa: E402
import pregame  # noqa: E402

UTC = timezone.utc
JOBS = {"board_runs": ["06:37", "15:37", "21:07"], "board_hours": 60, "check_minutes": [0, 15, 30, 45]}


def arizona(month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    """An Arizona wall-clock time in 2026 as a UTC datetime (Arizona is UTC-7 all year)."""
    return datetime(2026, month, day, hour, minute, second, tzinfo=UTC) + timedelta(hours=7)


def event(away: str, home: str, kick: datetime) -> dict:
    return {"id": f"{away}{home}", "away_team": away, "home_team": home, "commence_time": kick.strftime("%Y-%m-%dT%H:%M:%SZ")}


def with_jobs(**changes):
    """Run a test with these scheduler jobs instead of whatever config.json says."""
    return mock.patch.object(common, "load_config", return_value={"outside_schedule": {**JOBS, **changes}})


def ok(at: datetime) -> dict:
    """A board run that started then and finished successfully, as recent_board_runs reports it."""
    return {"at": at, "status": "completed", "conclusion": "success"}


def failed(at: datetime, how: str = "failure") -> dict:
    return {"at": at, "status": "completed", "conclusion": how}


def going(at: datetime, status: str = "in_progress") -> dict:
    return {"at": at, "status": status, "conclusion": None}


WEDNESDAY = event("Green Bay Packers", "Los Angeles Rams", arizona(11, 25, 18, 0))
THANKSGIVING_EARLY = event("Chicago Bears", "Detroit Lions", arizona(11, 26, 11, 0))
THANKSGIVING_AFTERNOON = event("Philadelphia Eagles", "Dallas Cowboys", arizona(11, 26, 14, 30))
THANKSGIVING_NIGHT = event("Kansas City Chiefs", "Buffalo Bills", arizona(11, 26, 18, 20))
BLACK_FRIDAY = event("Denver Broncos", "Pittsburgh Steelers", arizona(11, 27, 13, 0))
SUNDAY_EARLY = event("New Orleans Saints", "Cincinnati Bengals", arizona(11, 29, 11, 0))
SUNDAY_EARLY_TOO = event("Baltimore Ravens", "Houston Texans", arizona(11, 29, 11, 0))
SUNDAY_LATE_FIRST = event("New England Patriots", "Los Angeles Chargers", arizona(11, 29, 14, 5))
SUNDAY_LATE = event("Seattle Seahawks", "San Francisco 49ers", arizona(11, 29, 14, 25))
SUNDAY_NIGHT = event("Las Vegas Raiders", "Cleveland Browns", arizona(11, 29, 18, 20))
MONDAY_NIGHT = event("Carolina Panthers", "Tampa Bay Buccaneers", arizona(11, 30, 18, 15))
FIVE_PM = event("Dallas Cowboys", "New York Giants", arizona(12, 5, 17, 0))  # a made-up 5:00 PM kickoff


class WorkflowTests(unittest.TestCase):
    """The outside scheduler is the only scheduler. A schedule block put back by mistake would start every
    run twice if GitHub's cron ever woke up."""

    def test_neither_workflow_has_a_schedule_of_its_own(self):
        for path in (common.BOARD_WORKFLOW, common.PREGAME_WORKFLOW):
            with self.subTest(workflow=path.name):
                lines = [line.split("#")[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertIn("workflow_dispatch:", lines)
                self.assertNotIn("schedule:", lines)
                self.assertFalse([line for line in lines if line.startswith("- cron:")])

    def test_the_check_can_publish_its_notice_file_and_nothing_else(self):
        check = common.PREGAME_WORKFLOW.read_text(encoding="utf-8")
        self.assertRegex(check, r"contents: write")
        self.assertIn("git add docs/notices.json", check)
        self.assertEqual(check.count("git add"), 1)
        self.assertIn('if [ -z "$(git status --porcelain docs/notices.json)" ]; then', check)  # no change, no commit
        self.assertEqual(pregame.NOTICES_FILE, ROOT / "docs" / "notices.json")

    def test_the_board_takes_an_hours_input_and_the_check_takes_none(self):
        board = common.BOARD_WORKFLOW.read_text(encoding="utf-8")
        self.assertRegex(board, r"workflow_dispatch:\s*\n\s+inputs:\s*\n\s+hours:")
        self.assertIn("ASKED: ${{ inputs.hours }}", board)
        self.assertNotIn("inputs:", common.PREGAME_WORKFLOW.read_text(encoding="utf-8"))


class OutsideScheduleTests(unittest.TestCase):
    def test_the_jobs_are_read_from_the_config(self):
        with with_jobs():
            self.assertEqual(common.outside_schedule(), {"board_runs": [(6, 37), (15, 37), (21, 7)], "board_hours": "60",
                                                         "check_minutes": [0, 15, 30, 45]})
        with with_jobs(board_runs=["21:07", "7:05"], board_hours=36.0, check_minutes=[30, 0]):
            self.assertEqual(common.outside_schedule(), {"board_runs": [(7, 5), (21, 7)], "board_hours": "36", "check_minutes": [0, 30]})

    def test_a_config_file_that_sets_part_of_it_keeps_the_rest(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "config.json").write_text(json.dumps({"rooting_teams": ["GB"], "outside_schedule": {"board_hours": 48}}), encoding="utf-8")
        with mock.patch.object(common, "CONFIG_FILE", folder / "config.json"):
            jobs = common.outside_schedule()
            self.assertEqual(common.load_config()["rooting_teams"], ["GB"])
        self.assertEqual(jobs["board_hours"], "48")
        self.assertEqual(jobs["board_runs"], [(6, 37), (15, 37), (21, 7)])
        self.assertEqual(jobs["check_minutes"], [0, 15, 30, 45])
        with mock.patch.object(common, "CONFIG_FILE", folder / "no such file.json"):
            self.assertEqual(common.outside_schedule()["board_hours"], "60")  # no file at all: the defaults

    def test_a_value_that_cannot_be_read_falls_back_to_its_default(self):
        # a typo in config.json must not stop the check that starts the run before a kickoff
        usual = {"board_runs": [(6, 37), (15, 37), (21, 7)], "board_hours": "60", "check_minutes": [0, 15, 30, 45]}
        for bad in ({"board_runs": ["6:37 AM"]}, {"board_runs": "06:37"}, {"board_runs": ["25:00"]}, {"board_hours": "sixty"},
                    {"board_hours": 0}, {"board_hours": None}, {"check_minutes": [0, 75]}, {"check_minutes": "often"},
                    {"check_minutes": None}):
            with self.subTest(bad=bad), with_jobs(**bad):
                self.assertEqual(common.outside_schedule(), usual)
        with mock.patch.object(common, "load_config", return_value={}):  # no outside_schedule at all
            self.assertEqual(common.outside_schedule(), usual)
        with with_jobs(board_hours="sixty"):  # the good values next to a bad one are still used
            with mock.patch.object(pregame, "github", lambda method, path, token, body=None: (204, {})):
                pregame.start_board_run("someone/some-repo", "made-up-token")
            self.assertEqual(pregame.plan(arizona(11, 25, 16, 45, 20), [WEDNESDAY])[0]["state"], "due")

    def test_daily_runs_are_arizona_clock_times(self):
        with with_jobs():
            day = common.daily_board_runs(arizona(11, 25, 0), arizona(11, 26, 0))
            self.assertEqual([common.az(run) for run in day],
                             ["Wed Nov 25 6:37 AM MST", "Wed Nov 25 3:37 PM MST", "Wed Nov 25 9:07 PM MST"])
            self.assertEqual(day[2], datetime(2026, 11, 26, 4, 7, tzinfo=UTC))  # 9:07 PM in Arizona is the next day in UTC
            # the start is left out and the end is included
            at = arizona(11, 25, 15, 37)
            self.assertEqual(common.daily_board_runs(at, at + timedelta(hours=1)), [])
            self.assertEqual(common.daily_board_runs(at - timedelta(hours=1), at), [at])
            self.assertEqual(len(common.daily_board_runs(arizona(11, 23, 0), arizona(11, 30, 0))), 21)

    def test_the_first_check_at_or_after_a_moment(self):
        with with_jobs():
            self.assertEqual(common.first_check(arizona(11, 25, 16, 35)), arizona(11, 25, 16, 45))
            self.assertEqual(common.first_check(arizona(11, 25, 16, 45)), arizona(11, 25, 16, 45))
            self.assertEqual(common.first_check(arizona(11, 25, 16, 45, 1)), arizona(11, 25, 17, 0))
            self.assertEqual(common.first_check(arizona(11, 25, 23, 50)), arizona(11, 26, 0, 0))
        with with_jobs(check_minutes=[7, 37]):
            self.assertEqual(common.first_check(arizona(11, 25, 16, 35)), arizona(11, 25, 16, 37))
        with with_jobs(check_minutes=[]):
            self.assertIsNone(common.first_check(arizona(11, 25, 16, 35)))


class PlanTests(unittest.TestCase):
    def setUp(self):
        patcher = with_jobs()
        patcher.start()
        self.addCleanup(patcher.stop)

    def state(self, game: dict, now: datetime, recent: list[dict] | None = None) -> dict:
        return pregame.plan(now, [game], recent)[0]

    def test_a_six_oclock_kickoff_gets_its_run_at_the_445_check(self):
        # inactives are due at 4:30 PM and a run is useful from 4:35 PM
        early = self.state(WEDNESDAY, arizona(11, 25, 12))
        self.assertEqual((early["state"], early["by"], common.az(early["at"])), ("waiting", "check", "Wed Nov 25 4:45 PM MST"))
        self.assertEqual(self.state(WEDNESDAY, arizona(11, 25, 16, 30, 20))["state"], "waiting")  # the 4:30 check is too early
        self.assertEqual(self.state(WEDNESDAY, arizona(11, 25, 16, 45, 20))["state"], "due")  # the 4:45 check starts it
        started = [ok(arizona(11, 25, 16, 45, 25))]
        for minute in (0, 15, 30, 45):  # every later check sees that run succeeded and starts nothing
            self.assertEqual(self.state(WEDNESDAY, arizona(11, 25, 17, minute, 20), started)["state"], "done")

    def test_every_kickoff_time_gets_the_first_check_after_inactives_are_due(self):
        noon = arizona(11, 25, 12)
        for game, when in ((THANKSGIVING_EARLY, "Thu Nov 26 9:45 AM MST"), (THANKSGIVING_AFTERNOON, "Thu Nov 26 1:15 PM MST"),
                           (THANKSGIVING_NIGHT, "Thu Nov 26 5:00 PM MST"), (BLACK_FRIDAY, "Fri Nov 27 11:45 AM MST"),
                           (SUNDAY_EARLY, "Sun Nov 29 9:45 AM MST"), (SUNDAY_LATE_FIRST, "Sun Nov 29 12:45 PM MST"),
                           (SUNDAY_LATE, "Sun Nov 29 1:00 PM MST"), (SUNDAY_NIGHT, "Sun Nov 29 5:00 PM MST"),
                           (MONDAY_NIGHT, "Mon Nov 30 5:00 PM MST")):
            with self.subTest(game=pregame.label(game)):
                entry = self.state(game, noon)
                self.assertEqual((entry["state"], entry["by"]), ("waiting", "check"))
                self.assertEqual(common.az(entry["at"]), when)
                lead = pregame.kickoff(game) - entry["at"]
                self.assertTrue(timedelta(minutes=70) < lead <= timedelta(minutes=85))  # 5 to 20 minutes after inactives are due
                self.assertEqual(self.state(game, entry["at"] + timedelta(seconds=20))["state"], "due")
                self.assertEqual(self.state(game, entry["at"] - timedelta(minutes=15) + timedelta(seconds=20))["state"], "waiting")

    def test_due_only_inside_the_window(self):
        kick = pregame.kickoff(THANKSGIVING_EARLY)
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=86))["state"], "waiting")
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=85))["state"], "due")
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=25))["state"], "due")
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=24))["state"], "late")

    def test_no_second_run_once_one_has_succeeded_since_inactives(self):
        kick = pregame.kickoff(THANKSGIVING_EARLY)
        started = ok(kick - timedelta(minutes=75))
        for minutes_before in (60, 45, 30, 10):
            self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=minutes_before), [started])["state"], "done")
        # a run from before inactives were due does not count, for example that morning's daily run
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=75), [ok(arizona(11, 26, 6, 37))])["state"], "due")
        self.assertEqual(self.state(THANKSGIVING_EARLY, kick - timedelta(minutes=75), [ok(kick - timedelta(minutes=86))])["state"], "due")

    def test_a_check_whose_run_never_started_is_followed_by_the_next_check(self):
        # nothing started at 4:45 (GitHub refused the call, say): 5:00, 5:15 and 5:30 each try again, 5:45 is too late
        for minute, expected in ((0, "due"), (15, "due"), (30, "due"), (45, "late")):
            self.assertEqual(self.state(WEDNESDAY, arizona(11, 25, 17, minute, 20))["state"], expected)

    def test_nothing_waits_for_a_daily_run(self):
        # Monday night kicks off at 6:15 PM. The 3:37 PM daily run is before inactives, so the 5:00 PM check starts a run.
        self.assertEqual(self.state(MONDAY_NIGHT, arizona(11, 30, 17, 0, 20), [ok(arizona(11, 30, 15, 37))])["state"], "due")
        # Sunday's early games used to wait for a fixed 9:47 AM slot. Now the 9:45 AM check starts the run.
        self.assertEqual(self.state(SUNDAY_EARLY, arizona(11, 29, 9, 45, 20))["state"], "due")

    def test_a_daily_run_that_lands_after_inactives_counts_as_the_run(self):
        # 5:00 PM kickoff: a run is useful from 3:35 PM, and the daily run comes at 3:37 PM, before the 3:45 PM check
        entry = self.state(FIVE_PM, arizona(12, 5, 12))
        self.assertEqual((entry["state"], entry["by"], common.az(entry["at"])), ("waiting", "daily", "Sat Dec 05 3:37 PM MST".replace(" 05", " 5")))
        self.assertEqual(self.state(FIVE_PM, arizona(12, 5, 15, 45, 20), [ok(arizona(12, 5, 15, 37, 30))])["state"], "done")
        self.assertEqual(self.state(FIVE_PM, arizona(12, 5, 15, 45, 20))["state"], "due")  # the daily run never came
        self.assertIn("covered by the daily run at Sat Dec 5 3:37 PM MST", pregame.describe([entry])[0])

    def test_games_that_share_a_kickoff_time_need_one_run(self):
        games = [SUNDAY_EARLY, SUNDAY_EARLY_TOO]
        self.assertEqual([e["state"] for e in pregame.plan(arizona(11, 29, 9, 45, 20), games)], ["due", "due"])
        self.assertEqual([e["state"] for e in pregame.plan(arizona(11, 29, 10, 0, 20), games, [ok(arizona(11, 29, 9, 45, 30))])], ["done", "done"])

    def test_kickoffs_twenty_minutes_apart_get_a_run_each(self):
        games = [SUNDAY_LATE_FIRST, SUNDAY_LATE]  # 2:05 PM and 2:25 PM: inactives at 12:35 PM and 12:55 PM
        states = lambda now, runs: [e["state"] for e in pregame.plan(now, games, runs)]  # noqa: E731
        first, second = ok(arizona(11, 29, 12, 45, 30)), ok(arizona(11, 29, 13, 0, 30))
        self.assertEqual(states(arizona(11, 29, 12, 45, 20), []), ["due", "waiting"])
        self.assertEqual(states(arizona(11, 29, 13, 0, 20), [first]), ["done", "due"])  # the 12:45 run was before the 2:25 game's inactives
        self.assertEqual(states(arizona(11, 29, 13, 15, 20), [first, second]), ["done", "done"])

    def test_games_are_listed_in_kickoff_order_with_readable_lines(self):
        entries = pregame.plan(arizona(11, 25, 20), [BLACK_FRIDAY, THANKSGIVING_EARLY])
        self.assertEqual([e["game"] for e in entries], ["CHI@DET", "DEN@PIT"])
        lines = pregame.describe(entries)
        self.assertIn("board run at the Thu Nov 26 9:45 AM MST check, the first after inactives are due", lines[0])
        due = pregame.describe(pregame.plan(arizona(11, 26, 9, 45, 20), [THANKSGIVING_EARLY]))
        self.assertIn("DUE NOW: starting a board run", due[0])

    def test_no_checks_means_no_run_is_promised(self):
        with with_jobs(check_minutes=[], board_runs=[]):
            entry = self.state(WEDNESDAY, arizona(11, 25, 12))
            self.assertEqual((entry["state"], entry["at"], entry["by"]), ("waiting", None, None))
            self.assertIn("no check or daily run falls between inactives and kickoff", pregame.describe([entry])[0])


class RetryTests(unittest.TestCase):
    """Only a run that finished successfully settles a kickoff. A run that failed or was cancelled is retried
    by the next check inside the window, twice at most. Wednesday's game kicks off at 6:00 PM, so its window
    is 4:35 PM to 5:35 PM and the checks inside it are 4:45, 5:00, 5:15 and 5:30."""

    def setUp(self):
        patcher = with_jobs()
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, hour: int, minute: int, runs: list[dict]) -> dict:
        return pregame.plan(arizona(11, 25, hour, minute, 20), [WEDNESDAY], runs)[0]

    def test_a_failed_run_is_retried_by_the_next_check(self):
        first = failed(arizona(11, 25, 16, 45, 25))
        entry = self.check(17, 0, [first])
        self.assertEqual(entry["state"], "due")
        self.assertEqual(entry["why"], "retry 1 of 2: the run at 4:45 PM failed")
        self.assertIn("DUE NOW: starting a board run (retry 1 of 2: the run at 4:45 PM failed)", pregame.describe([entry])[0])
        retry = ok(arizona(11, 25, 17, 0, 25))  # the retry works: the kickoff is settled
        self.assertEqual(self.check(17, 15, [first, retry])["state"], "done")
        self.assertEqual(self.check(17, 30, [first, retry])["state"], "done")

    def test_a_cancelled_or_timed_out_run_is_retried_too(self):
        for how in ("cancelled", "timed_out", "startup_failure"):
            with self.subTest(how=how):
                self.assertEqual(self.check(17, 0, [failed(arizona(11, 25, 16, 45, 25), how)])["state"], "due")

    def test_two_retries_then_it_stops(self):
        runs = [failed(arizona(11, 25, 16, 45, 25))]
        self.assertEqual(self.check(17, 0, runs)["why"], "retry 1 of 2: the run at 4:45 PM failed")
        runs.append(failed(arizona(11, 25, 17, 0, 25), "cancelled"))
        self.assertEqual(self.check(17, 15, runs)["why"], "retry 2 of 2: the run at 5:00 PM failed")
        runs.append(failed(arizona(11, 25, 17, 15, 25)))
        entry = self.check(17, 30, runs)  # three runs have failed: the first and two retries
        self.assertEqual(entry["state"], "failed")
        self.assertEqual(entry["why"], "the run failed 3 times, at 4:45 PM, 5:00 PM and 5:15 PM, and after 2 retries it was not tried again")
        self.assertEqual(self.check(17, 45, runs)["state"], "failed")  # and it stays stopped
        self.assertIn("GAVE UP: the run failed 3 times", pregame.describe([entry])[0])

    def test_a_run_that_is_still_going_is_waited_for(self):
        for status in ("queued", "in_progress", "requested", "waiting", "pending"):
            with self.subTest(status=status):
                entry = self.check(17, 0, [going(arizona(11, 25, 16, 59, 50), status)])
                self.assertEqual(entry["state"], "running")  # no second run is started beside it
        # after a failure, a retry that is still going is waited for as well
        self.assertEqual(self.check(17, 15, [failed(arizona(11, 25, 16, 45, 25)), going(arizona(11, 25, 17, 14, 40))])["state"], "running")
        self.assertIn("waiting: a board run that started at Wed Nov 25 4:59 PM MST is still going",
                      pregame.describe([self.check(17, 0, [going(arizona(11, 25, 16, 59, 50))])])[0])

    def test_a_success_settles_it_whatever_else_happened(self):
        runs = [failed(arizona(11, 25, 16, 45, 25)), failed(arizona(11, 25, 17, 0, 25)), failed(arizona(11, 25, 17, 15, 25)),
                ok(arizona(11, 25, 17, 20, 0))]  # a tap on the site after the check gave up
        self.assertEqual(self.check(17, 30, runs)["state"], "done")

    def test_no_retry_once_the_window_has_closed(self):
        # the first run started at 5:30 PM and failed; at 5:45 PM it is less than 25 minutes to kickoff
        entry = self.check(17, 45, [failed(arizona(11, 25, 17, 30, 25))])
        self.assertEqual(entry["state"], "failed")
        self.assertEqual(entry["why"], "the run failed 1 time, at 5:30 PM, and there was no time left before kickoff to try again")
        self.assertEqual(self.check(17, 45, [])["state"], "late")  # nothing was ever started: that is "late", not "failed"

    def test_runs_from_before_the_window_are_neither_the_run_nor_a_failed_try(self):
        before = [failed(arizona(11, 25, 15, 37)), failed(arizona(11, 25, 16, 20)), failed(arizona(11, 25, 16, 34)), ok(arizona(11, 25, 16, 30))]
        entry = self.check(16, 45, before)
        self.assertEqual((entry["state"], entry["why"], entry["failed_at"]), ("due", "no board run since inactives were due", []))

    def test_overlapping_windows_count_their_own_failures(self):
        # 2:05 PM and 2:25 PM kickoffs: windows 12:40 to 1:40 and 1:00 to 2:00
        games = [SUNDAY_LATE_FIRST, SUNDAY_LATE]
        states = lambda hour, minute, runs: [e["state"] for e in pregame.plan(arizona(11, 29, hour, minute, 20), games, runs)]  # noqa: E731
        runs = [failed(arizona(11, 29, 12, 45, 25))]
        self.assertEqual(states(13, 0, runs), ["due", "due"])  # a retry for the first, the first run for the second
        runs.append(failed(arizona(11, 29, 13, 0, 25)))
        self.assertEqual(states(13, 15, runs), ["due", "due"])
        runs.append(failed(arizona(11, 29, 13, 15, 25)))
        self.assertEqual(states(13, 30, runs), ["failed", "due"])  # three failures for 2:05, only two inside the 2:25 window
        runs.append(ok(arizona(11, 29, 13, 30, 25)))
        self.assertEqual(states(13, 45, runs), ["done", "done"])  # that run is inside both windows


class NoticeTests(unittest.TestCase):
    """When a run after inactives is given up on, the check publishes a notice for the site and the next
    boards. It comes down if a run succeeds after all, and otherwise twelve hours after kickoff."""

    def setUp(self):
        patcher = with_jobs()
        patcher.start()
        self.addCleanup(patcher.stop)
        self.three = [failed(arizona(11, 25, 16, 45, 25)), failed(arizona(11, 25, 17, 0, 25)), failed(arizona(11, 25, 17, 15, 25))]

    def notices(self, now: datetime, games: list[dict], runs: list[dict], existing: list[dict] | None = None) -> list[dict]:
        return pregame.updated_notices(now, pregame.plan(now, games, runs), runs, existing or [])

    def test_giving_up_publishes_one_notice_that_says_what_happened(self):
        now = arizona(11, 25, 17, 30, 20)
        (notice,) = self.notices(now, [WEDNESDAY], self.three)
        self.assertEqual(notice["text"], "The board was not rebuilt after inactives for GB@LA (kickoff Wed Nov 25 6:00 PM MST): the run "
                                         "failed 3 times, at 4:45 PM, 5:00 PM and 5:15 PM, and after 2 retries it was not tried again.")
        self.assertEqual((notice["games"], notice["kickoff_utc"], notice["posted_arizona"]),
                         (["GB@LA"], "2026-11-26T01:00+00:00", "Wed Nov 25 5:30 PM MST"))
        json.dumps(notice)  # everything in it can be written to the file

    def test_nothing_is_published_while_it_is_still_trying_or_when_it_worked(self):
        self.assertEqual(self.notices(arizona(11, 25, 17, 0, 20), [WEDNESDAY], self.three[:1]), [])  # retry 1 is about to start
        self.assertEqual(self.notices(arizona(11, 25, 17, 15, 20), [WEDNESDAY], self.three[:2]), [])
        self.assertEqual(self.notices(arizona(11, 25, 17, 0, 20), [WEDNESDAY], [ok(arizona(11, 25, 16, 45, 25))]), [])
        self.assertEqual(self.notices(arizona(11, 25, 12), [WEDNESDAY], []), [])
        self.assertEqual(self.notices(arizona(11, 25, 17, 45, 20), [WEDNESDAY], []), [])  # "late" with no failed run is not this notice

    def test_running_out_of_time_after_a_failure_is_said_too(self):
        (notice,) = self.notices(arizona(11, 25, 17, 45, 20), [WEDNESDAY], [failed(arizona(11, 25, 17, 30, 25))])
        self.assertIn("the run failed 1 time, at 5:30 PM, and there was no time left before kickoff to try again.", notice["text"])

    def test_the_notice_is_stable_from_check_to_check(self):
        first = self.notices(arizona(11, 25, 17, 30, 20), [WEDNESDAY], self.three)
        again = self.notices(arizona(11, 25, 17, 45, 20), [WEDNESDAY], self.three, first)
        self.assertEqual(again, first)  # same words and the same "posted" time, so nothing new is committed

    def test_a_late_success_takes_the_notice_down(self):
        first = self.notices(arizona(11, 25, 17, 30, 20), [WEDNESDAY], self.three)
        tapped = self.three + [ok(arizona(11, 25, 17, 40, 0))]
        self.assertEqual(self.notices(arizona(11, 25, 17, 45, 20), [WEDNESDAY], tapped, first), [])
        # the same when the next check only comes after kickoff, with the game gone from the feed
        self.assertEqual(self.notices(arizona(11, 25, 18, 0, 20), [], tapped, first), [])
        # a run that is going when the check comes does not take it down yet
        trying = self.three + [going(arizona(11, 25, 17, 44, 50))]
        self.assertEqual(self.notices(arizona(11, 25, 17, 45, 20), [WEDNESDAY], trying, first), first)

    def test_after_kickoff_the_notice_stays_for_twelve_hours(self):
        first = self.notices(arizona(11, 25, 17, 30, 20), [WEDNESDAY], self.three)
        self.assertEqual(self.notices(arizona(11, 25, 18, 15, 20), [], self.three, first), first)
        self.assertEqual(self.notices(arizona(11, 25, 21, 7, 20), [], [ok(arizona(11, 25, 21, 7))], first), first)  # a later daily run is not a rebuild before kickoff
        self.assertEqual(self.notices(arizona(11, 26, 5, 59), [], [], first), first)
        self.assertEqual(self.notices(arizona(11, 26, 6, 1), [], [], first), [])

    def test_games_with_one_kickoff_time_share_a_notice(self):
        runs = [failed(arizona(11, 29, 9, 45, 25)), failed(arizona(11, 29, 10, 0, 25)), failed(arizona(11, 29, 10, 15, 25))]
        notices = self.notices(arizona(11, 29, 10, 30, 20), [SUNDAY_EARLY, SUNDAY_EARLY_TOO, SUNDAY_NIGHT], runs)
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0]["games"], ["NO@CIN", "BAL@HOU"])
        self.assertIn("for NO@CIN, BAL@HOU (kickoff Sun Nov 29 11:00 AM MST)", notices[0]["text"])

    def test_the_file_is_written_only_when_it_changes(self):
        folder = Path(tempfile.mkdtemp())
        with mock.patch.object(pregame, "NOTICES_FILE", folder / "docs" / "notices.json"):
            self.assertEqual(pregame.load_notices(), [])  # no file yet
            self.assertTrue(pregame.save_notices([]))  # the first save creates it
            self.assertFalse(pregame.save_notices([]))
            notice = self.notices(arizona(11, 25, 17, 30, 20), [WEDNESDAY], self.three)
            self.assertTrue(pregame.save_notices(notice))
            self.assertFalse(pregame.save_notices(notice))
            self.assertEqual(pregame.load_notices(), notice)
            raw = (folder / "docs" / "notices.json").read_bytes()
            self.assertNotIn(b"\r", raw)
            self.assertEqual(json.loads(raw)["notices"], notice)
            self.assertTrue(pregame.save_notices([]))
            (folder / "docs" / "notices.json").write_text("not json at all", encoding="utf-8")
            self.assertEqual(pregame.load_notices(), [])  # an unreadable file is the same as none


class CheckTests(unittest.TestCase):
    """The whole check, as pregame.yml runs it, with the outside world replaced: the games, GitHub's list of
    runs, the call that starts a run, the clock and the notice file."""

    def run_check(self, now: datetime, runs: list[dict], trigger: bool = True, start=None, games: list[dict] | None = None):
        started = []

        def record(repo, token):
            started.append((repo, token))
            if start is not None:
                start()

        folder = Path(tempfile.mkdtemp())
        patches = [with_jobs(), mock.patch.object(pregame, "utc_now", return_value=now),
                   mock.patch.object(pregame, "odds_api_key", return_value=("made-up-key", "test")),
                   mock.patch.object(pregame, "fetch_events", return_value=[WEDNESDAY] if games is None else games),
                   mock.patch.object(pregame, "repo_name", return_value="someone/some-repo"),
                   mock.patch.object(pregame, "recent_board_runs", return_value=runs),
                   mock.patch.object(pregame, "start_board_run", record),
                   mock.patch.object(pregame, "NOTICES_FILE", folder / "notices.json"),
                   mock.patch.dict(os.environ, {"GH_TOKEN": "made-up-token"}),
                   mock.patch.object(sys, "argv", ["pregame.py"] + (["--trigger"] if trigger else [])),
                   mock.patch("builtins.print")]
        mocks = [patch.start() for patch in patches]
        try:
            code = pregame.main()
            printed = "\n".join(" ".join(str(part) for part in call.args) for call in mocks[-1].call_args_list)
        finally:
            for patch in reversed(patches):
                patch.stop()
        notices = json.loads((folder / "notices.json").read_text(encoding="utf-8"))["notices"] if (folder / "notices.json").exists() else None
        return code, started, notices, printed

    def test_tonight_the_445_check_starts_one_run_and_the_later_checks_start_none(self):
        code, started, notices, _ = self.run_check(arizona(11, 25, 16, 30, 20), [])
        self.assertEqual((code, started), (0, []))  # 4:30 PM: too early
        code, started, notices, printed = self.run_check(arizona(11, 25, 16, 45, 20), [ok(arizona(11, 25, 15, 37))])
        self.assertEqual((code, started), (0, [("someone/some-repo", "made-up-token")]))  # 4:45 PM: started, once
        self.assertEqual(notices, [])
        self.assertIn("PREGAME CHECK | Wed Nov 25 4:45 PM MST", printed)
        self.assertIn("DUE NOW: starting a board run", printed)
        self.assertIn("Started the board run for GB@LA.", printed)
        for minute in (0, 15, 30, 45):
            code, started, _, printed = self.run_check(arizona(11, 25, 17, minute, 20), [ok(arizona(11, 25, 15, 37)), ok(arizona(11, 25, 16, 45, 25))])
            self.assertEqual((code, started), (0, []))
            self.assertIn("done: a board run that started at Wed Nov 25 4:45 PM MST finished successfully", printed)
            self.assertIn("Nothing is due.", printed)

    def test_three_failures_two_retries_then_a_notice(self):
        fails = [failed(arizona(11, 25, 16, 45, 25)), failed(arizona(11, 25, 17, 0, 25)), failed(arizona(11, 25, 17, 15, 25))]
        _, started, notices, _ = self.run_check(arizona(11, 25, 17, 0, 20), fails[:1])
        self.assertEqual((len(started), notices), (1, []))  # retry 1
        _, started, notices, _ = self.run_check(arizona(11, 25, 17, 15, 20), fails[:2])
        self.assertEqual((len(started), notices), (1, []))  # retry 2
        code, started, notices, printed = self.run_check(arizona(11, 25, 17, 30, 20), fails)
        self.assertEqual((code, started), (0, []))  # stopped
        self.assertEqual(len(notices), 1)
        self.assertIn("GAVE UP: the run failed 3 times", printed)
        self.assertIn("The notice file changed.", printed)
        self.assertIn("failed 3 times, at 4:45 PM, 5:00 PM and 5:15 PM, and after 2 retries it was not tried again", notices[0]["text"])

    def test_show_only_mode_starts_nothing_and_writes_nothing(self):
        fails = [failed(arizona(11, 25, 16, 45, 25)), failed(arizona(11, 25, 17, 0, 25)), failed(arizona(11, 25, 17, 15, 25))]
        code, started, notices, _ = self.run_check(arizona(11, 25, 17, 30, 20), fails, trigger=False)
        self.assertEqual((code, started, notices), (0, [], None))
        code, started, notices, _ = self.run_check(arizona(11, 25, 16, 45, 20), [], trigger=False)
        self.assertEqual((code, started, notices), (0, [], None))

    def test_a_start_that_github_refuses_fails_the_check_so_the_next_one_tries_again(self):
        def refuse():
            raise RuntimeError("could not start the board run (HTTP 500): made up")

        code, started, _, _ = self.run_check(arizona(11, 25, 16, 45, 20), [], start=refuse)
        self.assertEqual((code, len(started)), (1, 1))
        code, started, _, _ = self.run_check(arizona(11, 25, 17, 0, 20), [])  # nothing started at 4:45, so 5:00 is due
        self.assertEqual((code, len(started)), (0, 1))


class RunListTests(unittest.TestCase):
    def test_runs_are_read_with_how_they_ended(self):
        answer = {"workflow_runs": [
            {"created_at": "2026-11-25T23:45:25Z", "status": "completed", "conclusion": "failure"},
            {"created_at": "2026-11-26T00:00:25Z", "status": "in_progress", "conclusion": None},
            {"created_at": "2026-11-25T22:37:02Z", "status": "completed", "conclusion": "success"}]}
        asked = []

        def fake(method, path, token, body=None):
            asked.append((method, path))
            return 200, answer

        with mock.patch.object(pregame, "github", fake):
            runs = pregame.recent_board_runs("someone/some-repo", "made-up-token", arizona(11, 25, 15, 0))
        self.assertEqual(runs, [failed(arizona(11, 25, 16, 45, 25)), going(arizona(11, 25, 17, 0, 25)), ok(arizona(11, 25, 15, 37, 2))])
        self.assertEqual(asked[0][0], "GET")
        self.assertIn("/repos/someone/some-repo/actions/workflows/run.yml/runs?", asked[0][1])
        self.assertIn("created=%3E%3D2026-11-25T22%3A00%3A00Z", asked[0][1])
        with mock.patch.object(pregame, "github", lambda *a, **k: (403, {"message": "made up"})):
            with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
                pregame.recent_board_runs("someone/some-repo", "made-up-token", arizona(11, 25, 15, 0))


class StartRunTests(unittest.TestCase):
    def test_the_run_it_starts_asks_for_the_same_hours_as_the_daily_runs(self):
        calls = []

        def fake(method, path, token, body=None):
            calls.append((method, path, token, body))
            return 200, {"workflow_run_id": 1}

        with with_jobs(), mock.patch.object(pregame, "github", fake):
            pregame.start_board_run("someone/some-repo", "made-up-token")
        self.assertEqual(calls, [("POST", "/repos/someone/some-repo/actions/workflows/run.yml/dispatches", "made-up-token",
                                  {"ref": "main", "inputs": {"hours": "60"}})])
        with with_jobs(board_hours=36), mock.patch.object(pregame, "github", fake):
            pregame.start_board_run("someone/some-repo", "made-up-token")
        self.assertEqual(calls[-1][3], {"ref": "main", "inputs": {"hours": "36"}})

    def test_both_of_githubs_started_answers_are_accepted_and_anything_else_is_an_error(self):
        for status in (200, 204):
            with with_jobs(), mock.patch.object(pregame, "github", lambda *a, **k: (status, {})):  # noqa: B023
                pregame.start_board_run("someone/some-repo", "made-up-token")
        with with_jobs(), mock.patch.object(pregame, "github", lambda *a, **k: (422, {"message": "Unexpected inputs provided"})):
            with self.assertRaisesRegex(RuntimeError, "HTTP 422"):
                pregame.start_board_run("someone/some-repo", "made-up-token")


@unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "a README or config edit must never stop a board run")
class ReadmeTests(unittest.TestCase):
    """The README describes the outside scheduler's jobs and config.json describes them to the code. The two
    have to say the same thing.

    These run on a PC, where the files are edited. They are skipped inside GitHub Actions, because the board
    workflow runs the tests first and a wording change is no reason to miss a board.
    """

    @classmethod
    def setUpClass(cls):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.section = readme.split("\n## Schedule\n")[1].split("\n## ")[0]
        cls.jobs = common.outside_schedule()

    def test_the_readme_lists_the_jobs_that_config_json_describes(self):
        clock = [datetime(2000, 1, 1, hour, minute).strftime("%I:%M %p").lstrip("0") for hour, minute in self.jobs["board_runs"]]
        self.assertIn(", ".join(clock[:-1]) + " and " + clock[-1], self.section)
        minutes = [f":{minute:02d}" for minute in self.jobs["check_minutes"]]
        self.assertIn(", ".join(minutes[:-1]) + " and " + minutes[-1], self.section)

    def test_the_readme_call_is_the_real_one(self):
        bodies = re.findall(r'^\{"ref".*\}$', self.section, flags=re.MULTILINE)
        self.assertIn({"ref": "main", "inputs": {"hours": self.jobs["board_hours"]}}, [json.loads(body) for body in bodies])
        self.assertIn({"ref": "main"}, [json.loads(body) for body in bodies])
        repo = pregame.repo_name()
        if repo:  # the address in the README is this repository's own
            self.assertIn(f"POST https://api.github.com/repos/{repo}/actions/workflows/run.yml/dispatches", self.section)
            self.assertIn(f"POST https://api.github.com/repos/{repo}/actions/workflows/pregame.yml/dispatches", self.section)
        self.assertNotIn("OWNER/REPO", self.section)


if __name__ == "__main__":
    unittest.main()
