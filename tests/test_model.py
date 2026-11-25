"""Unit test for the model on one cached game: Green Bay at Los Angeles Rams, week 12 of 2026.

    python -m unittest discover -s tests -v

Everything runs from tests/fixtures/gb_la (see tests/make_fixture.py): no network, no dependence on
later data. The hand calculations in the comments use the Rams' last five games before this one:

    week  team carries  team plays  favoured by   Kyren Williams carries
      6       20           56          12.5                10
      7       38           68           3.5                16
      8       27           53           6.5                16
      9       29           73           4.5                15
     10       46           74          10.0                12
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model  # noqa: E402
from common import norm_name  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "gb_la"


class MathTests(unittest.TestCase):
    def test_american_odds_to_probability(self):
        self.assertAlmostEqual(model.american_to_prob(-110), 110 / 210)
        self.assertAlmostEqual(model.american_to_prob(150), 0.4)

    def test_vig_removed_from_a_two_way_price(self):
        over, under = model.devig_two_way(-110, -110)
        self.assertAlmostEqual(over, 0.5)
        self.assertAlmostEqual(under, 0.5)
        # DraftKings, Kyren Williams rush attempts: over -132, under +104
        over, under = model.devig_two_way(-132, 104)
        self.assertAlmostEqual(over + under, 1.0)
        self.assertAlmostEqual(under, (100 / 204) / (132 / 232 + 100 / 204))

    def test_poisson_when_dispersion_is_one(self):
        self.assertAlmostEqual(model.count_cdf(2, 2.0), 5 * 2.718281828459045 ** -2, places=9)  # e^-2 (1 + 2 + 2)
        over, under = model.count_over_under(2.5, 2.0)
        self.assertAlmostEqual(over + under, 1.0)
        self.assertAlmostEqual(under, model.count_cdf(2, 2.0))

    def test_negative_binomial_has_the_asked_for_mean_and_variance(self):
        mean, dispersion = 12.0, 2.5
        pmf = [model.count_cdf(k, mean, dispersion) - model.count_cdf(k - 1, mean, dispersion) for k in range(0, 200)]
        got_mean = sum(k * p for k, p in enumerate(pmf))
        got_var = sum((k - got_mean) ** 2 * p for k, p in enumerate(pmf))
        self.assertAlmostEqual(sum(pmf), 1.0, places=6)
        self.assertAlmostEqual(got_mean, mean, places=3)
        self.assertAlmostEqual(got_var, dispersion * mean, places=2)

    def test_wider_counts_are_less_sure(self):
        poisson_over, _ = model.count_over_under(8.5, 12.0, 1.0)
        wide_over, _ = model.count_over_under(8.5, 12.0, 2.5)
        self.assertGreater(poisson_over, wide_over)
        self.assertGreater(wide_over, 0.5)

    def test_push_is_taken_out_on_a_whole_number_line(self):
        over, under = model.count_over_under(3, 3.0)
        at_three = model.count_cdf(3, 3.0) - model.count_cdf(2, 3.0)
        self.assertAlmostEqual(over + under, 1.0)
        self.assertAlmostEqual(over, (1 - model.count_cdf(3, 3.0)) / (1 - at_three))

    def test_yards_are_a_coin_flip_at_the_typical_game(self):
        mean = 60.0
        typical = model.typical_yards("rush_yds", mean)
        over, under = model.model_probability("rush_yds", typical, mean)
        self.assertAlmostEqual(over, 0.5, places=6)
        self.assertAlmostEqual(over + under, 1.0)
        self.assertLess(typical, mean)  # the curve leans right, so the typical game is below the average one
        self.assertLess(model.model_probability("rush_yds", mean, mean)[0], 0.5)

    def test_gamma_curve_matches_the_closed_forms(self):
        e = 2.718281828459045
        # shape 1 is the exponential curve: P(X <= x) = 1 - e^(-x / scale). 3 uses the series, 30 the continued fraction
        self.assertAlmostEqual(model.gamma_cdf(3.0, 1.0, 10.0), 1 - e ** -0.3, places=9)
        self.assertAlmostEqual(model.gamma_cdf(30.0, 1.0, 10.0), 1 - e ** -3.0, places=9)
        # shape 2: P(X <= x) = 1 - e^(-z) (1 + z) with z = x / scale
        self.assertAlmostEqual(model.gamma_cdf(10.0, 2.0, 10.0), 1 - e ** -1.0 * 2.0, places=9)
        self.assertAlmostEqual(model.gamma_cdf(50.0, 2.0, 10.0), 1 - e ** -5.0 * 6.0, places=9)
        self.assertEqual(model.gamma_cdf(0.0, 2.0, 10.0), 0.0)
        self.assertEqual(model.gamma_cdf(-5.0, 2.0, 10.0), 0.0)

    def test_gamma_yards_have_the_asked_for_average_and_spread(self):
        mean, cv2 = 60.0, 0.4
        step = 0.05
        grid = [step * i for i in range(1, 40001)]  # out to 2,000 yards
        mass = [model.gamma_over_under(x - step, mean, cv2)[0] - model.gamma_over_under(x, mean, cv2)[0] for x in grid]
        got_mean = sum((x - step / 2) * m for x, m in zip(grid, mass))
        got_var = sum((x - step / 2 - got_mean) ** 2 * m for x, m in zip(grid, mass))
        self.assertAlmostEqual(sum(mass), 1.0, places=6)
        self.assertAlmostEqual(got_mean, mean, places=2)
        self.assertAlmostEqual(got_var / mean ** 2, cv2, places=3)

    def test_gamma_yards_lean_right(self):
        mean, cv2 = 60.0, 0.4
        sigma = mean * cv2 ** 0.5
        far_over = model.gamma_over_under(mean + 2.5 * sigma, mean, cv2)[0]
        same_normal = model.normal_over_under(mean + 2.5 * sigma, mean, sigma)[0]
        self.assertGreater(far_over, same_normal)  # big games are likelier than a normal with the same spread allows
        self.assertLess(model.gamma_median(mean, cv2), mean)
        overs = [model.gamma_over_under(line, mean, cv2)[0] for line in (20.5, 40.5, 60.5, 80.5, 120.5)]
        self.assertEqual(overs, sorted(overs, reverse=True))

    def test_yards_spread_moves_toward_the_players_own_game_log(self):
        mean = 60.0
        league = model.league_cv2("rush_yds", mean)
        base, slope = model.SIGMA["rush_yds"]
        self.assertAlmostEqual(league, ((base + slope * mean) / mean) ** 2)
        self.assertAlmostEqual(model.yards_cv2("rush_yds", mean, [70, 50]), league)  # under three games: league spread only
        self.assertAlmostEqual(model.yards_cv2("rush_yds", mean, [0, 2, 1, 3]), league)  # too few yards to say anything
        steady, wild = [58, 62, 60, 61, 59, 60], [5, 150, 20, 110, 8, 67]
        self.assertLess(model.yards_cv2("rush_yds", mean, steady), league)
        self.assertGreater(model.yards_cv2("rush_yds", mean, wild), league)
        # the log counts as its number of games against YARDS_PRIOR_GAMES games of the league spread
        log = pd.Series(wild, dtype=float)
        own = log.var(ddof=1) / log.mean() ** 2
        expected = (len(log) * own + model.YARDS_PRIOR_GAMES * league) / (len(log) + model.YARDS_PRIOR_GAMES)
        self.assertAlmostEqual(model.yards_cv2("rush_yds", mean, wild), expected)
        low, high = model.CV2_LIMITS
        self.assertGreaterEqual(model.yards_cv2("rush_yds", mean, [60] * 40), low)
        self.assertLessEqual(model.yards_cv2("rec_yds", 6.0, [0, 0, 0, 0, 0, 0, 0, 90]), high)

    def test_each_kind_of_yards_uses_its_own_curve(self):
        # decided 2026-11-25 from the back-test: the gamma for rushing yards, the normal for receiving yards
        self.assertEqual(model.YARDS_CURVE, {"rush_yds": "gamma", "rec_yds": "normal"})
        mean, line = 60.0, 70.5
        self.assertEqual(model.model_probability("rush_yds", line, mean),
                         model.gamma_over_under(line, mean, model.league_cv2("rush_yds", mean)))
        self.assertEqual(model.model_probability("rush_yds", line, mean, 0.3), model.gamma_over_under(line, mean, 0.3))
        self.assertEqual(model.model_probability("rec_yds", line, mean), model.normal_yards_over_under("rec_yds", line, mean))
        # the normal takes no notice of a fitted spread: it uses the league spread
        self.assertEqual(model.model_probability("rec_yds", line, mean, 0.3), model.model_probability("rec_yds", line, mean))
        # each curve is a coin flip at its own typical game, and that game is below the average
        for stat in ("rush_yds", "rec_yds"):
            typical = model.typical_yards(stat, mean)
            self.assertLess(typical, mean)
            self.assertAlmostEqual(model.model_probability(stat, typical, mean)[0], 0.5, places=6)
        self.assertAlmostEqual(model.typical_yards("rec_yds", mean), model.YARDS_CENTRE["rec_yds"] * mean)

    def test_calibrated_and_literal_agree_only_on_the_same_side(self):
        # over or under: the same side of 50%, whatever the book says
        self.assertTrue(model.literal_agrees("rec", 0.60, 0.55, 0.45))
        self.assertTrue(model.literal_agrees("rec", 0.45, 0.40, 0.38))
        self.assertFalse(model.literal_agrees("rec", 0.60, 0.45, 0.45))
        self.assertFalse(model.literal_agrees("rush_yds", 0.48, 0.52, 0.40))
        # touchdown: both above the book's probability
        self.assertTrue(model.literal_agrees("td", 0.41, 0.50, 0.34))
        self.assertTrue(model.literal_agrees("td", 0.41, 0.36, 0.34))
        self.assertFalse(model.literal_agrees("td", 0.41, 0.30, 0.34))
        self.assertFalse(model.literal_agrees("td", 0.30, 0.41, 0.34))

    def test_touchdown_probability(self):
        yes, no = model.model_probability("td", None, 0.5)
        self.assertAlmostEqual(yes, 1 - 2.718281828459045 ** -0.5)
        self.assertAlmostEqual(yes + no, 1.0)

    def test_names_match_across_sources(self):
        self.assertEqual(norm_name("Kenneth Walker III"), "kenneth walker")
        self.assertEqual(norm_name("D.J. Moore"), norm_name("DJ Moore"))
        self.assertEqual(norm_name("Amon-Ra St. Brown"), "amon ra st brown")
        self.assertEqual(norm_name("Marvin Harrison Jr."), "marvin harrison")


class CachedGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        read = lambda name: pd.read_parquet(FIXTURE / f"{name}.parquet")  # noqa: E731
        cls.tables = model.build_tables(pbp=read("pbp"), player_week=read("player_week"), snaps=read("snaps"),
                                        games=read("games"), players=read("players"),
                                        season=int((FIXTURE / "season.txt").read_text().strip()))
        cls.props = pd.read_csv(FIXTURE / "props.csv")
        cls.lines = pd.read_csv(FIXTURE / "game_lines.csv")
        cls.status = json.loads((FIXTURE / "status.json").read_text(encoding="utf-8"))
        cls.result = model.price_slate(cls.tables, cls.props, cls.lines, cls.status)
        cls.kick = pd.Timestamp(cls.lines["commence_time"].iloc[0])
        cls.ids = {row.display_name: row.gsis_id for row in cls.tables.players.itertuples()}

    def price(self, status=None, lines=None, props=None) -> dict:
        return model.price_slate(self.tables, self.props if props is None else props,
                                 self.lines if lines is None else lines, self.status if status is None else status)

    # ---- projections against hand calculations

    def test_team_projection_matches_the_hand_calculation(self):
        team = model.project_team(self.tables, "LA", self.kick, 2.5)
        carries = (20 * 1 + 38 * 2 + 27 * 3 + 29 * 4 + 46 * 5) / 15
        plays = (56 * 1 + 68 * 2 + 53 * 3 + 73 * 4 + 74 * 5) / 15
        usual_fav = (12.5 * 1 + 3.5 * 2 + 6.5 * 3 + 4.5 * 4 + 10.0 * 5) / 15
        self.assertAlmostEqual(team["plays"], plays, places=6)
        self.assertAlmostEqual(team["base_run_rate"], carries / plays, places=6)
        self.assertAlmostEqual(team["usual_fav"], usual_fav, places=6)
        # favoured by 2.5 instead of the usual 7.1 -> a slightly lower run rate
        self.assertAlmostEqual(team["run_rate"], carries / plays + model.SCRIPT_SLOPE * (2.5 - usual_fav), places=6)
        self.assertAlmostEqual(team["runs"], plays * team["run_rate"], places=6)

    def test_carry_share_and_carries_match_the_hand_calculation(self):
        kyren = model.project_player(self.tables, self.ids["Kyren Williams"], "LA", "GB", self.kick, 2.5, 44.0)
        share = (0.5 * 12 + 0.3 * 15 + 0.2 * 16) / (0.5 * 46 + 0.3 * 29 + 0.2 * 27)
        self.assertAlmostEqual(kyren["carry_share"], share, places=6)
        raw = kyren["team_proj"]["runs"] * share
        self.assertAlmostEqual(kyren["raw"]["rush_att"], raw, places=6)
        base, slope = model.CALIBRATION["rush_att"]
        self.assertAlmostEqual(kyren["rush_att"], base + slope * raw, places=6)
        self.assertEqual(kyren["games_used"], 3)
        self.assertEqual(kyren["games_missed"], 0)

    def test_target_share_blends_recent_and_season(self):
        golden = model.project_player(self.tables, self.ids["Matthew Golden"], "GB", "LA", self.kick, -2.5, 44.0)
        recent_weight = model.SHARE_RECENT["target"]
        blended = recent_weight * golden["recent_target_share"] + (1 - recent_weight) * golden["season_target_share"]
        self.assertAlmostEqual(golden["target_share"], blended, places=9)
        self.assertGreater(golden["target_share"], 0.2)  # Green Bay's lead receiver

    # ---- the slate

    def test_one_game_with_the_cached_lines(self):
        self.assertEqual(len(self.result["games"]), 1)
        game = self.result["games"][0]
        self.assertEqual(game["game"], "GB@LA")
        self.assertEqual(game["game_id"], "2026_12_GB_LA")
        self.assertEqual(game["home_spread"], -2.5)
        self.assertEqual(game["total"], 44.0)  # DraftKings 43.5, FanDuel 44.5
        self.assertAlmostEqual(game["implied_points"]["LA"], 23.25)
        self.assertFalse(game["blowout"])

    def test_every_leg_follows_the_rules(self):
        self.assertTrue(self.result["legs"])
        allowed = {spec["label"]: spec["positions"] for spec in model.MARKETS.values()}
        for leg in self.result["legs"]:
            with self.subTest(leg=leg["id"]):
                self.assertGreaterEqual(leg["edge"], model.MIN_EDGE)
                self.assertLess(leg["edge"], model.BIG_EDGE)
                self.assertAlmostEqual(leg["edge"], leg["model_prob"] - leg["implied_prob"], places=3)
                self.assertIn(leg["position"], allowed[leg["market"]])  # rule 1: only these positions for each prop
                self.assertIn(leg["book"], ("DK", "FD"))
                self.assertIn(leg["tier"], ("A", "B"))
                if leg["tier"] == "A":
                    self.assertGreaterEqual(leg["edge"], model.A_TIER_EDGE)
                    self.assertEqual(leg["flags"], [])
                else:
                    self.assertTrue(leg["flags"] or leg["edge"] < model.A_TIER_EDGE)
                self.assertTrue(leg["reasoning"])

    def test_a_known_leg(self):
        leg = next(l for l in self.result["legs"] if l["player"] == "Kyren Williams" and l["market"] == "rush attempts")
        self.assertEqual((leg["side"], leg["line"], leg["book"], leg["price"]), ("under", 12.5, "DK", 104))
        fair_under = model.devig_two_way(-132, 104)[1]
        self.assertAlmostEqual(leg["implied_prob"], fair_under, places=3)
        model_under = model.count_over_under(12.5, leg["average"], model.DISPERSION["rush_att"])[1]
        self.assertAlmostEqual(leg["model_prob"], model_under, places=2)
        self.assertEqual(leg["tier"], "A")

    def test_touchdown_prices_have_the_vig_scaled_out(self):
        priced = [l for l in self.result["legs"] + self.result["skips"] if l["market"] == "anytime TD"]
        self.assertTrue(priced)
        for leg in priced:
            self.assertLess(leg["implied_prob"], model.american_to_prob(leg["price"]))

    def test_small_roles_are_not_priced(self):
        self.assertTrue(any("role is too small" in note for note in self.result["notes"]))
        self.assertFalse([l for l in self.result["legs"] if l["player"] in ("CJ Daniels", "Max Klare", "Mark Redman")])

    def test_a_teammate_out_inside_the_window_is_flagged(self):
        # Puka Nacua is out but played in two of the Rams' last three games, so target shares there describe a
        # different line-up: every Rams leg that rests on a target share says so and cannot be A tier
        flagged = [l for l in self.result["legs"] if any("Puka Nacua" in flag for flag in l["flags"])]
        self.assertTrue(flagged)
        for leg in flagged:
            self.assertEqual(leg["team"], "LA")
            self.assertIn(leg["market"], ("receptions", "receiving yards", "anytime TD"))
            self.assertEqual(leg["tier"], "B")
        rams_catches = [l for l in self.result["legs"] if l["team"] == "LA" and l["market"] in ("receptions", "receiving yards")]
        self.assertEqual({l["id"] for l in rams_catches}, {l["id"] for l in flagged if l["market"] != "anytime TD"})

    def test_quarterback_touchdowns_are_not_priced(self):
        quarterbacks = {p["name"] for p in self.result["projections"] if p["position"] == "QB"}
        posted = set(self.props[self.props["market"] == "player_anytime_td"]["player"])
        self.assertTrue(quarterbacks & posted)  # the books do post them, so this checks something
        everything = self.result["legs"] + self.result["skips"] + self.result["priced"]
        self.assertFalse([row for row in everything if row["player"] in quarterbacks])
        self.assertNotIn("QB", model.MARKETS["player_anytime_td"]["positions"])
        self.assertTrue(any(f"{len(quarterbacks & posted)} quarterback anytime TD props were not priced" in note
                            for note in self.result["notes"]))

    def test_a_leg_the_two_models_disagree_on_is_flagged_and_kept_out_of_the_a_tier(self):
        flipped = [l for l in self.result["legs"] if not l["literal_agrees"]]
        self.assertTrue(flipped)
        for leg in self.result["legs"]:
            with self.subTest(leg=leg["id"]):
                if leg["side"] == "yes":
                    expected = leg["literal_prob"] > leg["implied_prob"]
                else:
                    expected = (leg["model_prob"] > 0.5) == (leg["literal_prob"] > 0.5)
                self.assertEqual(leg["literal_agrees"], expected)
                self.assertEqual(model.FLIP_FLAG in leg["flags"], not expected)
                if not expected:
                    self.assertEqual(leg["tier"], "B")
        # the known case: the rules as written lean over on Matthew Golden's catches, the calibrated model under
        golden = next(l for l in self.result["legs"] if l["player"] == "Matthew Golden" and l["market"] == "receptions")
        self.assertEqual(golden["side"], "under")
        self.assertFalse(golden["literal_agrees"])
        self.assertIn("calibration flips the literal lean", golden["flags"])
        self.assertEqual(golden["tier"], "B")
        self.assertGreaterEqual(golden["edge"], model.A_TIER_EDGE)  # the edge alone would have made it A tier

    def test_the_result_can_be_written_as_json(self):
        # board.py writes legs, skips, gated players and the slate to docs/picks.json; a numpy value in any of
        # them stops the whole run
        lines = self.lines.copy()
        lines["home_ml"], lines["away_ml"] = [-143, -146], [121, 124]
        for result in (self.result, self.price(lines=lines)):
            text = json.dumps({key: result[key] for key in ("games", "legs", "skips", "gated", "notes", "priced")})
            self.assertEqual(len(json.loads(text)["legs"]), len(result["legs"]))
        for leg in self.result["legs"]:
            self.assertIs(type(leg["literal_agrees"]), bool)

    def test_moneylines_are_carried_for_ticket_anchors(self):
        self.assertEqual(self.result["games"][0]["moneylines"], {})  # the cached lines were pulled before moneylines were
        lines = self.lines.copy()
        lines["home_ml"] = [-143, -146]
        lines["away_ml"] = [121, 124]
        money = self.price(lines=lines)["games"][0]["moneylines"]
        self.assertEqual(set(money), {"GB", "LA"})
        self.assertEqual(money["LA"]["DK"]["price"], -143)
        self.assertEqual(money["GB"]["FD"]["price"], 124)
        for book in ("DK", "FD"):
            self.assertAlmostEqual(money["LA"][book]["fair"] + money["GB"][book]["fair"], 1.0, places=3)
        self.assertAlmostEqual(money["LA"]["DK"]["fair"], model.devig_two_way(-143, 121)[0], places=4)
        # a book with only one side posted gives no moneyline for either team
        lines.loc[lines["book"] == "fanduel", "away_ml"] = None
        self.assertEqual(set(self.price(lines=lines)["games"][0]["moneylines"]["LA"]), {"DK"})

    def test_an_empty_window_gives_an_empty_board(self):
        result = self.price(props=self.props.iloc[0:0], lines=self.lines.iloc[0:0])
        self.assertEqual((result["games"], result["legs"], result["skips"], result["gated"]), ([], [], [], []))

    def test_every_leg_carries_the_literal_probability(self):
        for leg in self.result["legs"]:
            self.assertTrue(0.0 < leg["literal_prob"] < 1.0)
        golden = next(l for l in self.result["legs"] if l["player"] == "Matthew Golden" and l["market"] == "receptions")
        literal_mean = model.project_player(self.tables, self.ids["Matthew Golden"], "GB", "LA", self.kick, -2.5, 44.0)["literal"]["rec"]
        self.assertAlmostEqual(golden["literal_prob"], model.count_over_under(golden["line"], literal_mean, 1.0)[1], places=3)
        self.assertLess(golden["literal_prob"], 0.5)  # the rules as written lean over here; the calibrated model leans under
        self.assertGreater(golden["model_prob"], 0.5)

    def test_rooting_conflict_marks_unders_on_a_rooted_team_and_nothing_else(self):
        plain = self.result  # priced with no rooting teams
        self.assertFalse(any(l["rooting_conflict"] for l in plain["legs"]))
        rooted = model.price_slate(self.tables, self.props, self.lines, self.status, rooting=frozenset({"GB", "MIL"}))
        self.assertEqual(len(rooted["legs"]), len(plain["legs"]))  # tagged, never removed
        for before, after in zip(plain["legs"], rooted["legs"]):
            self.assertEqual((before["id"], before["tier"], before["edge"]), (after["id"], after["tier"], after["edge"]))
        for leg in rooted["legs"]:
            expected = leg["team"] == "GB" and leg["side"] == "under"
            self.assertEqual(leg["rooting_conflict"], expected)
            self.assertEqual(leg["tags"], ["rooting conflict"] if expected else [])
        self.assertTrue(any(l["rooting_conflict"] for l in rooted["legs"]))

    def test_legs_are_grouped_by_the_game_script_that_helps_them(self):
        for leg in self.result["legs"]:
            ahead = leg["correlation_group"].split(": ")[1].replace(" ahead", "")
            if leg["market"] in ("rush attempts", "rush yards"):
                self.assertEqual(ahead, leg["team"] if leg["side"] == "over" else leg["opponent"])
            elif leg["market"] in ("receptions", "receiving yards"):
                self.assertEqual(ahead, leg["opponent"] if leg["side"] == "over" else leg["team"])
            else:
                self.assertEqual(ahead, leg["team"])

    # ---- the injury gate

    def with_player_status(self, name: str, team: str, state: str, practice: dict | None = None) -> dict:
        status = copy.deepcopy(self.status)
        status["players"] = [p for p in status["players"] if p["name"] != name]
        status["players"].append({
            "player_key": self.ids[name], "gsis_id": self.ids[name], "espn_id": None, "name": name, "team": team,
            "position": "RB", "status": state, "designation": state.title(), "injury": "Ankle", "note_utc": None,
            "source": "test", "practice": practice or {"seen": [], "latest": None, "none_this_week": None}})
        return status

    def test_a_player_who_is_out_is_dropped(self):
        result = self.price(status=self.with_player_status("Kyren Williams", "LA", "out"))
        self.assertFalse([l for l in result["legs"] if l["player"] == "Kyren Williams"])
        self.assertTrue(any(g["reason"] == "Kyren Williams is out" for g in result["gated"]))

    def test_questionable_without_practice_is_dropped(self):
        none = {"seen": [{"date": "2026-11-24", "level": "dnp", "source": "espn_note"}], "latest": "dnp", "none_this_week": True}
        result = self.price(status=self.with_player_status("Kyren Williams", "LA", "questionable", none))
        self.assertFalse([l for l in result["legs"] if l["player"] == "Kyren Williams"])
        self.assertTrue(any("has not practised" in g["reason"] for g in result["gated"]))

    def test_questionable_with_practice_is_kept_but_marked(self):
        limited = {"seen": [{"date": "2026-11-24", "level": "limited", "source": "espn_note"}], "latest": "limited",
                   "none_this_week": False}
        result = self.price(status=self.with_player_status("Kyren Williams", "LA", "questionable", limited))
        legs = [l for l in result["legs"] if l["player"] == "Kyren Williams"]
        self.assertTrue(legs)
        for leg in legs:
            self.assertEqual(leg["status"], "questionable")
            self.assertEqual(leg["tier"], "B")
            self.assertTrue(any(flag.startswith("questionable") for flag in leg["flags"]))

    def test_starting_qb_out_drops_his_whole_offense(self):
        status = copy.deepcopy(self.status)
        status["teams"]["LA"]["starting_qb"]["status"] = "out"
        result = self.price(status=status)
        self.assertFalse([l for l in result["legs"] if l["team"] == "LA"])
        self.assertTrue([l for l in result["legs"] if l["team"] == "GB"])
        self.assertTrue(any("starting QB Matthew Stafford is out" in g["reason"] for g in result["gated"]))

    def test_no_status_data_keeps_legs_out_of_the_a_tier(self):
        result = self.price(status={"players": [], "teams": {}})
        self.assertTrue(result["legs"])
        for leg in result["legs"]:
            self.assertEqual(leg["tier"], "B")
            self.assertTrue(any(flag.startswith("no status data") for flag in leg["flags"]))

    # ---- the blowout rule

    def test_blowout_swaps_the_main_yardage_line_for_a_lower_alt(self):
        lines = self.lines.copy()
        lines["home_spread"] = -10.5  # Rams as 10.5-point favourites
        template = self.props[(self.props["player"] == "Kyren Williams") & (self.props["market"] == "player_rush_yds")
                              & (self.props["side"] == "Over")].iloc[0]
        alts = [{**template.to_dict(), "market": "player_rush_yds_alternate", "book": "draftkings", "line": line, "price": price}
                for line, price in ((44.5, -110), (34.5, -150), (24.5, -260))]
        result = self.price(lines=lines, props=pd.concat([self.props, pd.DataFrame(alts)], ignore_index=True))
        self.assertTrue(result["games"][0]["blowout"])
        self.assertEqual(result["games"][0]["favorite"], "LA")

        kyren = [l for l in result["legs"] if l["player"] == "Kyren Williams"]
        self.assertFalse([l for l in kyren if l["market"] == "rush yards"])  # the main line is not used
        alt = [l for l in kyren if l["market"] == "rush yards (alt line)"]
        self.assertEqual(len(alt), 1)
        self.assertEqual(alt[0]["side"], "over")
        self.assertLess(alt[0]["line"], 55.5)
        self.assertGreaterEqual(alt[0]["edge"], model.MIN_EDGE)
        self.assertEqual(alt[0]["tier"], "B")
        self.assertTrue(any(flag.startswith("blowout") for flag in alt[0]["flags"]))
        self.assertIn("Blowout rule", alt[0]["reasoning"])
        self.assertEqual(alt[0]["correlation_group"], "GB@LA: LA ahead")  # an alt over is a rush yards over
        # no Rams starter keeps a main yardage line
        starters = [l for l in result["legs"] if l["team"] == "LA" and l["starter"]
                    and l["market"] in ("rush yards", "receiving yards")]
        self.assertFalse(starters)
        # touchdown legs and the underdog's legs are untouched by the rule
        self.assertTrue([l for l in result["legs"] if l["team"] == "GB" and l["market"] == "receiving yards"])
        self.assertTrue([l for l in result["legs"] + result["skips"] if l["team"] == "LA" and l["market"] == "anytime TD"])

    def test_blowout_with_no_alt_line_skips_the_main_line_and_says_why(self):
        lines = self.lines.copy()
        lines["home_spread"] = -10.5
        result = self.price(lines=lines)  # the cached props hold no alt lines
        self.assertFalse([l for l in result["legs"] if l["player"] == "Kyren Williams" and "rush yards" in l["market"]])
        skipped = [s for s in result["skips"] if s["player"] == "Kyren Williams" and s["market"] == "rush yards"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["kind"], "blowout")
        self.assertIn("Blowout rule: LA are favoured by 10.5", skipped[0]["reason"])
        self.assertIn("No alt line with a 5% edge was priced", skipped[0]["reason"])


if __name__ == "__main__":
    unittest.main()
