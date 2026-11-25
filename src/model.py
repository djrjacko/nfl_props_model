"""Projections, probabilities and edges for the prop board.

    python src/model.py              # price the cached odds for the games in the window
    python src/model.py --backtest   # how the projections did on past weeks of this season
    python src/model.py --literal    # price with the rules exactly as written, no calibration

How a projection is built (the rules come from the owner's betting audit):

  team volume    plays and run rate over the team's last 5 games, recency weighted, with the run rate
                 moved 0.3 points per point of spread relative to the spreads those 5 games were played at
  player volume  team volume x the player's carry share or target share over his last 3 games played
                 (weights 50/30/20), with snap share used as a sanity check
  yards          volume x efficiency, where efficiency is 60% the player's own and 40% what the
                 opponent has allowed to his position
  anytime TD     the player's share of his team's scoring chances x the touchdowns the spread and
                 total imply for his team. Running backs, receivers and tight ends only: a quarterback's
                 anytime TD is not priced, because a share of touches does not describe how he scores

  probability    Poisson family for attempts, receptions and touchdowns; for yards a right-skewed
                 gamma curve (rushing) or a normal curve (receiving), see point 3 below
  edge           model probability minus the book's probability with the vig removed

The calibration layer. Back-testing those rules on 2026 weeks 5-11 (every player-game projected
from earlier games only) showed three things, and each has a measured correction:

  1. Raw projections ran high for players with a role (carries +5%, receptions +13%), because a hot
     three-game share falls back. Each raw projection is therefore pulled toward the middle:
     calibrated = a + b x raw, with b between 0.82 and 0.86. For targets the three-game share turned
     out to add nothing to the season share, so target share is 30% last-3 and 70% season; carry
     share stays on the last 3 games, where it is the best single predictor.
  2. Results are wider than a pure Poisson. Carries were 2.5 times as variable, so a pure Poisson
     said 85% for things that happened 71% of the time. Counts use a negative binomial, which is a
     Poisson with the variance widened by a measured factor (1.0 gives back the pure Poisson).
  3. Yardage is lopsided: big games pull the average up, and only 42% of results beat the average.
     Two curves handle that, and YARDS_CURVE says which one each stat uses:
       rushing yards    a gamma curve whose average is the calibrated projection. Its spread is the
                        league-wide spread moved toward the player's own game log, which counts as
                        his games against 12 games of the league figure (a game log alone
                        understates how far a projection can miss).
       receiving yards  a normal curve centred on the typical game, about 0.86 x the average, with
                        the league-wide spread.
     Both were tried for both stats on 2026-11-25. The gamma was the closer one for rushing yards.
     For receiving yards it made unders more likely for small roles than the results support, so
     receiving yards stayed on the normal. --backtest prints the two side by side.

  Touchdowns: a pure red-zone share said 62% for players who scored 52% of the time, and 6% for
  players who scored 13%. The share used is 0.015 + 0.20 x red-zone share + 0.60 x overall share of
  the team's carries and targets, which was accurate in every bucket.

Every leg is also priced by the rules exactly as written (its "literal" probability). When the two
disagree about the side, the leg is flagged "calibration flips the literal lean": for an over or an
under that means the two probabilities fall on opposite sides of 50%, and for a touchdown that the
as-written probability is not above the book's. A flagged leg cannot be A tier and board.py keeps it
out of every same-game ticket and boost.

--literal switches the calibration off. --backtest prints the evidence again from current data. The
fit and the check use the same weeks, with a split-sample check done by hand on 2026-11-24.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

import fetch_nflverse as nfl
import fetch_odds
import fetch_status
from common import az, load_config, norm_name, utc_now

# ----------------------------------------------------------------- constants

TEAM_WEIGHTS = (5, 4, 3, 2, 1)  # team's last 5 games, most recent first
PLAYER_WEIGHTS = (0.5, 0.3, 0.2)  # player's last 3 games played, most recent first
# How much of a share comes from those last 3 games; the rest is his season with the team. Carry share
# is all last-3 (the best single predictor of next-game carry share). Target share is not: over weeks
# 5-11 the last-3 share missed by 0.082 a game, the season share by 0.077, and a fit of both put its
# weight on the season share (0.84 against 0.06), which held up on weeks 9-11 when fitted on weeks 5-8.
SHARE_RECENT = {"carry": 1.0, "target": 0.3}
RED_ZONE_WEIGHTS = (5, 4, 3, 2, 1)  # red-zone samples are small, so five games

SCRIPT_SLOPE = 0.003  # run-rate change per point of spread (measured within-team: 0.0031)
SCRIPT_CAP = 14.0  # spreads beyond two touchdowns do not move the mix any further
RUN_RATE_LIMITS = (0.25, 0.65)

EFFICIENCY_OWN, EFFICIENCY_OPPONENT = 0.6, 0.4
PRIOR_CARRIES = 30  # a player's own efficiency is pulled toward his position's average
PRIOR_TARGETS = 25  # by this many league-average carries or targets
MIN_OPPONENT_CARRIES, MIN_OPPONENT_TARGETS = 40, 30  # below this, use the league average instead

# offensive touchdowns a team is priced for, from its market-implied points (326 team-games)
TD_INTERCEPT, TD_SLOPE, TD_FLOOR = -1.50, 0.176, 0.6
# how often a red-zone chance scores, by distance from the goal line: (carry, target)
RED_ZONE_VALUE = {5: (0.41, 0.52), 10: (0.12, 0.29), 20: (0.03, 0.16)}
# different players who score a touchdown in a game, from the game total (163 games)
SCORERS_INTERCEPT, SCORERS_SLOPE = -0.85, 0.114

# ---- the calibration layer (see the module docstring); use_literal_rules() neutralises all of it
CALIBRATION = {"rush_att": (1.71, 0.822), "rush_yds": (5.49, 0.860), "rec": (0.314, 0.829), "rec_yds": (3.42, 0.860)}
DISPERSION = {"rush_att": 2.5, "rec": 1.3}  # variance / mean; 1.0 is a pure Poisson
SIGMA = {"rush_yds": (22.0, 0.19), "rec_yds": (14.5, 0.41)}  # league-wide spread of yards = base + slope x average
# The curve each kind of yards uses: "gamma" or "normal".
#   gamma   leans right, and its average is the calibrated projection. Its spread starts from SIGMA and is moved
#           toward the player's own game log, which counts as his games against YARDS_PRIOR_GAMES games of the
#           league spread. A log alone understates how far a projection can miss (variance / mean^2 of 0.27 in
#           the logs against 0.48 in the back-test for rushing), so the league spread keeps most of the weight.
#   normal  centred on the typical game (YARDS_CENTRE x the average), with the league spread.
# Back-tested 2026-11-25. The gamma is the closer one for rushing. For receivers projected under 40 yards it
# overstates unders (71.2% said, 67.8% happened at a line 25% above the projection; the normal said 66.9%), so
# receiving yards use the normal.
YARDS_CURVE = {"rush_yds": "gamma", "rec_yds": "normal"}
YARDS_PRIOR_GAMES = 12
CV2_LIMITS = (0.08, 2.5)
YARDS_CENTRE = {"rush_yds": 0.88, "rec_yds": 0.86}  # normal curve only: the typical game as a fraction of the average
TD_SHARE = (0.015, 0.20, 0.60)  # floor + x red-zone share + x overall carry-and-target share
FLIP_FLAG = "calibration flips the literal lean"  # the calibrated model and the rules as written disagree on the side

MIN_EDGE = 0.05
A_TIER_EDGE = 0.08
BIG_EDGE = 0.20  # an edge this large means the model is missing something, so it is not offered as a leg
# raw volume a player needs before a leg is offered; the back-test only covers players above these
MIN_ROLE = {"carries": 6.0, "targets": 3.0}
SKIP_EDGE = -0.03  # a popular prop this far against the model is listed as a skip
BLOWOUT_SPREAD = 10.0
STARTER_SNAP_SHARE = 0.50
LINE_GAP = {"yards": 2.0, "count": 0.5}  # flag when the books differ by MORE than this
ROLE_SHARE = {"carry": 0.15, "target": 0.12}  # a teammate this involved changes roles when he sits or returns
SNAP_JUMP = 0.20
PART_TIME_SNAPS = 0.35
ALT_PRICE_FLOOR = -300  # do not suggest an alt line juiced past this
DEFAULT_HOLD = 0.07  # assumed margin on a one-sided alt line when the main line cannot supply one
NOT_ON_TEAM = {"CUT", "RET", "RLS", "TRC", "TRD", "TRT"}  # nflverse roster codes for players who are gone

SKILL = ("QB", "RB", "WR", "TE")
MARKETS = {
    "player_rush_attempts": {"stat": "rush_att", "kind": "count", "positions": ("RB",), "label": "rush attempts"},
    "player_rush_yds": {"stat": "rush_yds", "kind": "yards", "positions": ("RB",), "label": "rush yards"},
    "player_receptions": {"stat": "rec", "kind": "count", "positions": ("RB", "WR", "TE"), "label": "receptions"},
    "player_reception_yds": {"stat": "rec_yds", "kind": "yards", "positions": ("WR", "TE"), "label": "receiving yards"},
    # Quarterbacks are left out: the touch-share method does not describe how a quarterback scores.
    "player_anytime_td": {"stat": "td", "kind": "td", "positions": ("RB", "WR", "TE"), "label": "anytime TD"},
}
ALT_OF = {"player_rush_yds": "player_rush_yds_alternate", "player_reception_yds": "player_reception_yds_alternate"}
BOOK_NAME = {"draftkings": "DK", "fanduel": "FD"}


def use_literal_rules() -> None:
    """Switch the calibration layer off: last-3 shares only, no pull toward the middle, pure Poisson, normal
    centred on the average, pure red-zone share."""
    for stat in CALIBRATION:
        CALIBRATION[stat] = (0.0, 1.0)
    for stat in DISPERSION:
        DISPERSION[stat] = 1.0
    for kind in SHARE_RECENT:
        SHARE_RECENT[kind] = 1.0
    for stat in YARDS_CENTRE:
        YARDS_CENTRE[stat] = 1.0
    for stat in YARDS_CURVE:
        YARDS_CURVE[stat] = "normal"
    globals()["TD_SHARE"] = (0.0, 1.0, 0.0)


# ---------------------------------------------------------------------- math


def american_to_prob(price: float) -> float:
    return 100.0 / (price + 100.0) if price > 0 else -price / (100.0 - price)


def devig_two_way(over_price: float, under_price: float) -> tuple[float, float]:
    """Remove the vig from an over/under pair by scaling both sides to sum to 1."""
    over, under = american_to_prob(over_price), american_to_prob(under_price)
    return over / (over + under), under / (over + under)


def count_cdf(k: int, mean: float, dispersion: float = 1.0) -> float:
    """P(X <= k) for a count with this mean. dispersion 1 is a Poisson; above 1 a negative binomial with
    variance = dispersion x mean."""
    if k < 0:
        return 0.0
    if dispersion <= 1.0001:
        term = total = math.exp(-mean)
        for i in range(1, k + 1):
            term *= mean / i
            total += term
        return min(1.0, total)
    size, step = mean / (dispersion - 1.0), 1.0 - 1.0 / dispersion
    term = total = dispersion ** (-size)
    for i in range(k):
        term *= (i + size) / (i + 1) * step
        total += term
    return min(1.0, total)


def count_over_under(line: float, mean: float, dispersion: float = 1.0) -> tuple[float, float]:
    """P(over), P(under). On a whole-number line the push is taken out and the rest rescaled."""
    under = count_cdf(math.ceil(line) - 1, mean, dispersion)
    over = 1.0 - count_cdf(math.floor(line), mean, dispersion)
    return over / (over + under), under / (over + under)


def normal_over_under(line: float, centre: float, sigma: float) -> tuple[float, float]:
    under = 0.5 * (1.0 + math.erf((line - centre) / (sigma * math.sqrt(2.0))))
    return 1.0 - under, under


def gamma_cdf(x: float, shape: float, scale: float) -> float:
    """P(X <= x) for a gamma curve: the regularised lower incomplete gamma function, by series below
    shape + 1 and by continued fraction above it."""
    if x <= 0:
        return 0.0
    z = x / scale
    log_front = -z + shape * math.log(z) - math.lgamma(shape)
    if z < shape + 1.0:
        term = total = 1.0 / shape
        n = shape
        for _ in range(500):
            n += 1.0
            term *= z / n
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        return min(1.0, total * math.exp(log_front))
    tiny = 1e-300
    b = z + 1.0 - shape
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return max(0.0, 1.0 - math.exp(log_front) * h)


def gamma_over_under(line: float, mean: float, cv2: float) -> tuple[float, float]:
    """P(over), P(under) for yards with this average and this spread (cv2 = variance / mean squared)."""
    shape = 1.0 / cv2
    under = gamma_cdf(line, shape, mean / shape)
    return 1.0 - under, under


def gamma_median(mean: float, cv2: float) -> float:
    """The game a player is as likely to beat as to miss. It sits below the average because the curve leans right."""
    shape = 1.0 / cv2
    low, high = 0.0, mean
    for _ in range(60):
        middle = (low + high) / 2.0
        if gamma_cdf(middle, shape, mean / shape) < 0.5:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def league_cv2(stat: str, mean: float) -> float:
    """The league-wide spread of yards around a projection, as variance / mean squared."""
    base, slope = SIGMA[stat]
    mean = max(mean, 1.0)
    return ((base + slope * mean) / mean) ** 2


def yards_cv2(stat: str, mean: float, game_log) -> float:
    """Spread for one player: the league spread moved toward the spread in his own game log."""
    cv2 = league_cv2(stat, mean)
    log = np.asarray(game_log, dtype=float)
    if len(log) >= 3 and log.mean() >= 5.0:
        own = float(log.var(ddof=1) / log.mean() ** 2)
        cv2 = (len(log) * own + YARDS_PRIOR_GAMES * cv2) / (len(log) + YARDS_PRIOR_GAMES)
    return float(min(max(cv2, CV2_LIMITS[0]), CV2_LIMITS[1]))


def normal_yards_over_under(stat: str, line: float, mean: float) -> tuple[float, float]:
    """The yards curve used before the gamma: a normal centred on the typical game, with the league spread."""
    base, slope = SIGMA[stat]
    return normal_over_under(line, YARDS_CENTRE[stat] * mean, base + slope * max(mean, 0.0))


def typical_yards(stat: str, mean: float, cv2: float | None = None) -> float:
    """The median of whichever yards curve is in use."""
    if YARDS_CURVE[stat] == "gamma":
        return gamma_median(max(mean, 1e-6), cv2 if cv2 is not None else league_cv2(stat, mean))
    return YARDS_CENTRE[stat] * mean


def model_probability(stat: str, line: float | None, mean: float, cv2: float | None = None) -> tuple[float, float]:
    """(P(over or yes), P(under or no)) for a calibrated projection of `mean`. `cv2` is the yards spread
    fitted for the player; without it the league spread is used."""
    if stat == "td":
        yes = 1.0 - math.exp(-mean)
        return yes, 1.0 - yes
    if stat in DISPERSION:
        return count_over_under(line, max(mean, 1e-6), DISPERSION[stat])
    if YARDS_CURVE[stat] == "gamma":
        return gamma_over_under(line, max(mean, 1e-6), cv2 if cv2 is not None else league_cv2(stat, mean))
    return normal_yards_over_under(stat, line, mean)


def literal_probability(stat: str, line: float | None, mean: float) -> tuple[float, float]:
    """The same question answered by the rules exactly as written: a pure Poisson for counts and a normal
    centred on the average for yards. `mean` is the uncalibrated projection."""
    if stat == "td":
        yes = 1.0 - math.exp(-mean)
        return yes, 1.0 - yes
    if stat in DISPERSION:
        return count_over_under(line, max(mean, 1e-6), 1.0)
    base, slope = SIGMA[stat]
    return normal_over_under(line, mean, base + slope * max(mean, 0.0))


def literal_agrees(stat: str, model_p: float, literal_p: float, fair_p: float) -> bool:
    """Do the calibrated model and the rules as written point the same way on this bet?

    For an over or an under: both probabilities on the same side of 50%. For a touchdown, which has no other
    side to take: both above the book's probability with the vig removed.
    """
    if stat == "td":
        return bool(model_p > fair_p and literal_p > fair_p)
    return bool((model_p > 0.5) == (literal_p > 0.5))  # bool(): a numpy true/false cannot be written to JSON


def expected_scorers(total: float) -> float:
    return max(2.0, SCORERS_INTERCEPT + SCORERS_SLOPE * total)


def implied_team_tds(implied_points: float) -> float:
    return max(TD_FLOOR, TD_INTERCEPT + TD_SLOPE * implied_points)


# -------------------------------------------------------------------- tables


@dataclass
class Tables:
    """nflverse data reshaped for projecting: one row per team-game and per player-game."""

    games: pd.DataFrame
    team_games: pd.DataFrame
    player_games: pd.DataFrame
    players: pd.DataFrame
    league: dict
    season: int
    _team: dict = field(default_factory=dict)
    _player: dict = field(default_factory=dict)
    _defense: dict = field(default_factory=dict)
    _team_cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._team = {k: v.sort_values("kick") for k, v in self.team_games.groupby("team")}
        self._player = {k: v.sort_values("kick") for k, v in self.player_games.groupby("gsis_id")}
        self._defense = {k: v for k, v in self.player_games.groupby("opponent")}

    def team_before(self, team: str, kick) -> pd.DataFrame:
        rows = self._team.get(team)
        return rows[rows["kick"] < kick] if rows is not None else self.team_games.iloc[0:0]

    def player_before(self, gsis_id: str, kick) -> pd.DataFrame:
        rows = self._player.get(gsis_id)
        return rows[rows["kick"] < kick] if rows is not None else self.player_games.iloc[0:0]

    def defense_before(self, team: str, kick) -> pd.DataFrame:
        rows = self._defense.get(team)
        return rows[rows["kick"] < kick] if rows is not None else self.player_games.iloc[0:0]


def build_tables(pbp: pd.DataFrame, player_week: pd.DataFrame, snaps: pd.DataFrame, games: pd.DataFrame,
                 players: pd.DataFrame, season: int) -> Tables:
    games = games[games["season"] == season].copy()
    games["kick"] = nfl.kickoff_arizona(games).dt.tz_convert("UTC")

    # ---- plays -> team-game totals and each player's red-zone chances
    p = pbp[(pbp["season"] == season) & (pbp["two_point_attempt"] == 0) & pbp["posteam"].notna()].copy()
    p["carry"] = ((p["rush_attempt"] == 1) & p["rusher_player_id"].notna()).astype(float)
    p["target"] = ((p["pass_attempt"] == 1) & (p["sack"] == 0) & p["receiver_player_id"].notna()).astype(float)
    carry_value, target_value = pd.Series(0.0, index=p.index), pd.Series(0.0, index=p.index)
    for limit in sorted(RED_ZONE_VALUE, reverse=True):  # 20 first, then the closer zones overwrite it
        inside = p["yardline_100"] <= limit
        carry_value[inside], target_value[inside] = RED_ZONE_VALUE[limit]
    p["rz_n"] = (p["carry"] + p["target"]) * (p["yardline_100"] <= 20)
    p["rz_x"] = p["carry"] * carry_value + p["target"] * target_value
    p["off_td"] = ((p["rush_touchdown"] == 1) | (p["pass_touchdown"] == 1)).astype(float)
    team_games = p.groupby(["game_id", "posteam"], as_index=False).agg(
        rush_att=("rush_attempt", "sum"), dropbacks=("pass_attempt", "sum"), targets=("target", "sum"),
        rz_n=("rz_n", "sum"), rz_x=("rz_x", "sum"), off_tds=("off_td", "sum")).rename(columns={"posteam": "team"})
    team_games["plays"] = team_games["rush_att"] + team_games["dropbacks"]
    team_games = team_games.merge(games[["game_id", "week", "kick", "home_team", "away_team", "spread_line", "total_line"]],
                                  on="game_id")
    at_home = team_games["team"] == team_games["home_team"]
    team_games["opponent"] = np.where(at_home, team_games["away_team"], team_games["home_team"])
    team_games["fav"] = np.where(at_home, team_games["spread_line"], -team_games["spread_line"])  # points favoured by
    team_games["implied"] = team_games["total_line"] / 2 + team_games["fav"] / 2

    chances = pd.concat([
        p[p["carry"] == 1].groupby(["game_id", "rusher_player_id"])[["rz_n", "rz_x"]].sum().rename_axis(["game_id", "gsis_id"]),
        p[p["target"] == 1].groupby(["game_id", "receiver_player_id"])[["rz_n", "rz_x"]].sum().rename_axis(["game_id", "gsis_id"]),
    ]).groupby(level=[0, 1]).sum().reset_index()

    # ---- player-games: everyone who took an offensive snap, with his stat line (zeros if he had none)
    current = players[players["gsis_id"].notna()]
    by_pfr = current.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    names = current[current["last_season"] == season].assign(key=lambda d: d["display_name"].map(norm_name))
    by_name = names.drop_duplicates("key", keep=False).set_index("key")["gsis_id"]
    snaps = snaps[(snaps["season"] == season) & (snaps["offense_snaps"] > 0)].copy()
    snaps["gsis_id"] = snaps["pfr_player_id"].map(by_pfr)
    missing = snaps["gsis_id"].isna()
    snaps.loc[missing, "gsis_id"] = snaps.loc[missing, "player"].map(norm_name).map(by_name)
    snaps = snaps.dropna(subset=["gsis_id"]).drop_duplicates(["game_id", "gsis_id"])
    snaps = snaps[["game_id", "team", "gsis_id", "position", "offense_pct"]].rename(columns={"position": "snap_position"})

    stats = player_week[(player_week["season"] == season) & player_week["player_id"].notna()].rename(
        columns={"player_id": "gsis_id", "player_display_name": "name"})
    stats = stats[["game_id", "team", "gsis_id", "name", "position", "carries", "rushing_yards", "rushing_tds", "targets",
                   "receptions", "receiving_yards", "receiving_tds"]]
    pg = stats.merge(snaps, on=["game_id", "team", "gsis_id"], how="outer")
    pg["position"] = pg["position"].fillna(pg["snap_position"])
    pg["name"] = pg["name"].fillna(pg["gsis_id"].map(current.drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]))
    pg = pg[pg["position"].isin(SKILL)].drop(columns="snap_position")
    stat_cols = ["carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds"]
    pg[stat_cols] = pg[stat_cols].fillna(0).astype(float)
    pg = pg.merge(chances, on=["game_id", "gsis_id"], how="left").fillna({"rz_n": 0.0, "rz_x": 0.0})
    pg = pg.merge(team_games[["game_id", "team", "opponent", "week", "kick", "rush_att", "targets", "rz_n", "rz_x"]].rename(
        columns={"rush_att": "team_rush_att", "targets": "team_targets", "rz_n": "team_rz_n", "rz_x": "team_rz_x"}),
        on=["game_id", "team"])

    def rate(frame: pd.DataFrame, top: str, bottom: str) -> float:
        return float(frame[top].sum() / frame[bottom].sum()) if frame[bottom].sum() else 0.0

    league = {"ypc": {"RB": rate(pg[pg["position"] == "RB"], "rushing_yards", "carries")}, "catch": {}, "ypt": {}}
    for pos in ("RB", "WR", "TE"):
        league["catch"][pos] = rate(pg[pg["position"] == pos], "receptions", "targets")
        league["ypt"][pos] = rate(pg[pg["position"] == pos], "receiving_yards", "targets")
    return Tables(games=games, team_games=team_games, player_games=pg, players=current, league=league, season=season)


PBP_COLUMNS = ["game_id", "season", "posteam", "rush_attempt", "pass_attempt", "sack", "two_point_attempt",
               "rusher_player_id", "receiver_player_id", "rush_touchdown", "pass_touchdown", "yardline_100"]
PLAYER_WEEK_COLUMNS = ["player_id", "player_display_name", "position", "season", "game_id", "team", "carries",
                       "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds"]
SNAP_COLUMNS = ["game_id", "season", "player", "pfr_player_id", "position", "team", "offense_snaps", "offense_pct"]
GAME_COLUMNS = ["game_id", "season", "week", "gameday", "gametime", "away_team", "home_team", "away_score", "home_score",
                "spread_line", "total_line"]
PLAYER_COLUMNS = ["gsis_id", "display_name", "position", "latest_team", "status", "last_season", "pfr_id"]


def load_tables() -> Tables:
    return build_tables(pbp=nfl.load("pbp", PBP_COLUMNS), player_week=nfl.load("player_week", PLAYER_WEEK_COLUMNS),
                        snaps=nfl.load("snap_counts", SNAP_COLUMNS), games=nfl.load("schedules", GAME_COLUMNS),
                        players=nfl.load("players", PLAYER_COLUMNS), season=nfl.run_info()["current_season"])


# --------------------------------------------------------------- projections


def _weights(count: int, scheme: tuple) -> np.ndarray:
    """Weights for `count` games in oldest-to-newest order (the newest game gets scheme[0])."""
    return np.array(scheme[:count][::-1], dtype=float)


def project_team(t: Tables, team: str, kick, fav: float) -> dict | None:
    """Team plays, runs and targets for a game at `kick`, when the team is favoured by `fav` points."""
    key = (team, kick, round(float(fav), 2))
    if key not in t._team_cache:
        t._team_cache[key] = _project_team(t, team, kick, fav)
    return t._team_cache[key]


def _project_team(t: Tables, team: str, kick, fav: float) -> dict | None:
    recent = t.team_before(team, kick).tail(len(TEAM_WEIGHTS))
    if recent.empty:
        return None
    w = _weights(len(recent), TEAM_WEIGHTS)
    plays = float((w * recent["plays"]).sum() / w.sum())
    base_rate = float((w * recent["rush_att"]).sum() / (w * recent["plays"]).sum())
    usual_fav = float((w * recent["fav"].fillna(0.0)).sum() / w.sum())
    shift = SCRIPT_SLOPE * (float(np.clip(fav, -SCRIPT_CAP, SCRIPT_CAP)) - float(np.clip(usual_fav, -SCRIPT_CAP, SCRIPT_CAP)))
    run_rate = float(np.clip(base_rate + shift, *RUN_RATE_LIMITS))
    runs = plays * run_rate
    targets = (plays - runs) * float((w * recent["targets"]).sum() / (w * recent["dropbacks"]).sum())
    return {"games": len(recent), "plays": plays, "base_run_rate": base_rate, "run_rate": run_rate, "script_shift": shift,
            "usual_fav": usual_fav, "runs": runs, "targets": targets, "last_kick": recent["kick"].iloc[-1]}


def _calibrate(stat: str, raw: float) -> float:
    base, slope = CALIBRATION[stat]
    return base + slope * raw


def project_player(t: Tables, gsis_id: str, team: str, opponent: str, kick, fav: float, total: float) -> dict | None:
    """Everything the board needs for one player in one game, using only games before `kick`."""
    team_proj = project_team(t, team, kick, fav)
    history = t.player_before(gsis_id, kick)
    history = history[history["team"] == team]
    if team_proj is None or history.empty:
        return None
    recent = history.tail(len(PLAYER_WEIGHTS))
    w = _weights(len(recent), PLAYER_WEIGHTS)
    position = recent["position"].iloc[-1]
    recent_carry = float((w * recent["carries"]).sum() / max((w * recent["team_rush_att"]).sum(), 1e-9))
    recent_target = float((w * recent["targets"]).sum() / max((w * recent["team_targets"]).sum(), 1e-9))
    season_carry = float(history["carries"].sum() / max(history["team_rush_att"].sum(), 1e-9))
    season_target = float(history["targets"].sum() / max(history["team_targets"].sum(), 1e-9))
    carry_share = SHARE_RECENT["carry"] * recent_carry + (1 - SHARE_RECENT["carry"]) * season_carry
    target_share = SHARE_RECENT["target"] * recent_target + (1 - SHARE_RECENT["target"]) * season_target
    snap = recent["offense_pct"].astype(float)
    seen = snap.notna().to_numpy()
    snap_share = float((w[seen] * snap.to_numpy()[seen]).sum() / w[seen].sum()) if seen.any() else None

    # efficiency: his season so far, pulled toward the position average, then blended with the opponent
    lg = t.league
    carries, rush_yards = history["carries"].sum(), history["rushing_yards"].sum()
    tgts, recs, rec_yards = history["targets"].sum(), history["receptions"].sum(), history["receiving_yards"].sum()
    pos_key = position if position in lg["catch"] else "WR"
    own_ypc = (rush_yards + PRIOR_CARRIES * lg["ypc"]["RB"]) / (carries + PRIOR_CARRIES)
    own_ypt = (rec_yards + PRIOR_TARGETS * lg["ypt"][pos_key]) / (tgts + PRIOR_TARGETS)
    catch_rate = (recs + PRIOR_TARGETS * lg["catch"][pos_key]) / (tgts + PRIOR_TARGETS)
    allowed = t.defense_before(opponent, kick)
    to_backs, to_position = allowed[allowed["position"] == "RB"], allowed[allowed["position"] == pos_key]
    opp_ypc = (to_backs["rushing_yards"].sum() / to_backs["carries"].sum()
               if to_backs["carries"].sum() >= MIN_OPPONENT_CARRIES else lg["ypc"]["RB"])
    opp_ypt = (to_position["receiving_yards"].sum() / to_position["targets"].sum()
               if to_position["targets"].sum() >= MIN_OPPONENT_TARGETS else lg["ypt"][pos_key])
    ypc = EFFICIENCY_OWN * own_ypc + EFFICIENCY_OPPONENT * opp_ypc
    ypt = EFFICIENCY_OWN * own_ypt + EFFICIENCY_OPPONENT * opp_ypt

    raw_att = team_proj["runs"] * carry_share
    raw_targets = team_proj["targets"] * target_share
    raw = {"rush_att": raw_att, "rush_yds": raw_att * ypc, "targets": raw_targets, "rec": raw_targets * catch_rate,
           "rec_yds": raw_targets * ypt}

    # scoring chances: red-zone share over his last five games, blended with his overall share of touches
    zone = history.tail(len(RED_ZONE_WEIGHTS))
    wz = _weights(len(zone), RED_ZONE_WEIGHTS)
    rz_share = float((wz * zone["rz_x"]).sum() / max((wz * zone["team_rz_x"]).sum(), 1e-9))
    rz_share_plain = float((wz * zone["rz_n"]).sum() / max((wz * zone["team_rz_n"]).sum(), 1e-9))
    touch_share = float((raw_att + raw_targets) / max(team_proj["runs"] + team_proj["targets"], 1e-9))
    floor, on_red_zone, on_touches = TD_SHARE
    td_share = floor + on_red_zone * rz_share + on_touches * touch_share
    team_tds = implied_team_tds(total / 2 + fav / 2)

    # The same player under the rules exactly as written, kept beside the calibrated numbers so the two can be
    # compared over a season: last-3 shares only, no pull toward the middle, red-zone share alone for touchdowns.
    lit_att, lit_targets = team_proj["runs"] * recent_carry, team_proj["targets"] * recent_target
    literal = {"rush_att": lit_att, "rush_yds": lit_att * ypc, "rec": lit_targets * catch_rate,
               "rec_yds": lit_targets * ypt, "td": rz_share * team_tds}

    means = {stat: _calibrate(stat, raw[stat]) for stat in ("rush_att", "rush_yds", "rec", "rec_yds")}
    # the yards curve for this player: its spread comes partly from his own game log with this team
    cv2 = {"rush_yds": yards_cv2("rush_yds", means["rush_yds"], history["rushing_yards"]),
           "rec_yds": yards_cv2("rec_yds", means["rec_yds"], history["receiving_yards"])}
    typical = {stat: typical_yards(stat, means[stat], cv2[stat]) for stat in cv2}

    missed = int((t.team_before(team, kick)["kick"] > recent["kick"].iloc[-1]).sum())  # team games since he last played
    return {
        "gsis_id": gsis_id, "name": recent["name"].iloc[-1], "team": team, "opponent": opponent, "position": position,
        "games_used": len(recent), "games_missed": missed, "team_proj": team_proj, "season_games": len(history),
        "carry_share": carry_share, "target_share": target_share, "snap_share": snap_share,
        "recent_target_share": recent_target, "season_target_share": season_target,
        "snap_last": float(snap.iloc[-1]) if pd.notna(snap.iloc[-1]) else None,
        "snap_before": float(snap.iloc[:-1].mean()) if len(snap) > 1 and snap.iloc[:-1].notna().any() else None,
        "own_ypc": float(own_ypc), "opp_ypc": float(opp_ypc), "ypc": float(ypc),
        "own_ypt": float(own_ypt), "opp_ypt": float(opp_ypt), "ypt": float(ypt), "catch_rate": float(catch_rate),
        "rz_share": rz_share, "rz_share_plain": rz_share_plain, "touch_share": touch_share, "td_share": td_share,
        "team_tds": team_tds, "raw": raw, "literal": literal, "cv2": cv2, "typical": typical,
        "rush_att": means["rush_att"], "rush_yds": means["rush_yds"], "rec": means["rec"], "rec_yds": means["rec_yds"],
        "td": td_share * team_tds,
    }


# ------------------------------------------------------------ pricing a slate


def _offers(rows: pd.DataFrame) -> dict[str, dict]:
    """{book: {"line", "over", "under", "yes"}} for one player in one market."""
    out: dict[str, dict] = {}
    for row in rows.itertuples():
        offer = out.setdefault(row.book, {"line": None})
        if pd.notna(row.line):
            offer["line"] = float(row.line)
        offer[str(row.side).lower()] = float(row.price)
    return out


def _slate_games(t: Tables, lines: pd.DataFrame) -> list[dict]:
    games = []
    for event_id, rows in lines.groupby("event_id", sort=False):
        first = rows.iloc[0]
        spread, total = rows["home_spread"].mean(), rows["total"].mean()
        if pd.isna(spread) or pd.isna(total):
            continue
        kick = pd.Timestamp(first["commence_time"])
        known = t.games[(t.games["home_team"] == first["home"]) & (t.games["away_team"] == first["away"])
                        & ((t.games["kick"] - kick).abs() < pd.Timedelta(days=3))]
        # Moneylines are not picks. They are carried so a same-game ticket can be anchored on one, priced at
        # the book's own probability with the vig removed (no edge is claimed).
        money: dict[str, dict] = {}
        if "home_ml" in rows.columns:
            for row in rows.itertuples():
                if pd.notna(row.home_ml) and pd.notna(row.away_ml):
                    fair_home, fair_away = devig_two_way(row.home_ml, row.away_ml)
                    book = BOOK_NAME.get(row.book, row.book)
                    money.setdefault(first["home"], {})[book] = {"price": int(row.home_ml), "fair": round(fair_home, 4)}
                    money.setdefault(first["away"], {})[book] = {"price": int(row.away_ml), "fair": round(fair_away, 4)}
        games.append({
            "moneylines": money,
            "event_id": event_id, "game_id": known["game_id"].iloc[0] if len(known) else None,
            "week": int(known["week"].iloc[0]) if len(known) else None, "game": f"{first['away']}@{first['home']}",
            "away": first["away"], "home": first["home"], "kick": kick, "kickoff_utc": kick.isoformat(timespec="minutes"),
            "home_spread": float(spread), "total": float(total),
            "fav": {first["home"]: -float(spread), first["away"]: float(spread)},  # points each team is favoured by
            "implied_points": {first["home"]: float(total) / 2 - float(spread) / 2, first["away"]: float(total) / 2 + float(spread) / 2},
            "blowout": abs(float(spread)) >= BLOWOUT_SPREAD,
            "favorite": first["home"] if spread < 0 else (first["away"] if spread > 0 else None),
        })
    return games


def _status_lookup(status: dict | None) -> tuple[dict, dict, dict]:
    """(by nflverse id, by (name, team), teams) from the status file."""
    if not status:
        return {}, {}, {}
    by_id = {p["gsis_id"]: p for p in status["players"] if p.get("gsis_id")}
    by_name = {(norm_name(p["name"]), p["team"]): p for p in status["players"]}
    return by_id, by_name, status.get("teams", {})


def _availability(entry: dict | None) -> tuple[str | None, str | None]:
    """(reason to drop, flag to show) for one player's status entry, following the injury gate."""
    if not entry:
        return None, None
    state, practice = entry["status"], entry.get("practice") or {}
    if state in ("out", "doubtful", "inactive"):
        return f"is {state}", None
    if state == "questionable":
        if practice.get("none_this_week") is True:
            return "is questionable and has not practised this week", None
        seen = ", ".join(f"{d['level']} {datetime.strptime(d['date'], '%Y-%m-%d').strftime('%a')}" for d in practice.get("seen", []))
        return None, f"questionable ({entry.get('injury') or 'injury not given'}; practice: {seen or 'nothing seen'})"
    return None, None


def price_slate(t: Tables, props: pd.DataFrame, lines: pd.DataFrame, status: dict | None,
                rooting: frozenset[str] | set[str] = frozenset()) -> dict:
    """Turn cached odds into legs, skips and gated players. Nothing here touches the network.

    `rooting` is the set of teams the owner roots for: a leg that wins when one of them does badly is tagged
    "rooting conflict" and otherwise treated like any other leg.
    """
    by_id, by_status_name, status_teams = _status_lookup(status)
    games = _slate_games(t, lines)
    pool = t.players[(t.players["last_season"] == t.season) & ~t.players["status"].isin(NOT_ON_TEAM)]
    legs, skips, gated, notes, projections, priced = [], [], [], [], [], []
    too_small = quarterback_tds = 0

    for game in games:
        rows = props[props["event_id"] == game["event_id"]]
        teams = (game["away"], game["home"])
        roster = pool[pool["latest_team"].isin(teams)].assign(key=lambda d: d["display_name"].map(norm_name))
        if game["game_id"] is None:
            notes.append(f"{game['game']}: could not find this game in the nflverse schedule")

        # vig on the Yes-only touchdown list: scale each book's list to the expected number of scorers
        td_rows = rows[(rows["market"] == "player_anytime_td") & (rows["side"] == "Yes")]
        td_scale = {book: max(1.0, grp["price"].map(american_to_prob).sum() / expected_scorers(game["total"]))
                    for book, grp in td_rows.groupby("book")}

        # who is being priced, and what the model projects for each
        players: dict[str, dict] = {}
        for name in sorted(rows["player"].dropna().unique()):
            if name.endswith(("D/ST", "Defense")):
                continue
            match = roster[roster["key"] == norm_name(name)]
            if len(match) != 1:
                notes.append(f"{game['game']}: {name} " + ("matches no player on either team" if match.empty else "matches more than one player"))
                continue
            who = match.iloc[0]
            team = who["latest_team"]
            opponent = teams[0] if team == teams[1] else teams[1]
            proj = project_player(t, who["gsis_id"], team, opponent, game["kick"], game["fav"][team], game["total"])
            if proj is None:
                notes.append(f"{game['game']}: {name} has no games with {team} this season, so he is not priced")
                continue
            proj["book_name"] = name
            entry = by_id.get(who["gsis_id"]) or by_status_name.get((norm_name(name), team))
            proj["status"] = entry["status"] if entry else "active"
            proj["drop"], proj["status_flag"] = _availability(entry)
            players[who["gsis_id"]] = proj
        # quarterback touchdown props are not priced, so they do not take one of the "popular" places either
        quarterbacks = {p["book_name"] for p in players.values() if p["position"] == "QB"}
        popular_td_rows = td_rows[~td_rows["player"].isin(quarterbacks)]

        # team context: starting QB, and who else on the team has a real role
        team_notes: dict[str, dict] = {}
        for team in teams:
            info = {"drop": None, "flags": [], "roles": []}
            covered = status_teams.get(team)
            if not covered:
                info["flags"].append(f"no status data for {team}")
            else:
                qb = covered["starting_qb"]
                qb_entry = by_id.get(qb.get("gsis_id")) or {"status": qb.get("status", "active"), "practice": {
                    "none_this_week": qb.get("practice_none_this_week"), "seen": []}} if qb.get("status") != "active" else None
                drop, flag = _availability(qb_entry)
                if drop:
                    info["drop"] = f"starting QB {qb['name']} {drop}"
                elif flag:
                    info["flags"].append(f"QB {qb['name']} {flag}")
                if qb.get("same_as_last_game") is False:
                    info["flags"].append(f"QB change: {qb['name']} is listed first, {qb.get('last_game_passer')} played last game")
            last_kick = t.team_before(team, game["kick"])["kick"].max()
            window = t.team_before(team, game["kick"]).tail(len(TEAM_WEIGHTS))["game_id"]
            mates = t.player_games[(t.player_games["team"] == team) & t.player_games["game_id"].isin(window)]["gsis_id"].unique()
            for mate in mates:
                row = pool[pool["gsis_id"] == mate]
                if row.empty or row.iloc[0]["latest_team"] != team:
                    continue  # no longer with the team
                opponent = teams[0] if team == teams[1] else teams[1]
                role = players.get(mate) or project_player(t, mate, team, opponent, game["kick"], game["fav"][team], game["total"])
                if role is None or role["position"] == "QB" or max(
                        role["carry_share"] / ROLE_SHARE["carry"], role["target_share"] / ROLE_SHARE["target"]) < 1:
                    continue  # quarterbacks are handled by the starting-QB check above
                entry = by_id.get(mate)
                info["roles"].append({"gsis_id": mate, "name": role["name"], "position": role["position"],
                                      "carry_share": role["carry_share"], "target_share": role["target_share"],
                                      "missed": role["games_missed"], "status": entry["status"] if entry else "active"})
            info["last_kick"] = last_kick
            team_notes[team] = info

        for gsis_id, proj in players.items():
            team, info = proj["team"], team_notes[proj["team"]]
            projections.append({"game": game["game"], "typical_rush_yds": proj["typical"]["rush_yds"],
                                "typical_rec_yds": proj["typical"]["rec_yds"], **{k: proj[k] for k in (
                                    "name", "team", "position", "games_used", "carry_share", "target_share", "snap_share",
                                    "rush_att", "rush_yds", "rec", "rec_yds", "td", "status")}})
            player_rows = rows[rows["player"] == proj["book_name"]]
            starter = (proj["snap_share"] or 0) >= STARTER_SNAP_SHARE
            for market, spec in MARKETS.items():
                market_rows = player_rows[player_rows["market"] == market]
                if market_rows.empty or proj["position"] not in spec["positions"]:
                    if not market_rows.empty and spec["stat"] == "td" and proj["position"] == "QB":
                        quarterback_tds += 1
                    continue
                stat, kind = spec["stat"], spec["kind"]
                mean = proj[stat]
                spread = proj["cv2"].get(stat)  # the fitted yards spread; None for counts and touchdowns
                offers = _offers(market_rows)
                if ((stat in ("rush_att", "rush_yds") and proj["raw"]["rush_att"] < MIN_ROLE["carries"])
                        or (stat in ("rec", "rec_yds") and proj["raw"]["targets"] < MIN_ROLE["targets"])
                        or (stat == "td" and (proj["snap_share"] or 0.0) < PART_TIME_SNAPS)):
                    too_small += 1
                    continue

                # every (book, side) the player can be bet at, with the model's and the book's probability
                options = []
                for book, offer in offers.items():
                    if kind == "td":
                        if "yes" in offer:
                            fair = american_to_prob(offer["yes"]) / td_scale.get(book, 1.0)
                            options.append(("yes", book, None, offer["yes"], model_probability(stat, None, mean)[0], fair))
                        continue
                    if offer["line"] is None or "over" not in offer or "under" not in offer:
                        continue
                    fair_over, fair_under = devig_two_way(offer["over"], offer["under"])
                    p_over, p_under = model_probability(stat, offer["line"], mean, spread)
                    options.append(("over", book, offer["line"], offer["over"], p_over, fair_over))
                    options.append(("under", book, offer["line"], offer["under"], p_under, fair_under))
                if not options:
                    continue
                side, book, line, price, model_p, fair_p = max(options, key=lambda o: (o[4] - o[5], o[3]))
                edge = model_p - fair_p
                literal_p = literal_probability(stat, line, proj["literal"][stat])[1 if side == "under" else 0]
                priced.append({"id": f"{gsis_id}|{market}|{side}", "player_id": gsis_id, "player": proj["name"], "team": team,
                               "market_key": market, "market": spec["label"], "side": side, "line": line,
                               "book": BOOK_NAME.get(book, book), "price": int(price), "edge": round(edge, 4)})

                # The calibrated model and the rules as written must point the same way for a leg to be trusted
                # fully: the same side of 50% for an over/under, and both above the book's price for a touchdown.
                agrees = literal_agrees(stat, model_p, literal_p, fair_p)
                flags = list(info["flags"])
                if not agrees:
                    flags.append(FLIP_FLAG)
                if proj["status_flag"]:
                    flags.append(proj["status_flag"])
                book_lines = {b: o["line"] for b, o in offers.items() if o["line"] is not None}
                if kind != "td" and len(book_lines) == 2 and abs(book_lines["draftkings"] - book_lines["fanduel"]) > LINE_GAP[kind]:
                    flags.append(f"books differ: DK {book_lines['draftkings']:g}, FD {book_lines['fanduel']:g}")
                if proj["games_used"] < len(PLAYER_WEIGHTS):
                    flags.append(f"only {proj['games_used']} game{'s' if proj['games_used'] != 1 else ''} of history with {team}")
                if proj["games_missed"]:
                    flags.append(f"did not play in {team}'s last {proj['games_missed']} game{'s' if proj['games_missed'] != 1 else ''}")
                if proj["snap_last"] is not None and proj["snap_before"] is not None and abs(proj["snap_last"] - proj["snap_before"]) >= SNAP_JUMP:
                    flags.append(f"snap share moved {proj['snap_before']:.0%} to {proj['snap_last']:.0%}")
                if side == "over" and proj["snap_share"] is not None and proj["snap_share"] < PART_TIME_SNAPS:
                    flags.append(f"part-time role ({proj['snap_share']:.0%} of snaps)")
                # Shares come from the last three games. If a teammate with a real role sat out or came back
                # inside that window, those shares describe a different line-up. Back-testing showed that
                # re-dividing his share by formula makes projections worse, so the leg is flagged instead.
                share_kind = "carry" if stat in ("rush_att", "rush_yds") or (stat == "td" and proj["position"] == "RB") else "target"
                window = len(PLAYER_WEIGHTS)
                for mate in info["roles"]:
                    if mate["gsis_id"] == gsis_id or mate[f"{share_kind}_share"] < ROLE_SHARE[share_kind]:
                        continue
                    gone = mate["status"] in ("out", "doubtful", "inactive")
                    if gone and mate["missed"] < window:
                        flags.append(f"role change: {mate['name']} is {mate['status']} but played in {window - mate['missed']} of the last {window} games")
                    elif not gone and mate["missed"] > 0:
                        flags.append(f"role change: {mate['name']} missed the last {mate['missed']} and may return")

                is_favorite = game["favorite"] == team
                # Legs share a game script when the same team being ahead helps all of them: that team's backs
                # run more and its players score, it throws less, and the other side runs less and throws more.
                primary = max((m for m in info["roles"] if m["position"] in ("WR", "TE")), key=lambda m: m["target_share"], default=None)
                other = proj["opponent"]
                if stat == "td":
                    ahead = team
                    role = "rb_td" if proj["position"] == "RB" else (
                        "primary_target_td" if primary and primary["gsis_id"] == gsis_id else "td")
                elif stat in ("rush_att", "rush_yds"):
                    ahead, role = (team, "rb_volume") if side == "over" else (other, "rb_volume_under")
                else:
                    ahead, role = (other, "target_volume") if side == "over" else (team, "target_volume_under")
                group = f"{game['game']}: {ahead} ahead"
                conflict = side == "under" and team in rooting  # wins when a team the owner roots for does badly

                shown = proj["typical"][stat] if kind == "yards" else mean
                leg = {
                    "id": f"{gsis_id}|{market}|{side}", "game_id": game["game_id"], "game": game["game"],
                    "kickoff_utc": game["kickoff_utc"], "player": proj["name"], "player_id": gsis_id,
                    "position": proj["position"], "team": team, "opponent": proj["opponent"], "market": spec["label"],
                    "market_key": market, "side": side, "line": line, "book": BOOK_NAME.get(book, book), "price": int(price),
                    "books": {BOOK_NAME.get(b, b): o for b, o in offers.items()}, "projection": round(shown, 2),
                    "average": round(mean, 2), "model_prob": round(model_p, 4), "implied_prob": round(fair_p, 4),
                    "edge": round(edge, 4), "literal_prob": round(literal_p, 4), "literal_agrees": agrees, "tier": None,
                    "flags": flags,
                    "tags": ["rooting conflict"] if conflict else [], "rooting_conflict": conflict,
                    "status": proj["status"], "correlation_group": group, "correlation_role": role, "starter": starter,
                    # this side at every book that offers it, so a same-game ticket can be priced at one book
                    "offers": [{"book": BOOK_NAME.get(b, b), "line": ln, "price": int(pr), "model_prob": round(mp, 4),
                                "implied_prob": round(fp, 4), "edge": round(mp - fp, 4)}
                               for s, b, ln, pr, mp, fp in options if s == side],
                    "reasoning": _reasoning(stat, side, line, proj, game),
                }
                drop = proj["drop"] and f"{proj['name']} {proj['drop']}" or info["drop"]
                popular = _is_popular(stat, side, proj, players, popular_td_rows)
                if drop:
                    gated.append({"id": leg["id"], "player": proj["name"], "team": team, "market": spec["label"], "reason": drop})
                    if popular:
                        skips.append({**_brief(leg), "kind": "injury", "reason": f"Injury gate: {drop}."})
                    continue
                if edge >= BIG_EDGE:
                    skips.append({**_brief(leg), "kind": "not trusted", "reason": (
                        f"Model edge of {edge:.0%} is too large to trust; the book probably knows something the model does not."
                        + (f" {'; '.join(flags)}." if flags else ""))})
                    continue
                if kind == "yards" and game["blowout"] and is_favorite and starter:
                    why = (f"Blowout rule: {team} are favoured by {abs(game['home_spread']):g}, so starters may sit late and "
                           "the main yardage line is not used")
                    alt = _alt_line(leg, player_rows, market, offers, mean, stat, proj["literal"][stat], spread)
                    if alt and MIN_EDGE <= alt["edge"] < BIG_EDGE:
                        alt["flags"] = ([flag for flag in flags if flag != FLIP_FLAG]
                                        + ([] if alt["literal_agrees"] else [FLIP_FLAG])
                                        + ["blowout: lower alt line in place of the main line"])
                        alt["reasoning"] = f"{why}. {alt['reasoning']} The first-half version is the other option if your book has it."
                        alt["tier"] = "B"
                        legs.append(alt)
                    elif edge >= MIN_EDGE or popular:
                        skips.append({**_brief(leg), "kind": "blowout", "reason": (
                            f"{why}. No alt line with a 5% edge was priced; the first-half version is the other option if "
                            "your book has it.")})
                    continue
                if edge >= MIN_EDGE:
                    leg["tier"] = "A" if edge >= A_TIER_EDGE and not flags else "B"
                    legs.append(leg)
                elif popular:
                    worst = min((o for o in options if o[0] in ("over", "yes")), key=lambda o: o[4] - o[5], default=None)
                    if worst and worst[4] - worst[5] <= SKIP_EDGE:
                        popular_leg = {**leg, "side": worst[0], "line": worst[2], "book": BOOK_NAME.get(worst[1], worst[1]),
                                       "price": int(worst[3]), "model_prob": round(worst[4], 4), "implied_prob": round(worst[5], 4),
                                       "edge": round(worst[4] - worst[5], 4)}
                        skips.append({**_brief(popular_leg), "kind": "popular", "reason": _skip_reason(stat, popular_leg, shown)})

    legs.sort(key=lambda leg: (leg["tier"], -leg["edge"]))
    order = {"popular": 0, "injury": 1, "blowout": 2, "not trusted": 3}
    skips.sort(key=lambda s: (order[s["kind"]], s["edge"] if s["kind"] != "not trusted" else -s["edge"]))
    if quarterback_tds:
        notes.append(f"{quarterback_tds} quarterback anytime TD props were not priced: the touch-share method does "
                     "not describe how a quarterback scores")
    if too_small:
        notes.append(f"{too_small} posted props were not priced because the player's role is too small to trust "
                     f"(under {MIN_ROLE['carries']:g} carries or {MIN_ROLE['targets']:g} targets a game, or under "
                     f"{PART_TIME_SNAPS:.0%} of snaps for a touchdown)")
    return {
        "generated_utc": utc_now().isoformat(timespec="seconds"), "season": t.season,
        "games": [{k: v for k, v in g.items() if k != "kick"} for g in games],
        "legs": legs, "skips": skips, "gated": gated, "notes": sorted(set(notes)), "projections": projections,
        "priced": priced,  # every prop the model put a number on, leg or not: board.py uses it to explain changes
    }


def _brief(leg: dict) -> dict:
    keep = ("id", "game", "player", "team", "position", "market", "side", "line", "book", "price", "projection",
            "model_prob", "implied_prob", "edge")
    return {k: leg[k] for k in keep}


def _is_popular(stat: str, side: str, proj: dict, players: dict[str, dict], td_rows: pd.DataFrame) -> bool:
    """Popular props are the ones most people bet: the shortest touchdown prices and the lead players' overs."""
    mates = [p for p in players.values() if p["team"] == proj["team"]]
    if stat == "td":
        shortest = td_rows.groupby("player")["price"].min().nsmallest(3).index
        return proj["book_name"] in shortest
    if stat in ("rush_att", "rush_yds"):
        backs = [p for p in mates if p["position"] == "RB"]
        return bool(backs) and max(backs, key=lambda p: p["rush_att"])["gsis_id"] == proj["gsis_id"]
    catchers = sorted((p for p in mates if p["position"] in ("WR", "TE")), key=lambda p: -p["rec_yds"])[:2]
    return proj["gsis_id"] in [p["gsis_id"] for p in catchers]


def _skip_reason(stat: str, leg: dict, shown: float) -> str:
    if stat == "td":
        return f"Model gives {leg['model_prob']:.0%} to score, the price implies {leg['implied_prob']:.0%} after vig."
    return (f"Model gives the over {leg['model_prob']:.0%}, the price implies {leg['implied_prob']:.0%} after vig: "
            f"a typical game is about {shown:.1f} against a line of {leg['line']:g}.")


def _alt_line(leg: dict, player_rows: pd.DataFrame, market: str, offers: dict, mean: float, stat: str,
              literal_mean: float, spread: float | None) -> dict | None:
    """Blowout rule: the best lower alt Over, with the main market's margin taken off its one-sided price."""
    alts = player_rows[(player_rows["market"] == ALT_OF.get(market)) & (player_rows["side"] == "Over")
                       & (player_rows["price"] >= ALT_PRICE_FLOOR)]
    main_line = max((o["line"] for o in offers.values() if o["line"] is not None), default=leg["line"])
    alts = alts[alts["line"] < main_line]
    best = None
    for row in alts.itertuples():
        main = offers.get(row.book, {})
        hold = (american_to_prob(main["over"]) + american_to_prob(main["under"]) - 1.0) if "over" in main and "under" in main else DEFAULT_HOLD
        fair = american_to_prob(row.price) / (1.0 + hold)
        model_p = model_probability(stat, float(row.line), mean, spread)[0]
        if best is None or model_p - fair > best["edge"]:
            book = BOOK_NAME.get(row.book, row.book)
            literal_p = literal_probability(stat, float(row.line), literal_mean)[0]
            # an alt over is an over, so it never profits from the player's own team doing badly
            ahead = leg["team"] if stat == "rush_yds" else leg["opponent"]
            best = {**leg, "id": leg["id"].rsplit("|", 1)[0] + "|over|alt", "side": "over", "tags": [],
                    "rooting_conflict": False, "correlation_group": f"{leg['game']}: {ahead} ahead",
                    "correlation_role": "rb_volume" if stat == "rush_yds" else "target_volume",
                    "market": leg["market"] + " (alt line)", "line": float(row.line), "book": book, "price": int(row.price),
                    "model_prob": round(model_p, 4), "implied_prob": round(fair, 4), "edge": round(model_p - fair, 4),
                    "literal_prob": round(literal_p, 4), "literal_agrees": literal_agrees(stat, model_p, literal_p, fair),
                    "offers": [{"book": book, "line": float(row.line), "price": int(row.price), "model_prob": round(model_p, 4),
                                "implied_prob": round(fair, 4), "edge": round(model_p - fair, 4)}],
                    "reasoning": f"Alt line {float(row.line):g} at {int(row.price):+d}: model {model_p:.0%} against {fair:.0%} implied."}
    return best


def _reasoning(stat: str, side: str, line: float | None, p: dict, game: dict) -> str:
    tp, team = p["team_proj"], p["team"]
    fav = game["fav"][team]
    role = "favourites" if fav > 0 else "underdogs"
    usual = f"{abs(tp['usual_fav']):.1f}-point {'favourites' if tp['usual_fav'] > 0 else 'underdogs'}"
    script = "" if abs(tp["script_shift"]) < 0.01 else (
        f" Their run rate was {tp['base_run_rate']:.0%} over five games as {usual} on average; as {abs(fav):g}-point "
        f"{role} it is set at {tp['run_rate']:.0%}.")
    n = p["games_used"]
    if stat == "rush_att":
        return (f"Projects {p['rush_att']:.1f} carries against {line:g}: {team} run about {tp['runs']:.0f} times and he has "
                f"taken {p['carry_share']:.0%} of them over his last {n} games.{script}")
    if stat == "rush_yds":
        return (f"A typical game is about {p['typical']['rush_yds']:.0f} yards against {line:g}: {p['rush_att']:.1f} carries "
                f"at {p['ypc']:.1f} a carry (his {p['own_ypc']:.1f} blended with the {p['opp_ypc']:.1f} {p['opponent']} allow).{script}")
    if stat == "rec":
        return (f"Projects {p['rec']:.1f} catches against {line:g}: {p['target_share']:.0%} of about {tp['targets']:.0f} team "
                f"targets ({p['season_target_share']:.0%} this season, {p['recent_target_share']:.0%} over his last {n}), at a "
                f"{p['catch_rate']:.0%} catch rate.{script}")
    if stat == "rec_yds":
        return (f"A typical game is about {p['typical']['rec_yds']:.0f} yards against {line:g}: {p['target_share']:.0%} of about "
                f"{tp['targets']:.0f} team targets ({p['season_target_share']:.0%} this season, {p['recent_target_share']:.0%} over "
                f"his last {n}) at {p['ypt']:.1f} a target (his {p['own_ypt']:.1f} blended with the {p['opp_ypt']:.1f} "
                f"{p['opponent']} allow).{script}")
    return (f"Model gives {1 - math.exp(-p['td']):.0%} to score: {p['rz_share']:.0%} of {team}'s red-zone chances and "
            f"{p['touch_share']:.0%} of their touches, with {team} priced for {p['team_tds']:.1f} touchdowns.")


# ------------------------------------------------------------------ backtest


def backtest(t: Tables, first_week: int = 5) -> pd.DataFrame:
    """Project every past player-game from `first_week` on, using only earlier games, and line it up with what happened."""
    rows = []
    for game in t.team_games[t.team_games["week"] >= first_week].itertuples():
        played = t.player_games[(t.player_games["game_id"] == game.game_id) & (t.player_games["team"] == game.team)]
        for actual in played.itertuples():
            proj = project_player(t, actual.gsis_id, game.team, game.opponent, game.kick, game.fav, game.total_line)
            if proj is None or proj["games_missed"]:
                continue  # the board only prices players who are in their current role
            rows.append({
                "week": game.week, "position": actual.position, "raw_targets": proj["raw"]["targets"],
                "raw_rush_att": proj["raw"]["rush_att"], "raw_rush_yds": proj["raw"]["rush_yds"], "raw_rec": proj["raw"]["rec"],
                "raw_rec_yds": proj["raw"]["rec_yds"], "p_rush_att": proj["rush_att"], "p_rush_yds": proj["rush_yds"],
                "p_rec": proj["rec"], "p_rec_yds": proj["rec_yds"], "p_td": proj["td"],
                "cv2_rush_yds": proj["cv2"]["rush_yds"], "cv2_rec_yds": proj["cv2"]["rec_yds"],
                "td_red_zone_only": proj["rz_share"] * proj["team_tds"],
                "a_rush_att": actual.carries, "a_rush_yds": actual.rushing_yards, "a_rec": actual.receptions,
                "a_rec_yds": actual.receiving_yards, "a_td": float(actual.rushing_tds + actual.receiving_tds > 0),
            })
    return pd.DataFrame(rows)


def print_backtest(t: Tables) -> None:
    bt = backtest(t)
    backs = (bt["position"] == "RB") & (bt["raw_rush_att"] >= 6)
    checks = [("rush attempts", "rush_att", backs, "RB, 6+ carries"), ("rush yards", "rush_yds", backs, "RB, 6+ carries"),
              ("receptions", "rec", bt["position"].isin(["RB", "WR", "TE"]) & (bt["raw_targets"] >= 3), "3+ targets"),
              ("receiving yards", "rec_yds", bt["position"].isin(["WR", "TE"]) & (bt["raw_targets"] >= 3), "WR/TE, 3+ targets")]
    print(f"\nBACKTEST | season {t.season} | weeks {int(bt['week'].min())}-{int(bt['week'].max())} | {len(bt):,} player-games, "
          "each projected from earlier games only")
    print("\n1. are the projections centred? (players with a real role)")
    print(f"  {'stat':<17}{'who':<19}{'n':>5}{'raw rules':>11}{'calibrated':>12}{'actual':>8}{'avg miss':>10}")
    for label, stat, mask, who in checks:
        s = bt[mask]
        print(f"  {label:<17}{who:<19}{len(s):>5}{s[f'raw_{stat}'].mean():>11.2f}{s[f'p_{stat}'].mean():>12.2f}"
              f"{s[f'a_{stat}'].mean():>8.2f}{(s[f'a_{stat}'] - s[f'p_{stat}']).abs().mean():>10.2f}")
    print("\n2. are the probabilities honest? a line set below, at and above each player's projection:")
    print(f"  {'stat':<17}{'line vs projection':<20}{'over happened':>14}{'model said':>12}{'literal rules said':>20}")
    for label, stat, mask, _ in checks:
        s = bt[mask]
        spreads = s[f"cv2_{stat}"] if f"cv2_{stat}" in s else [None] * len(s)
        for offset in (-0.25, 0.0, 0.25):
            lines = np.floor(s[f"p_{stat}"] * (1 + offset)) + 0.5
            said = np.mean([model_probability(stat, line, mean, cv2)[0]
                            for line, mean, cv2 in zip(lines, s[f"p_{stat}"], spreads)])
            if stat in DISPERSION:
                literal = np.mean([count_over_under(line, max(mean, 1e-6))[0] for line, mean in zip(lines, s[f"raw_{stat}"])])
            else:
                literal = np.mean([normal_over_under(line, mean, SIGMA[stat][0] + SIGMA[stat][1] * mean)[0]
                                   for line, mean in zip(lines, s[f"raw_{stat}"])])
            print(f"  {label:<17}{f'{offset:+.0%}':<20}{(s[f'a_{stat}'] > lines).mean():>14.1%}{said:>12.1%}{literal:>20.1%}")
    print("\n3. yards: the right-skewed gamma curve and the normal curve, on the same projections")
    offsets = (-0.5, -0.25, 0.0, 0.25, 0.5)
    for label, stat, mask, _ in checks:
        if stat not in SIGMA:
            continue
        s = bt[mask]
        means, spreads, actual = s[f"p_{stat}"].to_numpy(), s[f"cv2_{stat}"].to_numpy(), s[f"a_{stat}"].to_numpy()
        table = {"happened": [], "gamma": [], "normal": []}
        miss = {"gamma": [], "normal": []}
        for offset in offsets:
            lines = np.floor(means * (1 + offset)) + 0.5
            over = actual > lines
            said = {"gamma": np.array([gamma_over_under(line, max(mean, 1e-6), cv2)[0] for line, mean, cv2 in zip(lines, means, spreads)]),
                    "normal": np.array([normal_yards_over_under(stat, line, mean)[0] for line, mean in zip(lines, means)])}
            table["happened"].append(over.mean())
            for curve in said:
                table[curve].append(said[curve].mean())
                miss[curve].append(((said[curve] - over) ** 2).mean())
        print(f"  {label} (n={len(s)}): chance of the over at a line set this far from the projection")
        print(f"    {'':<22}" + "".join(f"{f'{offset:+.0%}':>9}" for offset in offsets) + f"{'Brier score':>14}")
        print(f"    {'over happened':<22}" + "".join(f"{value:>9.1%}" for value in table["happened"]))
        for curve in ("gamma", "normal"):
            name = f"{curve} ({'in use' if YARDS_CURVE[stat] == curve else 'not in use'})"
            print(f"    {name:<22}" + "".join(f"{value:>9.1%}" for value in table[curve]) + f"{np.mean(miss[curve]):>14.4f}")
        for position, offset in ((3, 0.25), (4, 0.5)):
            print(f"    under at a line {offset:.0%} above the projection: happened {1 - table['happened'][position]:.1%} | "
                  f"gamma says {1 - table['gamma'][position]:.1%} | normal says {1 - table['normal'][position]:.1%}")
    print("  A lower Brier score is better. Under legs sit at lines above the projection, so the last two rows of each")
    print("  block show what the choice of curve does to under edges.")
    print("\n4. anytime TD: how often players the model put in each range actually scored (RB, WR and TE; quarterbacks are not priced)")
    td = bt[bt["position"].isin(MARKETS["player_anytime_td"]["positions"])].copy()
    td["said"], td["literal"] = 1 - np.exp(-td["p_td"]), 1 - np.exp(-td["td_red_zone_only"])
    print(f"  {'model range':<14}{'n':>5}{'model said':>12}{'scored':>9}   |  {'red-zone share only':<21}{'n':>5}{'said':>8}{'scored':>9}")
    edges = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    for lo, hi in zip(edges, edges[1:]):
        a, b = td[(td["said"] > lo) & (td["said"] <= hi)], td[(td["literal"] > lo) & (td["literal"] <= hi)]
        left = f"{len(a):>5}{a['said'].mean():>12.1%}{a['a_td'].mean():>9.1%}" if len(a) else f"{0:>5}{'-':>12}{'-':>9}"
        right = f"{len(b):>5}{b['literal'].mean():>8.1%}{b['a_td'].mean():>9.1%}" if len(b) else f"{0:>5}{'-':>8}{'-':>9}"
        print(f"  {f'{lo:.0%}-{hi:.0%}':<14}{left}   |  {'':<21}{right}")
    print("\n5. tested and not adopted: taking shares only from the games since a long-term absence began")
    for line in absence_window_check(t):
        print(f"  {line}")
    print("\nThe calibration constants were fitted on these same weeks, so this is a fit check, not a forecast test.")


def absence_window_check(t: Tables, first_week: int = 5) -> list[str]:
    """Back-test of an idea that is NOT in the model, kept so it can be re-checked as the season goes on.

    The idea: when a teammate with a real role has been out for at least one game and is still out, take
    everyone else's share only from the games since his absence began (the latest absence if several),
    instead of the model's shares. Each is scored against the share the player actually got.
    """
    rows = []
    for game in t.team_games[t.team_games["week"] >= first_week].itertuples():
        prior = t.player_games[(t.player_games["team"] == game.team) & (t.player_games["kick"] < game.kick)]
        if prior.empty:
            continue
        last_team_kick = prior["kick"].max()
        now = t.player_games[(t.player_games["game_id"] == game.game_id) & (t.player_games["team"] == game.team)]
        per = prior.groupby("gsis_id").agg(position=("position", "last"), carries=("carries", "sum"),
                                           runs=("team_rush_att", "sum"), targets=("targets", "sum"),
                                           throws=("team_targets", "sum"), last=("kick", "max"))
        absent = per[~per.index.isin(now["gsis_id"]) & (per["last"] < last_team_kick)]
        starts = {
            "carry": absent[(absent["position"] == "RB") & (absent["carries"] / absent["runs"] >= ROLE_SHARE["carry"])]["last"].max(),
            "target": absent[absent["targets"] / absent["throws"] >= ROLE_SHARE["target"]]["last"].max(),
        }
        if pd.isna(starts["carry"]) and pd.isna(starts["target"]):
            continue
        for actual in now.itertuples():
            proj = project_player(t, actual.gsis_id, game.team, game.opponent, game.kick, game.fav, game.total_line)
            if proj is None or proj["games_missed"]:
                continue
            mine = prior[prior["gsis_id"] == actual.gsis_id]
            for kind, stat, total, team_now, positions, floor in (
                    ("carry", "carries", "team_rush_att", game.rush_att, ("RB",), 0.08),
                    ("target", "targets", "team_targets", game.targets, ("RB", "WR", "TE"), 0.05)):
                if pd.isna(starts[kind]) or actual.position not in positions or proj[f"{kind}_share"] < floor:
                    continue
                since = mine[mine["kick"] > starts[kind]]
                if since.empty or not since[total].sum():
                    continue
                rows.append({"kind": kind, "model": proj[f"{kind}_share"], "since": since[stat].sum() / since[total].sum(),
                             "actual": getattr(actual, stat) / max(team_now, 1), "games": len(since)})
    found = pd.DataFrame(rows)
    lines = []
    for kind in ("target", "carry"):
        s = found[found["kind"] == kind] if len(found) else found
        if len(s) < 20:
            lines.append(f"{kind} share: too few cases to judge yet ({len(s)})")
            continue
        miss_model, miss_since = (s["model"] - s["actual"]).abs().mean(), (s["since"] - s["actual"]).abs().mean()
        verdict = "the model's shares are closer" if miss_model <= miss_since else "the since-absence window is closer"
        lines.append(f"{kind + ' share':<13} n={len(s):>4}  average miss: model's shares {miss_model:.3f}, since-absence window "
                     f"{miss_since:.3f}  ->  {verdict}")
    return lines


# ----------------------------------------------------------------------- CLI


def print_slate(result: dict) -> None:
    print(f"\nMODEL | {az(utc_now())} | season {result['season']} | {len(result['games'])} game(s) priced from cached odds")
    for game in result["games"]:
        kick = datetime.fromisoformat(game["kickoff_utc"])
        points = ", ".join(f"{team} {pts:.1f}" for team, pts in game["implied_points"].items())
        print(f"\n{game['game']} | {az(kick)} | {game['home']} {game['home_spread']:+g}, total {game['total']:g} | implied points: {points}"
              + (" | BLOWOUT RULE ON" if game["blowout"] else ""))
        table = [p for p in result["projections"] if p["game"] == game["game"]]
        print(f"  {'player':<22}{'pos':<4}{'team':<5}{'gms':>3}{'carry%':>8}{'tgt%':>7}{'snap%':>7}{'carries':>9}{'rush yd':>9}"
              f"{'catches':>9}{'rec yd':>8}{'TD%':>6}  status")
        for p in sorted(table, key=lambda p: (p["team"], -(p["rush_att"] + p["rec"]))):
            snap = f"{p['snap_share']:.0%}" if p["snap_share"] is not None else "-"
            print(f"  {p['name'][:21]:<22}{p['position']:<4}{p['team']:<5}{p['games_used']:>3}{p['carry_share']:>8.0%}"
                  f"{p['target_share']:>7.0%}{snap:>7}{p['rush_att']:>9.1f}{p['typical_rush_yds']:>9.0f}{p['rec']:>9.1f}"
                  f"{p['typical_rec_yds']:>8.0f}{1 - math.exp(-p['td']):>6.0%}  {p['status']}")
    print(f"\nLEGS with an edge of {MIN_EDGE:.0%} or more: {len(result['legs'])}")
    for leg in result["legs"]:
        line = "" if leg["line"] is None else f" {leg['line']:g}"
        print(f"  [{leg['tier']}] {leg['player']} ({leg['team']} {leg['position']}) {leg['side']}{line} {leg['market']} "
              f"{leg['book']} {leg['price']:+d} | model {leg['model_prob']:.1%} vs {leg['implied_prob']:.1%} = edge {leg['edge']:+.1%}"
              f" | literal {leg['literal_prob']:.1%}")
        print(f"        {leg['reasoning']}")
        if leg["flags"]:
            print(f"        flags: {'; '.join(leg['flags'])}")
        if leg["tags"]:
            print(f"        tags: {'; '.join(leg['tags'])}")
        print(f"        script: {leg['correlation_group']} [{leg['correlation_role']}]")
    print(f"\nSKIPS: {len(result['skips'])}")
    for skip in result["skips"]:
        line = "" if skip["line"] is None else f" {skip['line']:g}"
        print(f"  [{skip['kind']}] {skip['player']} ({skip['team']}) {skip['side']}{line} {skip['market']} {skip['book']} "
              f"{skip['price']:+d}: {skip['reason']}")
    print(f"\nDROPPED BY THE INJURY GATE: {len(result['gated'])}")
    for reason in sorted({g["reason"] for g in result["gated"]}):
        print(f"  {reason}")
    if result["notes"]:
        print("\nnotes:")
        for note in result["notes"]:
            print(f"  - {note}")


def run() -> dict:
    """Price the cached odds with the cached status. Used by board.py."""
    try:
        status = fetch_status.load()
    except FileNotFoundError:
        status = None
    return price_slate(load_tables(), fetch_odds.props(), fetch_odds.game_lines(), status,
                       rooting=frozenset(load_config()["rooting_teams"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Price the cached slate, or back-test the projections.")
    parser.add_argument("--backtest", action="store_true", help="check the projections against past weeks")
    parser.add_argument("--literal", action="store_true", help="use the rules exactly as written, with no calibration")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.literal:
        use_literal_rules()
        print("LITERAL RULES: calibration off (last-3 shares only, pure Poisson, normal centred on the average, red-zone share only)")
    if args.backtest:
        print_backtest(load_tables())
        return 0
    print_slate(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
