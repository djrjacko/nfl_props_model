"""Download the nflverse files the prop model needs, and report their shape.

nflverse publishes its data as parquet files attached to GitHub releases of
github.com/nflverse/nflverse-data. This script reads those files directly.
(nfl_data_py is not used: it is archived, pins numpy<2 and pandas<2 so it does
not install on current Python, and its weekly stats still point at the
`player_stats` release that nflverse retired on 2025-08-01.)

Run it by hand:

    python src/fetch_nflverse.py                    # current season
    python src/fetch_nflverse.py --seasons 2025 2026
    python src/fetch_nflverse.py --force            # re-download even if unchanged
    python src/fetch_nflverse.py --no-report        # download only

Other scripts read the files through load(), e.g. load("pbp").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
import requests

from common import ARIZONA, ROOT, az

DATA_DIR = ROOT / "data" / "nflverse"
MANIFEST_PATH = DATA_DIR / "_manifest.json"
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"

EASTERN = "America/New_York"  # nflverse schedule times (gameday/gametime) are US Eastern

# The model uses "last 5 games" windows. Until the current season has this many
# completed weeks, last season's files are downloaded too so the windows are full.
MIN_WEEKS_BEFORE_DROPPING_LAST_SEASON = 5


@dataclass(frozen=True)
class Dataset:
    key: str  # name the rest of the code uses, e.g. load("pbp")
    tag: str  # nflverse-data release tag
    filename: str  # "{season}" is filled in for per-season files
    history: bool  # usage data: also keep last season early in the year
    required: tuple[str, ...]  # columns the model relies on; the run fails if one disappears
    what: str

    @property
    def per_season(self) -> bool:
        return "{season}" in self.filename

    def file_for(self, season: int | None = None) -> str:
        return self.filename.format(season=season)


# Every column name below was read from the live 2026 files on 2026-11-24.
DATASETS: tuple[Dataset, ...] = (
    Dataset(
        key="schedules",
        tag="schedules",
        filename="games.parquet",
        history=False,
        required=(
            "game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "home_team",
            "away_score", "home_score", "spread_line", "total_line", "espn",
            "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name",
        ),
        what="one row per game: kickoff (Eastern), teams, final score, ESPN game id, listed starting QBs",
    ),
    Dataset(
        key="player_week",
        tag="stats_player",
        filename="stats_player_week_{season}.parquet",
        history=True,
        required=(
            "player_id", "player_display_name", "position", "position_group", "season", "week", "season_type",
            "game_id", "team", "opponent_team", "attempts", "completions", "passing_yards",
            "carries", "rushing_yards", "rushing_tds",
            "targets", "receptions", "receiving_yards", "receiving_tds", "target_share",
        ),
        what="one row per player per game: carries, targets, receptions, yards, touchdowns",
    ),
    Dataset(
        key="pbp",
        tag="pbp",
        filename="play_by_play_{season}.parquet",
        history=True,
        required=(
            "game_id", "season", "week", "season_type", "home_team", "away_team", "posteam", "defteam",
            "play_type", "rush_attempt", "pass_attempt", "complete_pass", "sack", "qb_scramble", "qb_kneel",
            "qb_spike", "two_point_attempt", "rusher_player_id", "receiver_player_id", "passer_player_id",
            "td_player_id", "touchdown", "rush_touchdown", "pass_touchdown", "yardline_100", "down", "qtr",
            "game_half", "yards_gained", "rushing_yards", "receiving_yards", "score_differential",
        ),
        what="one row per play: who ran or was targeted, field position (red zone), half, yards, touchdowns",
    ),
    Dataset(
        key="snap_counts",
        tag="snap_counts",
        filename="snap_counts_{season}.parquet",
        history=True,
        required=(
            "game_id", "season", "game_type", "week", "player", "pfr_player_id", "position", "team",
            "opponent", "offense_snaps", "offense_pct",
        ),
        what="one row per player per game: offensive snaps and snap share (keyed by PFR id, not GSIS id)",
    ),
    Dataset(
        key="depth_charts",
        tag="depth_charts",
        filename="depth_charts_{season}.parquet",
        history=False,
        required=("dt", "team", "player_name", "espn_id", "gsis_id", "pos_abb", "pos_slot", "pos_rank"),
        what="daily depth chart snapshots (ESPN-sourced): one row per team, position slot and rank",
    ),
    Dataset(
        key="injuries",
        tag="injuries",
        filename="injuries_{season}.parquet",
        history=False,
        required=(
            "season", "week", "team", "gsis_id", "full_name", "position",
            "report_primary_injury", "report_status", "practice_status",
        ),
        what="one row per listed player per week: game status and that week's last practice status",
    ),
    Dataset(
        key="players",
        tag="players",
        filename="players.parquet",
        history=False,
        required=("gsis_id", "display_name", "position", "latest_team", "status", "last_season", "pfr_id", "espn_id"),
        what="one row per player: id crosswalk (GSIS, PFR, ESPN), current team and roster status",
    ),
)
BY_KEY = {ds.key: ds for ds in DATASETS}


# ---------------------------------------------------------------- downloading


def _read_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def fetch_file(tag: str, filename: str, manifest: dict, force: bool = False) -> str:
    """Download one release file unless the local copy is already the current one.

    Returns "downloaded" or "unchanged". Raises FileNotFoundError if nflverse has
    no such file. The file's ETag is remembered so an unchanged file costs one
    small request instead of a full download.
    """
    url = f"{BASE_URL}/{tag}/{filename}"
    dest = DATA_DIR / filename
    known = manifest.get(filename, {})
    have_current_copy = dest.exists() and bool(known.get("etag")) and known.get("bytes") == dest.stat().st_size
    headers = {"If-None-Match": known["etag"]} if (have_current_copy and not force) else {}
    checked = datetime.now(timezone.utc).isoformat(timespec="seconds")

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 180)) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(f"nflverse has no file {filename} in release '{tag}' ({url})")
                same_etag = bool(headers) and r.headers.get("ETag") == known.get("etag")
                if r.status_code == 304 or same_etag:
                    known["checked_utc"] = checked
                    manifest[filename] = known
                    return "unchanged"
                r.raise_for_status()
                part = dest.with_name(dest.name + ".part")
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                part.replace(dest)
                manifest[filename] = {
                    "url": url,
                    "etag": r.headers.get("ETag"),
                    "last_modified": r.headers.get("Last-Modified"),
                    "bytes": dest.stat().st_size,
                    "downloaded_utc": checked,
                    "checked_utc": checked,
                }
                return "downloaded"
        except requests.RequestException as exc:  # network trouble or a 5xx: wait and retry
            last_error = exc
            time.sleep(3 * attempt)
    raise RuntimeError(f"could not download {url} after 3 tries: {last_error}")


def resolve_seasons(games: pd.DataFrame, requested: list[int] | None, today: date) -> tuple[int, list[int], int]:
    """Return (current season, seasons to keep usage data for, last completed week)."""
    if requested:
        current = max(requested)
        history = sorted(set(requested))
    else:
        started = games.loc[games["gameday"] <= today.isoformat(), "season"]
        current = int(started.max())
        history = [current]
    finished = games[(games["season"] == current) & games["home_score"].notna()]
    last_week = int(finished["week"].max()) if len(finished) else 0
    if not requested and last_week < MIN_WEEKS_BEFORE_DROPPING_LAST_SEASON:
        history = [current - 1, current]
    return current, history, last_week


def fetch_all(seasons: list[int] | None = None, force: bool = False, today: date | None = None) -> dict:
    """Download every dataset. Returns the run summary that load() relies on."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    today = today or datetime.now(ARIZONA).date()
    files: dict[str, list[str]] = {}

    def get(ds: Dataset, season: int | None = None, optional: bool = False) -> str | None:
        name = ds.file_for(season)
        try:
            status = fetch_file(ds.tag, name, manifest, force)
        except FileNotFoundError:
            if not optional:
                raise
            status = "not yet"
        print(f"  {status:<11}{name}")
        return None if status == "not yet" else name

    try:
        files["schedules"] = [get(BY_KEY["schedules"])]
        games = pd.read_parquet(DATA_DIR / "games.parquet", columns=["season", "week", "gameday", "home_score"])
        current, history, last_week = resolve_seasons(games, seasons, today)
        for ds in DATASETS:
            if ds.key == "schedules":
                continue
            if not ds.per_season:
                files[ds.key] = [get(ds)]
                continue
            wanted = history if ds.history else [current]
            # Before week 1 is final, this season's stat files may not exist yet.
            got = [get(ds, s, optional=(s == current and last_week == 0)) for s in wanted]
            files[ds.key] = [name for name in got if name]
        manifest["_run"] = {
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "current_season": current,
            "history_seasons": history,
            "last_completed_week": last_week,
            "files": files,
        }
    finally:
        _write_manifest(manifest)
    return manifest["_run"]


# -------------------------------------------------------------------- loading


def run_info() -> dict:
    """Summary of the last fetch: current season, last completed week, files on disk."""
    run = _read_manifest().get("_run")
    if not run:
        raise FileNotFoundError("No nflverse data yet. Run: python src/fetch_nflverse.py")
    return run


def load(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one dataset from data/nflverse/ (all seasons the last fetch kept for it)."""
    names = run_info()["files"][key]
    frames = [pd.read_parquet(DATA_DIR / name, columns=columns) for name in names]
    if not frames:
        raise FileNotFoundError(f"nflverse has not published '{key}' for this season yet")
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def kickoff_arizona(games: pd.DataFrame) -> pd.Series:
    """Kickoff as an Arizona-time timestamp, from the schedule's Eastern gameday + gametime."""
    eastern = pd.to_datetime(games["gameday"] + " " + games["gametime"], errors="coerce")
    return eastern.dt.tz_localize(EASTERN, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(ARIZONA)


# ------------------------------------------------------------------ reporting


def _table(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("\n".join("  " + line for line in df.to_string(index=False).splitlines()))


def _heading(key: str, extra: str = "") -> None:
    print(f"\n[{key}] {BY_KEY[key].what}")
    if extra:
        print(f"  {extra}")


def report() -> bool:
    """Print the shape of everything on disk. Returns False if a required column is missing."""
    run = run_info()
    season, last_week = run["current_season"], run["last_completed_week"]
    print(f"\nNFLVERSE DATA | season {season} | last completed week {last_week} | checked {az(datetime.now(timezone.utc))}")
    print(f"stored in {DATA_DIR}\n")

    ok = True
    print(f"{'dataset':<14}{'file':<34}{'rows':>9}{'cols':>6}{'MB':>8}  columns the model needs")
    for ds in DATASETS:
        for name in run["files"][ds.key]:
            path = DATA_DIR / name
            meta = pq.read_metadata(path)
            missing = [c for c in ds.required if c not in set(pq.read_schema(path).names)]
            ok = ok and not missing
            verdict = f"all {len(ds.required)} present" if not missing else "MISSING: " + ", ".join(missing)
            print(f"{ds.key:<14}{name:<34}{meta.num_rows:>9,}{meta.num_columns:>6}{path.stat().st_size / 1e6:>8.2f}  {verdict}")
    if not ok:
        print("\nA required column is missing, so nflverse changed a file layout. Stopping on purpose.")
        return False
    if last_week == 0:
        print("\nNo completed games this season yet, so there is nothing more to summarise.")
        return True

    # ---- weekly player stats
    pw = load("player_week", list(BY_KEY["player_week"].required))
    pw = pw[pw["season"] == season]
    wk = pw[pw["week"] == last_week]
    _heading("player_week", f"weeks {pw['week'].min()}-{pw['week'].max()} | {pw['player_id'].nunique():,} players | "
                            f"player_id is the GSIS id (example {wk['player_id'].dropna().iloc[0]})")
    cols = ["player_display_name", "position", "team", "opponent_team", "week", "carries", "rushing_yards",
            "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "target_share"]
    print(f"  week {last_week}, most carries (RB) and most targets (WR/TE):")
    _table(wk[wk["position"] == "RB"].nlargest(4, "carries")[cols].round({"target_share": 3}))
    _table(wk[wk["position"].isin(["WR", "TE"])].nlargest(4, "targets")[cols].round({"target_share": 3}))

    # ---- play by play
    pbp = load("pbp", list(BY_KEY["pbp"].required))
    pbp = pbp[pbp["season"] == season]
    scrimmage = pbp[((pbp["rush_attempt"] == 1) | (pbp["pass_attempt"] == 1)) & (pbp["two_point_attempt"] == 0)]
    red_zone = scrimmage[scrimmage["yardline_100"] <= 20]
    _heading("pbp", f"weeks {pbp['week'].min()}-{pbp['week'].max()} | {pbp['game_id'].nunique()} games | "
                    f"{len(scrimmage):,} rush/pass plays, {len(red_zone):,} of them in the red zone (yardline_100 <= 20)")
    print(f"  game_half values: {sorted(pbp['game_half'].dropna().unique().tolist())}")
    cols = ["game_id", "qtr", "posteam", "defteam", "play_type", "down", "yardline_100",
            "rusher_player_id", "receiver_player_id", "yards_gained", "touchdown"]
    _table(scrimmage[scrimmage["week"] == last_week].iloc[[0, 1, 2, 3]][cols])

    # ---- snap counts
    sc = load("snap_counts", list(BY_KEY["snap_counts"].required))
    sc = sc[sc["season"] == season]
    _heading("snap_counts", f"weeks {sc['week'].min()}-{sc['week'].max()} | offense_pct runs "
                            f"{sc['offense_pct'].min():.2f} to {sc['offense_pct'].max():.2f} (a fraction, not a percent)")
    cols = ["player", "pfr_player_id", "position", "team", "opponent", "week", "offense_snaps", "offense_pct"]
    _table(sc[(sc["week"] == sc["week"].max()) & (sc["position"] == "RB")].nlargest(4, "offense_snaps")[cols])

    # ---- depth charts
    dc = load("depth_charts", list(BY_KEY["depth_charts"].required))
    latest = dc[dc["dt"] == dc["dt"].max()]
    newest = datetime.fromisoformat(dc["dt"].max().replace("Z", "+00:00"))
    _heading("depth_charts", f"{dc['dt'].nunique()} snapshots since {dc['dt'].min()[:10]} | newest {az(newest)} "
                             f"covers {latest['team'].nunique()} teams, {len(latest):,} rows")
    team = sorted(latest["team"].unique())[0]
    skill = latest[(latest["team"] == team) & latest["pos_abb"].isin(["QB", "RB", "WR", "TE"])]
    starters = skill[skill["pos_rank"] <= skill["pos_abb"].map({"QB": 1, "RB": 2, "WR": 3, "TE": 1})]
    print(f"  {team} top of the chart (pos_rank 1 = starter; WR ranks 1-3 are the three starting slots):")
    _table(starters.sort_values(["pos_abb", "pos_rank"])[["team", "pos_abb", "pos_slot", "pos_rank", "player_name", "gsis_id", "espn_id"]])

    # ---- injuries
    inj = load("injuries", list(BY_KEY["injuries"].required))
    inj_week = int(inj["week"].max())
    _heading("injuries", f"weeks {inj['week'].min()}-{inj_week} | one row per player per week "
                         f"(duplicate player-weeks: {int(inj.duplicated(['gsis_id', 'week']).sum())})")
    if inj_week <= last_week:
        print(f"  NOTE: no week {last_week + 1} rows yet; the newest report is for week {inj_week}, which is already played.")
    print(f"  report_status values:   {inj['report_status'].value_counts(dropna=False).to_dict()}")
    print(f"  practice_status values: {inj['practice_status'].value_counts(dropna=False).to_dict()}")
    listed = inj[(inj["week"] == inj_week) & inj["position"].isin(["QB", "RB", "WR", "TE"]) & inj["report_status"].notna()]
    _table(listed.head(4)[["week", "team", "full_name", "position", "report_primary_injury", "report_status", "practice_status"]])

    # ---- schedule
    games = load("schedules", list(BY_KEY["schedules"].required))
    games = games[games["season"] == season]
    todo = games[games["home_score"].isna()].copy()
    _heading("schedules", f"season {season}: {len(games)} games, {len(games) - len(todo)} final | "
                          "spread_line > 0 means the home team is favoured")
    if len(todo):
        todo["kick"] = kickoff_arizona(todo)
        slate = todo[todo["week"] == todo["week"].min()].sort_values("kick")
        print(f"  week {int(slate['week'].iloc[0])} kickoffs in Arizona time ({len(slate)} games):")
        for _, day in slate.groupby(slate["kick"].dt.strftime("%Y-%m-%d")):
            slots = [
                f"{kick.strftime('%I:%M %p').lstrip('0')} " + ", ".join(grp["away_team"] + "@" + grp["home_team"])
                for kick, grp in day.groupby("kick")
            ]
            print(f"    {day['kick'].iloc[0].strftime('%a %b %d').replace(' 0', ' ')}: " + " | ".join(slots))
        _table(slate.head(3)[["game_id", "gameday", "gametime", "away_team", "home_team", "spread_line",
                               "total_line", "espn", "away_qb_name", "home_qb_name"]])

    # ---- players
    pl = load("players", list(BY_KEY["players"].required))
    active = pl[pl["last_season"] == season]
    _heading("players", f"{len(pl):,} players all-time, {len(active):,} with last_season == {season} | "
                        f"their status codes: {active['status'].value_counts().to_dict()}")
    _table(pl[pl["gsis_id"].isin(wk.nlargest(3, "carries")["player_id"])])

    # ---- do the files agree with each other?
    print("\n[cross-checks]")
    crosswalk = pl[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    skill_snaps = sc[sc["position"].isin(["QB", "RB", "WR", "TE", "FB"])]
    hit = skill_snaps["pfr_player_id"].isin(crosswalk["pfr_id"])
    print(f"  snap_counts -> GSIS id through players.pfr_id: {hit.mean():.1%} of {len(skill_snaps):,} skill-position rows"
          f" ({skill_snaps.loc[~hit, 'player'].nunique()} players unmatched)")
    plays = scrimmage[scrimmage["week"] == last_week]
    by_rusher = plays[plays["rush_attempt"] == 1].groupby("rusher_player_id").size()
    by_target = plays[(plays["pass_attempt"] == 1) & (plays["sack"] == 0)].groupby("receiver_player_id").size()
    stats = wk.dropna(subset=["player_id"]).set_index("player_id")
    for label, counted, official in (("carries", by_rusher, stats["carries"]), ("targets", by_target, stats["targets"])):
        both = pd.concat([counted.rename("pbp"), official.rename("stats")], axis=1).dropna()
        print(f"  week {last_week} {label}: counted from pbp == player_week for {int((both['pbp'] == both['stats']).sum())}"
              f" of {len(both)} players")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download nflverse data and report its shape.")
    parser.add_argument("--seasons", type=int, nargs="+", help="seasons to fetch (default: work it out from the schedule)")
    parser.add_argument("--force", action="store_true", help="re-download files even if unchanged")
    parser.add_argument("--no-report", action="store_true", help="skip the shape report")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # an unusual character in a name must not crash a Windows console

    print("Fetching nflverse files:")
    run = fetch_all(seasons=args.seasons, force=args.force)
    print(f"Season {run['current_season']}, last completed week {run['last_completed_week']}, "
          f"usage history from {run['history_seasons']}.")
    if args.no_report:
        return 0
    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
