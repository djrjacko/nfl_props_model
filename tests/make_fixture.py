"""Rebuild the cached GB@LA fixture the model's unit test runs on.

    python tests/make_fixture.py

It copies, from whatever is on disk right now, only what one game needs: the
nflverse rows for every game either team has played this season, the odds
pulled for the game, and the player status file. The test then runs with no
network and no dependence on later data. The copy in git was made on
2026-11-24 from the odds pulled at 7:59 PM MST for Green Bay at Los Angeles
Rams (week 12).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fetch_nflverse as nfl  # noqa: E402
import fetch_odds  # noqa: E402
import fetch_status  # noqa: E402
import model  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "gb_la"
TEAMS = {"GB", "LA"}


def main() -> None:
    season = nfl.run_info()["current_season"]
    OUT.mkdir(parents=True, exist_ok=True)

    games = nfl.load("schedules", model.GAME_COLUMNS)
    games = games[games["season"] == season]
    ours = games[games["home_team"].isin(TEAMS) | games["away_team"].isin(TEAMS)]["game_id"]

    pbp = nfl.load("pbp", model.PBP_COLUMNS)
    pbp = pbp[pbp["game_id"].isin(ours)]
    stats = nfl.load("player_week", model.PLAYER_WEEK_COLUMNS)
    stats = stats[stats["game_id"].isin(ours)]
    snaps = nfl.load("snap_counts", model.SNAP_COLUMNS)
    snaps = snaps[snaps["game_id"].isin(ours)]
    players = nfl.load("players", model.PLAYER_COLUMNS)
    seen = set(stats["player_id"].dropna()) | set(pbp["rusher_player_id"].dropna()) | set(pbp["receiver_player_id"].dropna())
    players = players[players["gsis_id"].isin(seen) | players["pfr_id"].isin(snaps["pfr_player_id"])
                      | (players["latest_team"].isin(TEAMS) & (players["last_season"] == season))]

    for name, frame in (("games", games), ("pbp", pbp), ("player_week", stats), ("snaps", snaps), ("players", players)):
        frame.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"  {name + '.parquet':<22}{len(frame):>7,} rows")

    props, lines = fetch_odds.props(), fetch_odds.game_lines()
    keep = lines[lines["home"].isin(TEAMS) & lines["away"].isin(TEAMS)]["event_id"]
    props[props["event_id"].isin(keep)].to_csv(OUT / "props.csv", index=False, lineterminator="\n")
    lines[lines["event_id"].isin(keep)].to_csv(OUT / "game_lines.csv", index=False, lineterminator="\n")
    shutil.copyfile(fetch_status.STATUS_FILE, OUT / "status.json")
    (OUT / "season.txt").write_text(f"{season}\n", encoding="utf-8", newline="\n")
    print(f"  props.csv, game_lines.csv, status.json, season.txt -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
