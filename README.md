# NFL prop model and picks board

An automatic board of NFL player prop picks. GitHub Actions runs it on a schedule, it writes JSON into
`docs/`, and GitHub Pages serves a phone-friendly page that reads that JSON. It re-runs before kickoffs to
catch inactives and line moves and says what changed.

This is a personal model. Its probabilities have been checked against results, not against the books'
lines, so whether its edges are real can only be judged from the picks log over time.

## How a run works

Each run does four things in order, only for games kicking off in the look-ahead window (36 hours, or 60
for the Friday run so it can see Sunday):

| Step | Script | What it does |
|---|---|---|
| 1 | `src/fetch_nflverse.py` | Downloads nflverse files: weekly player stats, play by play, snap counts, depth charts, injuries, the schedule and the player id table. Stops if a column the model needs has disappeared. |
| 2 | `src/fetch_odds.py --pull` | The Odds API: spreads and totals, then props for each game (rush attempts, rush yards, receptions, receiving yards, anytime TD) from DraftKings and FanDuel only. Games with a spread of 10 or more also get alt yardage lines. Logs every credit spent. |
| 3 | `src/fetch_status.py` | ESPN's public JSON for who is out, doubtful, questionable or inactive and who the starting QB is, with nflverse as the fallback. |
| 4 | `src/board.py` | Runs `src/model.py` and writes the board files. |

Outputs:

- `docs/picks.json` everything the Board tab shows
- `docs/board.txt` the same board as plain text
- `docs/changes.json` what changed since the previous run, in plain words
- `logs/picks_log.csv` one row per recommended leg per run, for grading against box scores later
- `logs/odds_credits.csv` one row per paid Odds API call
- `logs/practice_log.csv` practice participation seen each run, used for the "no practice this week" gate

## The model

Only these props: RB rushing attempts, RB rushing yards, RB/WR/TE receptions, WR/TE receiving yards, and
anytime touchdown. No game totals and no QB passing props.

1. **Team volume.** Plays and run rate over the team's last 5 games, recency weighted. The run rate moves
   0.3 points per point of spread, relative to the spreads those 5 games were played at.
2. **Player volume.** Team volume times the player's share. Carry share is his last 3 games (weights
   50/30/20). Target share is 30% last 3 games and 70% season.
3. **Yards.** Volume times efficiency, where efficiency is 60% the player's own and 40% what the opponent
   has allowed to his position.
4. **Anytime TD.** His share of the team's scoring chances times the touchdowns the spread and total imply.
5. **Probability and edge.** Counts use a Poisson-family distribution and yards a normal curve. Edge is the
   model probability minus the book's probability with the vig removed. Anytime TD is priced Yes-only, so
   its vig is removed by scaling each game's list to the expected number of different scorers.
6. **Legs.** Edge of 5% or more, at the better of the two books. A tier is 8% or more with no flags; B tier
   is 5% to 8% or any flag. Edges of 20% or more are listed as "not trusted" instead of offered, and a
   player needs a real role (6 carries or 3 targets a game, 35% of snaps for a touchdown).
7. **Injury gate.** A leg is dropped when the player or his starting QB is out, doubtful, inactive, or
   questionable with practice evidence this week that is all "did not practice". Questionable players are
   flagged.
8. **Blowout rule.** With a spread of 10 or more, the favourite's starters do not get the main yardage
   line; the board offers a lower alt line if one has an edge, and says so. TD legs are unaffected.
9. **Game scripts.** Legs are grouped by which team being ahead helps them, and same-game tickets are built
   inside a group, one leg per player, at one book.
10. **Boosts.** The 50% boost goes to the safest 2-leg ticket priced +150 to +250 and the 100% boost to the
    best 3-leg ticket priced +300 to +400. The same anchor leg is never on both.

### The calibration layer

The rules above were back-tested on 2026 weeks 5 to 11, projecting each player-game from earlier games
only. As written they ran high and were too sure of themselves, so a calibration layer sits on top:
projections are pulled toward the middle, counts use a negative binomial instead of a pure Poisson, the
yards curve is centred on the typical game rather than the average, target share leans on the season, and
the touchdown share blends red-zone share with overall touches. Every leg also carries a "literal"
probability, which is the same bet priced by the rules exactly as written, so the two can be compared over
a season.

```
python src/model.py --backtest   # the evidence, recomputed from current data
python src/model.py --literal    # price the cached slate with the rules as written
```

Two ideas were tested and left out because they made the back-test worse: handing an absent teammate's
share to the others, and taking shares only from the games since a long-term absence began. Those
situations are flagged instead. The second comparison is reprinted by `--backtest`.

### What is still unverified

- How ESPN shows game-day inactives before kickoff. Until that has been seen live, "inactive" comes from an
  ESPN note saying so on game day, and `fetch_status.py` records what the game-day roster shows without
  acting on it.
- The blowout rule has only run in a unit test with a made-up alt line.
- Same-game ticket prices are the legs multiplied together. A book prices a same-game parlay shorter than
  that when the legs move together, so check the real price before using a boost.

## The website

`docs/index.html` is a single file with three tabs: Board, Changes and Re-run. It reads `picks.json` and
`changes.json` with a fresh address every time so a phone never shows a stale board. Times are Arizona
time. The top of the board counts overs, unders and TDs and warns when more than 75% of the legs are
unders. The Re-run tab starts a run with one tap and lists the next runs.

## Re-running by hand

**From the site:** the Re-run tab has a "Run now" button once a token is saved on the phone (see below).

**On GitHub:** open the Actions tab, pick "Run the prop board", press "Run workflow". The Re-run tab
links straight to that page.

### The one-tap button and its token

Starting a run needs a GitHub token. A token written into the page would be readable by anyone, so the
page asks for one once and keeps it in that browser's storage on that device only. It is sent to
`api.github.com` and nowhere else.

1. On the phone, open the Re-run tab and tap "Create the token on GitHub". The form opens with the name,
   a 90-day expiry and the Actions permission filled in.
2. Under "Repository access" choose "Only select repositories" and pick this repository.
3. Check that Permissions shows Actions: Read and write (Metadata: Read-only is added automatically).
4. Tap "Generate token", copy it, go back to the Re-run tab, paste it and tap "Save on this phone".

What that token can do: start, cancel and read workflow runs in this one repository. It cannot read or
change code, secrets or settings. To revoke it, delete it on GitHub under Settings, Developer settings,
Personal access tokens, Fine-grained tokens. When it expires, create a new one the same way.

One thing to know: every GitHub Pages site under the same account shares one web address origin, so a
script on another Pages site of yours could read this browser storage. Keep that in mind before adding
third-party scripts to any of them.

**On your PC** (PowerShell, from the project folder):

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your Odds API key after `ODDS_API_KEY=`. Then:

```
.\.venv\Scripts\python.exe src\fetch_nflverse.py
.\.venv\Scripts\python.exe src\fetch_odds.py          # free dry run: what a pull would cost
.\.venv\Scripts\python.exe src\fetch_odds.py --pull   # spends credits
.\.venv\Scripts\python.exe src\fetch_status.py
.\.venv\Scripts\python.exe src\board.py
```

Add `--hours 60` to the last three to look further ahead. Commit and push `docs/` and `logs/` to publish.

Tests: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

## Schedule

GitHub's cron is UTC; Arizona is UTC-7 all year. Scheduled runs can start several minutes late.

| Arizona time | Cron (UTC) | Purpose |
|---|---|---|
| Fri 3:37 PM | `37 22 * * 5` | Final injury report board (looks 60 hours ahead) |
| Sat 9:07 PM | `7 4 * * 0` | Roster moves and line moves |
| Sun 6:37 AM | `37 13 * * 0` | Overnight line moves |
| Sun 9:47 AM | `47 16 * * 0` | After early-game inactives |
| Sun 10:37 AM | `37 17 * * 0` | Final early check |
| Sun 1:07 PM | `7 20 * * 0` | After late-game inactives |
| Sun 5:07 PM | `7 0 * * 1` | After Sunday-night inactives |
| Wed 9:07 PM | `7 4 * * 4` | Thursday game |
| Thu 3:07 PM | `7 22 * * 4` | Thursday game |
| Thu 4:52 PM | `52 23 * * 4` | Thursday game, after inactives |
| Sun 9:07 PM | `7 4 * * 1` | Monday game |
| Mon 3:07 PM | `7 22 * * 1` | Monday game |
| Mon 4:52 PM | `52 23 * * 1` | Monday game, after inactives |

A normal week costs roughly 370 to 450 Odds API credits.

### Kickoffs the fixed schedule misses

Inactives come out 90 minutes before kickoff, and the table above only has a run after that point for the
usual Thursday-night, Sunday and Monday-night kickoffs. `.github/workflows/pregame.yml` covers the rest.
Every 15 minutes from September to February (4:07 AM to 7:52 PM Arizona time) it runs `src/pregame.py`,
which reads the upcoming games from The Odds API's free events feed and starts "Run the prop board" when a
game is 85 to 25 minutes from kickoff and nothing has run since inactives were due. A game counts as
covered when a fixed run falls 85 to 40 minutes before its kickoff. If a fixed run is 20 minutes overdue
and never started, the check stops waiting for it and starts one.

That picks up Wednesday games, Thanksgiving, Saturdays late in the season and early kickoffs abroad. It
also picks up September and October, when Eastern daylight time puts every kickoff an hour earlier on the
Arizona clock than the fixed times assume. The check itself costs no credits; each run it starts costs the
same as any other.

```
python src/pregame.py   # show which games are covered, which get their own run, and when
```

`run.yml` also holds one dated line for the Wednesday November 25, 2026 game. A dated cron line repeats
every year, so delete it once that game has been played.

GitHub turns scheduled workflows off after 60 days without a commit. During the season the board commits
keep them alive; before a new season, check the Actions tab and re-enable both workflows if needed.

## Settings

`config.json`:

- `rooting_teams` teams you root for. A leg that wins when one of them does badly (an under on one of
  their players) is tagged "rooting conflict". Tagged legs stay on the board; the site has a switch to hide
  them. Use nflverse team codes such as `GB`.
- `boosts` the boosts you usually have: size, maximum stake, number of legs, price range, and whether to
  pick the safest ticket or the best-value one.

The Odds API key lives in `.env` locally and in the repository secret `ODDS_API_KEY` for Actions. It is
never printed, logged or cached.

## Data sources

- [nflverse](https://github.com/nflverse/nflverse-data) release files for usage data
- [The Odds API](https://the-odds-api.com) for lines; raw responses are cached in `data/odds/` and not
  committed, because its terms do not allow republishing the data as files
- ESPN's public JSON endpoints for player status; they are unofficial and may change without notice
