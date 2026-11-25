# NFL prop model and picks board

An automatic board of NFL player prop picks. An outside scheduler starts it on GitHub Actions several
times a day, it writes JSON into `docs/`, and GitHub Pages serves a phone-friendly page that reads that
JSON. It re-runs once inactives are due before each kickoff and says what changed.

This is a personal model. Its probabilities have been checked against results, not against the books'
lines, so whether its edges are real can only be judged from the picks log over time.

## How a run works

Each run does four things in order, only for games kicking off in the look-ahead window (the `hours` the
run is started with: 60 for every automatic run):

| Step | Script | What it does |
|---|---|---|
| 1 | `src/fetch_nflverse.py` | Downloads nflverse files: weekly player stats, play by play, snap counts, depth charts, injuries, the schedule and the player id table. Stops if a column the model needs has disappeared. |
| 2 | `src/fetch_odds.py --pull` | The Odds API: spreads, totals and moneylines, then props for each game (rush attempts, rush yards, receptions, receiving yards, anytime TD) from DraftKings and FanDuel only. Games with a spread of 10 or more also get alt yardage lines. Logs every credit spent. |
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
anytime touchdown for running backs, receivers and tight ends. No game totals, no QB passing props and no
QB anytime touchdowns.

1. **Team volume.** Plays and run rate over the team's last 5 games, recency weighted. The run rate moves
   0.3 points per point of spread, relative to the spreads those 5 games were played at.
2. **Player volume.** Team volume times the player's share. Carry share is his last 3 games (weights
   50/30/20). Target share is 30% last 3 games and 70% season.
3. **Yards.** Volume times efficiency, where efficiency is 60% the player's own and 40% what the opponent
   has allowed to his position.
4. **Anytime TD.** His share of the team's scoring chances times the touchdowns the spread and total imply.
   A quarterback's anytime TD is not priced: a share of touches does not describe how a quarterback
   scores, so those props are left off every tier and ticket and counted in the board's notes.
5. **Probability and edge.** Counts use a Poisson-family distribution. Rushing yards use a right-skewed
   gamma curve and receiving yards a normal curve centred on the typical game (see "The yards curves"
   below). Edge is the model probability minus the book's probability with the vig removed. Anytime TD is
   priced Yes-only, so its vig is removed by scaling each game's list to the expected number of different
   scorers.
6. **Legs.** Edge of 5% or more, at the better of the two books. A tier is 8% or more with no flags; B tier
   is 5% to 8% or any flag. Edges of 20% or more are listed as "not trusted" instead of offered, and a
   player needs a real role (6 carries or 3 targets a game, 35% of snaps for a touchdown).
   **The two models must agree.** Every leg is also priced by the rules exactly as written. For an over or
   an under, the calibrated probability and the as-written one must fall on the same side of 50%; for a
   touchdown, both must be above the book's probability. A leg that fails gets the flag "calibration
   flips the literal lean", which puts it in B tier and keeps it out of every ticket and boost.
7. **Injury gate.** A leg is dropped when the player or his starting QB is out, doubtful, inactive, or
   questionable with practice evidence this week that is all "did not practice". Questionable players are
   flagged.
8. **Blowout rule.** With a spread of 10 or more, the favourite's starters do not get the main yardage
   line; the board offers a lower alt line if one has an edge, and says so. TD legs are unaffected.
9. **Game scripts.** Legs are grouped by which team being ahead helps them, and same-game tickets are built
   inside a group, one leg per player, at one book.
10. **The anchor rule.** Every ticket starts from an anchor leg: a running back's rush attempts or rush
    yards over, an anytime TD, or a team moneyline. The other legs come from the same game script, so no
    ticket is made of unders alone. The model has no view on who wins a game, so a moneyline is carried at
    the book's own probability with the vig removed: it adds no edge and is marked that way on the ticket.
    Set `require_anchor` to `false` in `config.json` to go back to "any legs that share a game script".
11. **Boosts.** The 50% boost goes to the safest 2-leg ticket priced +150 to +250 and the 100% boost to the
    best 3-leg ticket priced +300 to +400. The same anchor leg is never on both.

### The calibration layer

The rules above were back-tested on 2026 weeks 5 to 11, projecting each player-game from earlier games
only. As written they ran high and were too sure of themselves, so a calibration layer sits on top:
projections are pulled toward the middle, counts use a negative binomial instead of a pure Poisson, the
yards curves are built around the typical game instead of the average, target share leans on the season,
and the touchdown share blends red-zone share with overall touches. Every leg also carries a "literal"
probability, which is the same bet priced by the rules exactly as written, so the two can be compared over
a season.

```
python src/model.py --backtest   # the evidence, recomputed from current data
python src/model.py --literal    # price the cached slate with the rules as written
```

Two ideas were tested and left out because they made the back-test worse: handing an absent teammate's
share to the others, and taking shares only from the games since a long-term absence began. Those
situations are flagged instead. The second comparison is reprinted by `--backtest`.

### The yards curves

Yards are lopsided: a few big games pull the average up, and only about 42% of results beat the
projection. Two curves deal with that, and `YARDS_CURVE` in `model.py` says which one each stat uses:

- **Rushing yards use a gamma curve** whose average is the calibrated projection. Its spread is the
  league-wide spread moved toward the player's own game log, with the log counting as its number of games
  against 12 games of the league figure. A game log on its own understates how far a projection can miss,
  and fitting on it alone made the back-test worse.
- **Receiving yards use a normal curve** centred on the typical game (about 0.86 of the average), with the
  league-wide spread.

Both curves were tried for both stats on 2026 weeks 5 to 11 (`--backtest` prints them side by side,
section 3), and so was a log normal, which did worse than either:

- **Rushing yards:** the gamma is closer. At a line 50% above the projection the under hit 80.8% of the
  time; the gamma says 82.2%, the normal 83.5%. For backs projected for 50 yards or more the normal
  overstates unders by 6 to 7 points and the gamma by about 3.5.
- **Receiving yards:** about level overall, but for players projected under 40 yards the gamma makes
  unders more likely than the results support. At a line 25% above the projection the under hit 67.8%;
  the gamma says 71.2%, the normal 66.9%. That is why receiving yards stayed on the normal.
- **Neither curve removes the under lean.** A lopsided curve puts the typical game below the average,
  which is what makes unders likely in the first place. Both still overstate unders at lines well above
  the projection, by 1 to 4 points.
- The gamma is too generous to overs at rushing lines far below the projection (77.6% said, 72.6%
  happened, at a line half the projection). That is where the blowout rule's alt lines sit.

### What is still unverified

- How ESPN shows game-day inactives before kickoff. Until that has been seen live, "inactive" comes from an
  ESPN note saying so on game day, and `fetch_status.py` records what the game-day roster shows without
  acting on it.
- The blowout rule has only run in a unit test with a made-up alt line.
- Same-game ticket prices are the legs multiplied together. A book prices a same-game parlay shorter than
  that when the legs move together, so check the real price before using a boost.
- A moneyline on a ticket is priced from the book's own odds. The model does not predict winners, so a
  ticket's edge comes only from its other legs.

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

GitHub's own cron never started a run in this repository, so neither workflow has a schedule. An outside
scheduler (cron-job.org) is the only thing that starts runs, and it has two jobs. Times are Arizona time
(`America/Phoenix`, UTC-7 all year).

| Job | When | What it starts |
|---|---|---|
| Board | Every day at 6:37 AM, 3:37 PM and 9:07 PM | `run.yml` with `hours` 60: the board for every game kicking off in the next 60 hours |
| Check | Every 15 minutes, at :00, :15, :30 and :45 | `pregame.yml`: a free look at the kickoffs, which starts a board run once inactives are due |

`config.json` repeats those facts under `outside_schedule`, so that the board can list the next runs and a
run started by the check asks for the same `hours`. Changing them there does not change the scheduler:
when a job changes at cron-job.org, change `config.json` to match.

### Runs after inactives

Inactives come out 90 minutes before kickoff. Each check runs `src/pregame.py`, which reads the upcoming
games from The Odds API's free events feed. A game is due from 85 minutes before kickoff (five minutes
after its inactives are due) until 25 minutes before. The first check inside that window starts "Run the
prop board", which puts the run 70 to 85 minutes before kickoff. Nothing waits for a daily run.

| Kickoff | Inactives due | Check that starts the run |
|---|---|---|
| 11:00 AM | 9:30 AM | 9:45 AM |
| 2:05 PM | 12:35 PM | 12:45 PM |
| 2:25 PM | 12:55 PM | 1:00 PM |
| 6:00 PM | 4:30 PM | 4:45 PM |
| 6:15 PM or 6:20 PM | 4:45 PM or 4:50 PM | 5:00 PM |

**No second run since inactives.** Once a board run has started inside a game's window, every later check
leaves that game alone, so a kickoff time gets one run however many checks follow. Any board run counts,
whoever started it: a daily run that happens to land in the window, or a tap on the Re-run tab. A run from
before the window does not count. Games that kick off together share one run. Kickoffs 20 minutes apart
get one each, because their inactives come out 20 minutes apart.

If a check cannot start the run, the next check tries again, until 25 minutes before kickoff. A run that
starts and then fails still counts as the run, so the checks do not start another one.

```
python src/pregame.py   # show each upcoming game and the check that will start its run
```

### What it costs

A run costs 3 credits for the game lines (spreads, totals, moneylines) plus 5 to 7 for each game in its
window, and nothing when no game is in the window. A check is free. With the two jobs above a week has 21
daily runs and 6 to 10 runs after inactives, and costs roughly 750 to 1,000 Odds API credits.

### The calls the scheduler sends

Both jobs use GitHub's "create a workflow dispatch event" call. For the board:

```
POST https://api.github.com/repos/djrjacko/nfl_props_model/actions/workflows/run.yml/dispatches

Accept: application/vnd.github+json
Authorization: Bearer YOUR_TOKEN
X-GitHub-Api-Version: 2026-03-10
Content-Type: application/json

{"ref":"main","inputs":{"hours":"60"}}
```

For the check, which takes no inputs (GitHub refuses a body that names one):

```
POST https://api.github.com/repos/djrjacko/nfl_props_model/actions/workflows/pregame.yml/dispatches

Accept: application/vnd.github+json
Authorization: Bearer YOUR_TOKEN
X-GitHub-Api-Version: 2026-03-10
Content-Type: application/json

{"ref":"main"}
```

GitHub answers `200` when the run has started, with the run's id and address in the reply. Without the
`X-GitHub-Api-Version` line it answers `204` with an empty reply instead; both mean started. `401` means
the token is wrong, expired or missing, and `422` that the body is wrong, such as `hours` sent to
`pregame.yml`. Those four were seen on this repository on 2026-11-25. `403` or `404` is what GitHub
answers when a token does not cover the repository or lacks Actions write.

**The token** is a fine-grained personal access token limited to this repository with Actions set to Read
and write, made the same way as the one for the Re-run tab. Keep a separate one for the scheduler, so that
either can be deleted without breaking the other. Whoever holds it can start and cancel runs here, which
spends Odds API credits, and nothing else.

**If the scheduler stops,** nothing runs. The board's "Updated" time on the site and the Actions tab show
it; the Re-run tab starts a run by hand. Do not put a `schedule:` block back into a workflow as a spare:
if GitHub's cron ever does fire, every time would get two runs and cost twice the credits.

## Settings

`config.json`:

- `rooting_teams` teams you root for. A leg that wins when one of them does badly (an under on one of
  their players) is tagged "rooting conflict". Tagged legs stay on the board; the site has a switch to hide
  them. Use nflverse team codes such as `GB`.
- `boosts` the boosts you usually have: size, maximum stake, number of legs, price range, and whether to
  pick the safest ticket or the best-value one.
- `require_anchor` `true` (the default) makes every same-game ticket and boost start from an anchor leg: an
  RB rush attempts or rush yards over, an anytime TD, or a team moneyline. `false` turns the rule off and
  moneylines are no longer used.
- `outside_schedule` what the outside scheduler is set to do (see Schedule): `board_runs` are its daily
  times in Arizona time on a 24-hour clock, `board_hours` the `hours` its runs ask for, and `check_minutes`
  the minutes of the hour at which it starts the check. It describes the jobs; it does not create them.

The Odds API key lives in `.env` locally and in the repository secret `ODDS_API_KEY` for Actions. It is
never printed, logged or cached.

## Data sources

- [nflverse](https://github.com/nflverse/nflverse-data) release files for usage data
- [The Odds API](https://the-odds-api.com) for lines; raw responses are cached in `data/odds/` and not
  committed, because its terms do not allow republishing the data as files
- ESPN's public JSON endpoints for player status; they are unofficial and may change without notice
