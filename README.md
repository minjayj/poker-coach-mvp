# Poker Coach MVP

Poker Coach MVP is a local-first assistant for consent-based home games.
The main page is a focused shared-table preview plus a coach conversation.
The **Setup** toggle at the top opens seats and detailed tools over the live
view; the same tools are also available at `/advanced`.

For the engineering architecture, model assumptions, complete file map, and
solver mathematics, see [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md).

It includes an on-demand Hold'em spot-analysis tool. You choose when to run
it; it does not connect to poker software, control a client, or create
automatic pop-ups.

## What it does

- Configure players, seats, stacks, and blinds.
- Switch between cash and tournament rules. Cash mode accepts rake and a cap;
  tournament mode accepts remaining players, paid places, prizes, and ante.
- Select NLH or PLO5 independently of cash/tournament format. A shared ClubGG
  window can identify the green NLH or blue PLO5 felt after three consistent
  frames; uncertain or multi-window captures require manual confirmation.
- Log actions manually or parse simple spoken-action transcripts.
- Enter hero and board cards from the vision module or manually.
- Recalculate pots, stacks, current betting information, VPIP, and PFR.
- Store local player history in SQLite after each completed hand.
- Calculate calls, raises, and Aggression Factor (AF) per player.
- Discuss a spot in the live chat by stating cards, board, pot, price to call,
  and a known opponent's action. The coach asks for missing facts before
  analyzing and can read its replies aloud when you turn on **Voice**.
- With the configured Azure `gpt-6-luna` model (or optional local Ollama),
  discuss general poker concepts and ask follow-up questions in natural
  language. The model receives only bounded chat history and recorded hand
  context; the parser and decision engine still control facts and specific
  move analysis.
- Run optional NLH fold/call/raise analysis against a selected opponent's
  locally learned NLH tendencies. PLO5 has a separate, bounded equity study.
- Inspect a separately labeled open-data heads-up NLH push/fold study library
  after a hand. This does not power live coaching.
- Show a browser camera, browser screen share, or the latest frame supplied by
  an external video/CNN adapter.
- Turn analysis into compact coaching notes and optionally read those notes
  aloud with the browser's built-in speech feature.
- Opt in to record a single shared window locally, review sampled frames after
  play, and test a card-suit calibration using only human-verified labels.

## Important boundaries

- Use only in friendly games where everyone has consented.
- The chat responds to your messages; it does not yet recognize complete
  hands from screen video or proactively call a move on each turn.
- The solver is a fast single-decision betting abstraction. It is not a full
  solution of the complete no-limit Hold'em game tree and should be treated as
  a decision-support estimate.
- PLO5 coaching compares your five cards against one random five-card hand,
  enforcing exactly two hole and three board cards. It is not a calibrated
  opponent range, multiway solver, or raise strategy; it may decline to name a
  move. It only suggests a check/call/fold candidate on a heads-up river when
  the equity gap is large. It does not yet account for cash rake or tournament ICM.
- The open study library covers only eight exact heads-up NLH stack depths in a
  simplified shove/fold game with no ante, rake, or ICM. It is **not full-game
  GTO**. No open PLO5/PLO6 strategy data is installed; PLO6 is unsupported.
- Player information is stored only on this computer in `data/`.

## Requirements

- Python 3.10 or newer
- A working camera is optional; ClubGG Hold'em window reading requires the
  local RapidOCR/ONNX packages installed from `requirements.txt`
- Windows PowerShell commands below (adapt activation for another shell)

## Setup

From the project folder:

```powershell
cd poker-coach-mvp
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the app:

```powershell
uvicorn backend.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser.
Keep the terminal running while you use the app. If the chat says the local
coach server is offline (or the page reports "Failed to fetch"), restart the
command above and refresh the browser; the browser page alone does not run
the Python server.
The focused live view opens at `/`; the old detailed dashboard is at
[http://127.0.0.1:8000/advanced](http://127.0.0.1:8000/advanced).
The chat transcript scrolls inside its panel; new replies do not make the
whole page grow.

### Azure conversation model

The default conversation provider is the user's Azure OpenAI v1 deployment:
`gpt-6-luna` at `https://pokeragentmodel.services.ai.azure.com/openai/v1/`.
The separate Azure project URL is not the inference URL. Open the live coach at
`http://127.0.0.1:8000/` and enter your Azure key in **Connect your Azure coach**.
The browser sends it once to the loopback server; it is held only in server
memory until restart, and is not saved in the browser or repository. If the
server restarts, reconnect on the live page. A key posted in chat should be
rotated when practical. Alternatively, enter a key in a PowerShell session
before starting the server:

```powershell
$secret = Read-Host "Azure API key" -AsSecureString
$env:AZURE_OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $secret).Password
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Keep that terminal open. `POKER_COACH_AZURE_ENDPOINT` and
`POKER_COACH_AZURE_DEPLOYMENT` can override the configured URL/deployment.
`GET /api/coach/model-status` and the coach header show whether a credential is
configured; the live-page Connect action tests the connection. Only recent
chat text and a small recorded-hand summary are sent to Azure—**not** screen
video. The engine remains responsible for specific move analysis. If Azure is
not configured, that rule-based path still works and open-ended chat reports
the missing model configuration.

For fully offline conversation instead, set `POKER_COACH_LLM_PROVIDER=ollama`,
install/start [Ollama](https://docs.ollama.com/api/chat), and run
`ollama pull gemma3:1b`. You can override that model with
`POKER_COACH_LLM_MODEL`. No Ollama installation is required for Azure mode.

## Focused live workflow

1. Open the top **Setup** toggle, then enter Hero, familiar usernames, stacks, and any
   relevant rake or tournament payouts. This data is retained locally.
2. Return to `/`, choose Hold'em or PLO5 and cash or tournament, then click
   **Share game window** and select the single game window.
3. Chat with the coach: “I have Ah Kh, board Qs Jd 3c, pot 24, to call 6.
   Alex raised. What do you think?” The coach asks for any missing fact,
   gives a tentative line when the model supports it, and explains why.
4. You make the action. After the hand, use **Setup** to correct or
   end the hand and inspect opponent memory.

### Recording plays and improving the reader

While sharing a single ClubGG application window, click **Start local
recording**. A visible recording badge stays on until you click **Stop
recording** or end the share. Recording is opt-in, video-only, and saved on
this computer in `data/play/<session-id>/capture.webm`; no game clicks or
online video service are involved. Each session also stores one JPEG frame
roughly every three seconds and a timestamped log of confirmed actions entered
through this app. A 4 GB video cap prevents unlimited recording. Stop before
leaving the page so the final WebM chunk is saved.

Open [Review plays](http://127.0.0.1:8000/play) after the game. Check sampled
frames, correct Hero/board cards, optionally note the action, and explicitly
save verified labels. The reader's own guesses are **not** ground truth. Once
at least 12 distinct frames covering four different card patterns have been
reviewed, **Train from verified frames**
tests extra suit-shape templates on held-out reviewed frames. It activates a
new local calibration only if that test improves recognition without new
errors; otherwise the old reader stays active. The model file is
`data/play/suit_model.json`. This MVP does **not** train card ranks, betting
actions, bet sizes, or a poker strategy from raw video. More varied,
independently checked examples are needed for those tasks.

The calibrated Hold'em reader can read Hero's two visible cards, complete
board (when dealt), pot, visible seat names, and an opened ClubGG profile's
displayed hands/VPIP/PFR. It accepts a shared ClubGG window or crops one
clearly isolated table from a full-screen share. It requires the layout shown
in the supplied recording; multiple visible tables and other layouts abstain.
Two matching frames are required before cards, pot, or profile numbers update
live state. In that layout, it also detects which named seats visibly have
cards and reads Hero's **Call** price and displayed raise target while the red
action buttons are open. Those reads are stored in the live state; newly seen
usernames create empty local profiles, while a possible name alias waits for
your confirmation before histories are merged. The coach asks only for the
prior bettor/action, unreadable amounts, or obscured seats. It does not infer
the entire betting history from occasional frames. It never reads opponents' hidden
cards or reconstructs unseen past actions. Its chat parser recognizes clearly
labelled values; the optional local language model only handles conversation
and explanations. NLH uses a bounded single-spot
regret-matching estimate, **not full GTO**. PLO5 uses a narrower equity study
and may decline to recommend a move.

If the captured pixels stay identical for 12 seconds, the live view warns that
the image may be frozen. Some Windows game windows stop updating in browser
window capture when minimized or not visible. Keep ClubGG unminimized; if its
own game is moving while the preview is static, stop sharing and select your
display instead, with only one ClubGG table visible. The coach will not assume
that a static picture is current. Text updates appear in chat; spoken replies
require switching **Voice off** to **Voice on**.
An explicitly stated preflop raise/3-bet/four-bet changes the broad range
prior. Postflop betting lines are described but are not separately solved or
treated as proof of a specific hand.
If a future vision adapter sends a complete, high-confidence spot while Hero
is explicitly to act, the live view can post one coaching update per changed
spot. Video frames or color detection alone do not trigger a move call.

## Detailed dashboard workflow

1. Choose **Game Format** and **Game** (NLH or PLO5). For cash, enter the rake if known. For a tournament,
   enter the number remaining and the prizes by finishing place. On a final
   table, list every remaining player in **Seat Setup** with their stack at
   the start of this hand.
2. In **Seat Setup**, enter each player, stack, blind level, and select Hero.
3. In **Cards & Hand Context**, enter Hero Cards and Board Cards (for example,
   `Ah, Kh` for NLH or five cards for PLO5) or let a vision adapter supply them.
4. In **Action Capture**, log raises, calls, folds, bets, and all-ins. Amounts
   can be marked as a total amount or an additional amount.
5. If voice capture is available, type or send a transcript such as
   `Alex raises to 10`; the app parses and can apply it.
6. When the hand is over, click **End Hand**. This archives the hand and
   commits player tendencies to the local SQLite database.
7. Check **Opponent Memory** to see VPIP, PFR, calls, raises, AF, and cash,
   bubble, or final-table history for a familiar username in the selected game.
8. When you want analysis, use **Request Coaching**: select an opponent, set the
   amount to call and (optionally) a raise size, then click
   **Coach Me Through This Spot**.
9. For **post-hand** Hold'em study, use **Open Solver Study**: enter two cards,
   choose an exact available stack depth, and select SB first action or BB
   response to a shove. Read the assumptions and quality label.

The spot-analysis result appears as coaching notes: what the app sees, why the
line has its modeled frequency, and what to verify. The browser can read the
notes aloud. This is an explanation feature, not a voice-controlled bot.

## Live input options

The **Live Input** panel is a source-agnostic preview surface.

- **Use Camera** requests a browser camera and uploads a compressed JPEG frame
  to the local app about every 1.2 seconds.
- **Share ClubGG Window** opens the browser's normal capture picker. Choose the
  ClubGG **application window** (not the entire desktop) and the app sends a
  compressed local preview about once per second. The MVP reports whether the
  selected source is a window and warns when it is not. Capture starts only
  after you approve it in the picker; it never clicks, controls, injects into,
  or otherwise interacts with ClubGG.
- A future OpenCV/CNN process can send frames and recognition data directly to
  the local API. The web UI will display its latest frame even when no browser
  camera is connected.

The app identifies NLH versus PLO5 from three consistent felt-color reads.
For the supplied ClubGG Hold'em layout only, it also runs a local calibrated
OCR/card reader on native-resolution window frames. Clear values must agree
across two frames before changing state. The read status appears below the
preview; the coach asks about missing betting information. A full-desktop or
multi-table share, changed table theme, animation, or obstructing overlay may
make fields unreadable. Select one application window and use chat or the
detailed dashboard to correct anything wrong. It does not infer actions from
pot changes or click the game client.

## Hand truth and vision review

The live state now calculates dealer-based positions, active and folded
players, Hero's amount to call, legal action choices, total contribution per
player, and a side-pot summary. Set the dealer seat in **Seat Setup** so the
position labels are meaningful.

Vision data can include an overall confidence plus confidence per field. The
app automatically applies only high-confidence recognition. Low-confidence
cards, pot values, actors, active-player lists, and actions appear in the
**Vision Review Queue**, where you can apply or dismiss them. This is the
intended correction path for imperfect camera/CNN input.

The queue also protects player identity. If the detector reads `Fantamj` but a
seated player is `Fanta`, it asks whether they are the same person before
recording the action. Approving the match saves a local alias, so later
`Fantamj` detections go to Fanta's existing history. An unknown player, or a
raise/bet with no clear amount, is held for clarification rather than being
silently added to the hand.
The focused chat likewise asks whether a similar username is the same person
before using or saving that alias; it does not infer identity from spelling.

### External adapter contract

Send `POST /api/input/frame` from any local process. `image_data` is optional;
include it as a JPEG/PNG data URL when you want the browser to display a
preview. `observation` is optional structured recognition output. Each action
needs a stable `event_id`, so repeated frames cannot log the same raise twice.

```json
{
  "source_id": "my-local-cnn",
  "image_data": "data:image/jpeg;base64,...",
  "observation": {
    "hero_cards": ["Ah", "Kh"],
    "board_cards": ["Qs", "Jd", "3c"],
    "current_pot": 12,
    "current_street": "flop",
    "current_actor": "Alex",
    "active_players": ["Hero", "Alex"],
    "confidence": 0.91,
    "actions": [
      {
        "event_id": "table-42-action-17",
        "player": "Alex",
        "action": "raise",
        "amount": 6,
        "amount_type": "total",
        "street": "preflop",
        "confidence": 0.94
      }
    ]
  }
}
```

The app keeps only the latest preview frame in memory. It records structured
observations and accepted action events in the local session log.

## Opponent memory

The local database is `data/player_stats.sqlite3`. It has an indexed
`player_id` lookup and stores both fast summaries and complete local hand
history. It groups every familiar player's tendencies by active-player count:
`1-2`, `3-4`, `5-6`, and `7-9` players. The coach uses the matching group when
there is enough context, otherwise it uses the lifetime profile.

| Field | Meaning |
| --- | --- |
| `hands_seen` | Completed hands in which the player was seated |
| `vpip_count` | Hands where they voluntarily put money in preflop |
| `pfr_count` | Hands where they raised preflop |
| `total_pots` | Sum of completed-hand pot sizes in which they participated |
| `call_count` | Calls recorded across completed hands |
| `raise_count` | Raises/bets/all-ins recorded across completed hands |
| `aggressive_actions` | Aggressive actions used to calculate AF |

It also records folds, checks, bets, all-ins, showdowns, wins, total money
contributed, and total action count. For later analysis, it stores one
per-player row per completed hand plus every individual action with its street,
amount, contribution, source, confidence, and timestamp.

The player profile calculates:

```text
VPIP = vpip_count / hands_seen
PFR  = pfr_count / hands_seen
AF   = aggressive_actions / call_count
```

If a player has no calls yet, the app reports their aggressive-action count as
their AF rather than dividing by zero.

Stats update only after **End Hand**. This prevents an action that is corrected
or undone during a hand from being counted twice.

### Displayed ClubGG profile stats

When a player profile shows an existing hand count, VPIP, PFR, and AF, you can
save it in **Displayed Player Stats** after seat setup. The calibrated NLH
window reader also captures visible hands/VPIP/PFR from the profile panel after
two matching frames. Exact configured identities are saved; unknown or fuzzy
names go to review, not silently merged. Those values are stored as
a clearly separate `clubgg_profile` baseline; they are not converted into fake
local actions or hand records. The player card shows both the modeled numbers
and the imported baseline.

The same `player_stats` observation contract remains available for a future
OCR/CNN adapter and other table layouts.

### Cash and tournament analysis

Game variant is stored on each completed hand and action. NLH and PLO5 local
stats are calculated separately for the same username; displayed profile
baselines are also tagged by game. Old untagged displayed baselines remain in
the database but are not silently assigned to a variant. The NLH solver is
blocked when PLO5 is selected.

Cash-game utility subtracts an estimated rake on pots you win, using the
configured percentage, optional cap, and no-flop-no-drop setting. If rake is
unknown, leave it at zero and treat close decisions cautiously.

Tournament hands are tagged as regular, bubble, or final table using remaining
players and paid places. Action and hand records keep that tag. When a player
has at least ten recorded hands in the same stage, the coach uses that measured
stage sample; otherwise it falls back to their broader profile. This allows
the model to learn whether a particular player actually tightens up while
waiting for eliminations.

At a complete final table (every remaining stack and every paid prize known),
the coach compares simplified fold/call/raise branches in Independent Chip
Model (ICM) prize equity. Equal prizes can represent satellite tickets. If
the field or prizes are incomplete, the app says so and uses chip EV instead.
ICM is an approximation of payout pressure, not a complete tournament solver.

## Solver overview

The quantitative solver endpoint below is **NLH only**. Select PLO5 in the
dashboard and use **Coach Me Through This Spot** for its separate limited
equity study; enter five Hero cards. The NLH endpoint rejects PLO5 requests.

The separate MIT-licensed open study artifact in `data/open_study/` comes
from [David Vayntrub's pokersolver](https://github.com/davidvayn/pokersolver)
at a pinned commit. It supplies approximate heads-up push/fold frequencies
for 2, 3, 5, 8, 10, 12, 15, and 20bb. The UI does not interpolate stacks or
inject this data into live coaching. Source, checksum, license, assumptions,
and original quality flags are recorded in
[data/open_study/README.md](data/open_study/README.md).

The on-demand solver takes:

- `board`: zero to five cards, such as `["Qs", "Jd", "3c"]`
- `hero_hand`: exactly two cards, such as `["Ah", "Kh"]`
- `current_pot`: current pot size
- `opponent_stats`: VPIP, PFR, AF, and optional sample counts; or an
  `opponent_id` that loads the local profile
- optional `to_call` and `raise_to` values

It works in three bounded steps:

1. Builds a weighted legal opponent range. VPIP and PFR are Bayesian-shrunk
   toward population priors when the sample is small. A high VPIP increases the
   weight of weaker holdings, including hands such as 72o and J4s. Position,
   preflop sequence, effective stack depth, and table size also shape the
   initial range.
2. Samples legal opponent holdings and remaining board cards to estimate hero
   equity. Hand comparisons use the `phevaluator` perfect-hash evaluator.
3. Runs NumPy-based regret matching over `fold`, `call`, and `raise`, returning
   the average action frequencies for that decision abstraction.

That third step is a **one-decision heuristic**, not full-game CFR or a Nash
equilibrium. It cannot recover an optimal strategy without accurate ranges,
stacks, rake, positions, betting history, and the complete future game tree.
Adding an external library does not remove those input requirements.

### Optional solver-framework study lane

`backend/engine/texas_solver_study.py` imports a native TexasSolver JSON
strategy node for **post-hand, heads-up NLH cash-game study**. It requires a
manifest written by the person who configured the solve. The manifest records
the board, pot, call price, effective stack, rake, position, exact action path,
and both assumed ranges. Importing preserves the native per-combo frequencies
and the source SHA-256. Lookup refuses a different pot, rake, stack, board,
path, or hand; it never silently treats a similar tree as the current game.

After creating an export and a manifest, run:

```powershell
python -m backend.engine.texas_solver_study C:\path\to\texas-export.json C:\path\to\manifest.json data\solver_study
```

Example manifest (change **every** value to match the actual solve):

```json
{
  "provider": "TexasSolver",
  "range_notes": "Hero opening range and villain defending range used in this solve",
  "spot": {
    "board": ["8h", "Qc", "Tc"], "pot": 5.20, "to_call": 2.60,
    "effective_stack": 100, "rake_percent": 0, "hero_position": "IP",
    "action_path": ["CHECK"]
  }
}
```

Use `POST /api/study/texas-solver` with those exact spot fields plus
`"hero_hand": ["Kc", "3s"]`. The API returns 404 when nothing matches.
The live chat and automatic recommendation path **do not consume** these
exports. TexasSolver's own benchmark describes minutes of convergence for a
flop tree; it is not a reliable 500 ms live solve. Check its AGPL/commercial
license before redistributing or offering a hosted service.

#### Weighing a solver reference against an opponent read

`POST /api/study/texas-solver/adjusted` takes the same exact lookup fields,
plus `opponent_id`. Optional `proposed_raise_to` lets the read use the
opponent's observed fold response to a comparable small/medium/large price.
Optional `observed_opponent_bet_to_pot` describes the **actual** bet they made;
an uncommon size reduces confidence in the adjustment, but is never assumed
to mean strength or a bluff.

The response keeps three distinct distributions: the imported solver's
`baseline`, the app's approximate player-specific model on the tree's legal
actions, and `blended_frequencies`. The latter is a bounded mixture, **not a
new GTO solution or a validated best response**. It stays identical to the
baseline until at least 12 comparable, trusted completed hands contain a
facing-bet decision. Even with ample history, player-model weight is capped
at 35%. The response exposes the exact weight, counts, size buckets, source,
and reason so the coach can explain the adjustment without hiding uncertainty.

Completed hands are kept in `data/hand_archive.json`. New records include the
pot and call price before each action; older records without these fields are
excluded from size analysis. The indexed SQLite database remains the source
for VPIP/PFR and other aggregate profile stats. `GET /api/players/{id}/read`
shows the JSON-derived evidence for the current format/phase/table-size
context. The live coach does not substitute this post-hand blend for an
unmatched or incompletely observed spot. It can report a sufficiently sampled
contextual read in its explanation while keeping the model estimate and the
opponent evidence separate.

`backend/engine/research_frameworks.py` also contains optional OpenSpiel CFR+
and RLCard chance-sampling CFR runs on **Kuhn** and **Leduc** poker. These are
offline algorithm checks, not strategies for ClubGG Hold'em. In a separate
compatible research environment, install `open_spiel` or `rlcard`, then run
`python -m backend.engine.research_frameworks openspiel --iterations 1000` or
`python -m backend.engine.research_frameworks rlcard --iterations 100`.

The default 3,000 equity samples are intended for quick local responses. Each
answer reports the actual elapsed time; speed depends on your hardware and the
number of samples selected.

## API

### State and input

- `GET /api/state` — current game state
- `POST /api/config/seats` — configure players, stacks, blinds, and Hero
- `POST /api/config/game-format` — cash rake or tournament payout context
- `POST /api/action` — log an action
- `POST /api/correction` — fix or undo the latest action
- `POST /api/speech/transcript` — parse a spoken-action transcript
- `POST /api/vision/card_manual` — set hero and board cards
- `POST /api/hand/end` — archive the hand and update player memory
- `POST /api/input/frame` — receive a generic camera/screen/CNN frame and
  optional structured observation
- `GET /api/input/latest` — retrieve the newest generic preview frame/status
- `POST /api/input/stop` — clear a stopped browser preview
- `GET /api/reviews` — low-confidence recognition awaiting review
- `POST /api/reviews/{review_id}` — apply or dismiss an uncertain detection

### Player profiles

- `GET /api/players` — all locally known player profiles
- `GET /api/players/{player_id}` — one player profile
- `GET /api/players/{player_id}/table-breakdown` — player stats by table size
- `GET /api/players/{player_id}/history` — recent per-hand and per-action data
- `GET /api/players/{player_id}/phase-breakdown` — cash and tournament stage samples
- `GET /api/players/{player_id}/read` — contextual action/size evidence from completed-hand JSON

### On-demand analysis

`POST /api/solver/recommend`

`POST /api/coach/brief` builds the same analysis into explanation-first notes
for the current state. It accepts an optional `opponent_id`, `to_call`,
`raise_to`, and `simulations`.

`GET /api/study/push-fold` lists the open study catalog.
`POST /api/study/texas-solver/adjusted` compares an exact imported tree with
a conservative player-history overlay for post-hand study.
`POST /api/study/push-fold` looks up one exact heads-up NLH spot:

```json
{"effective_stack_bb": 10, "hero_hand": ["Ah", "Kd"], "decision": "sb_first"}
```

Use `bb_vs_shove` for the big blind's call/fold response. Unavailable stacks,
duplicate cards, and Omaha-size hands are rejected.

Example request using the local profile for Alex:

```json
{
  "board": ["Qs", "Jd", "3c"],
  "hero_hand": ["Ah", "Kh"],
  "current_pot": 12,
  "to_call": 4,
  "raise_to": 14,
  "opponent_id": "Alex"
}
```

Or provide stats explicitly. Rates may use decimal (`0.40`) or percentage
(`40`) notation:

```json
{
  "board": ["Qs", "Jd", "3c"],
  "hero_hand": ["Ah", "Kh"],
  "current_pot": 12,
  "to_call": 4,
  "opponent_stats": {
    "vpip": 45,
    "pfr": 12,
    "af": 1.2,
    "hands_seen": 100,
    "vpip_count": 45,
    "pfr_count": 12
  }
}
```

Example response fields:

```json
{
  "recommended_action": "raise",
  "action_frequencies": {"fold": 0.001, "call": 0.002, "raise": 0.997},
  "equity": 0.6313,
  "fold_equity": 0.3642,
  "bayesian_adjustment": {
    "posterior_vpip": 0.4167,
    "posterior_pfr": 0.13,
    "loose_adjustment": 0.6667
  },
  "latency_ms": 141.19
}
```

## Project layout

```text
backend/
  engine/
    cfr_solver.py        # range model, equity sampling, regret matching
    plo5.py              # exact-two/exact-three PLO5 equity study
    tournament.py        # exact small-field ICM payout equity
    hand_evaluator.py    # PH evaluator adapter and card validation
    open_study.py        # read-only, exact-spot open-data lookup
    texas_solver_study.py # native-export import and exact post-hand lookup
    opponent_adjustment.py # contextual JSON read and capped policy blend
    research_frameworks.py # optional Kuhn/Leduc CFR research checks
  player_database.py     # SQLite schema and completed-hand stat updater
  vision/variant.py      # conservative single-window felt-color game cue
  vision/clubgg_holdem.py # calibrated local NLH cards/pot/names/profile reader
  vision/play_training.py # verified-frame suit calibration and validation
  play_recordings.py     # opt-in local WebM, sampled frames, labels, events
  main.py                # FastAPI routes
  models.py              # API request and response models
  storage.py             # game state, archive, and DB integration
  speech/                # deterministic speech parser
  vision/camera.py       # camera placeholder
frontend/                # local browser interface
  play.html, play.js      # post-game video/frame review and training UI
data/                    # game state, archive, and local SQLite database
  play/                  # opt-in recordings and learned suit model (gitignored)
  open_study/            # pinned MIT-licensed HU NLH push/fold study artifact
  solver_study/          # locally imported TexasSolver exports (gitignored)
tests/                   # API, parser, and database tests
```

## Tests

Run the test suite from `poker-coach-mvp`:

```powershell
python -m pytest -q
```

## Dependencies and attribution

The project uses the Apache-2.0 licensed `phevaluator` Python package, based on
HenryRLee's PokerHandEvaluator perfect-hash implementation:

- [PokerHandEvaluator project](https://github.com/HenryRLee/PokerHandEvaluator)
- [phevaluator package](https://pypi.org/project/phevaluator/)

The calibrated Hold'em window reader uses local [RapidOCR](https://github.com/RapidAI/RapidOCR)
with ONNX Runtime for visible text. Captured frames stay in the local app's
latest-frame memory; the adapter does not upload them to an online OCR service.
