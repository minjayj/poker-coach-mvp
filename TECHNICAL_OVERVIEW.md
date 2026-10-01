# Poker Coach — CTO Technical Overview

## Executive summary

Poker Coach is a local-first FastAPI web application for consent-based home
games. Its architecture separates four concerns:

1. **Input transport** — browser camera, browser screen share, manual entry, or
   a future vision/CNN adapter provide frames and structured observations.
2. **State and memory** — the application maintains a deterministic hand
   ledger plus a persistent local opponent-statistics database.
3. **Analysis** — a bounded, explainable Hold'em decision model estimates
   equity and regret-matched fold/call/raise frequencies. A separate PLO5
   study samples one random opponent under the exactly-two/exactly-three rule.
   An isolated open-data catalog supports post-hand heads-up NLH push/fold
   study; it never drives live recommendations.
4. **Coaching presentation** — the focused UI uses a shared-table preview and
   coach chat with independently scrolling messages. A top Setup toggle opens
   the detailed dashboard in an overlay; `/advanced` remains available directly.
   Neither interface acts for the user.

The design is intentionally modular: replacing the card-recognition CNN,
using a native equity library, or improving the strategy model does not require
rewriting the user interface or player-memory layer.

## System architecture

```text
Browser camera / screen share     External CNN / vision process      Manual entry
              │                              │                           │
              └────────────── POST /api/input/frame ─────────────────────┘
                                             │
                                  InputStreamHub (latest frame,
                                    de-duplicate action events)
                                             │
                                     GameService / GameState
                                      ├── JSON live hand state
                                      ├── hand archive/session log
                                      └── SQLite opponent memory
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    │                                                 │
          Coaching Brief API                                   Browser dashboard
       range → equity → regret match                      preview → hand → notes
                    │                                                 │
                    └────────────── explanation-first coaching ───────┘
```

Hand processing and storage are local to the computer running the app. A frame
is held only as the most recent in-memory preview. Hand state, logs, archives,
and opponent statistics are written under `data/`. When Azure conversation is
configured, bounded chat text and a small recorded-hand summary go to the
user's Azure endpoint; raw screen frames do not.
The ClubGG Hold'em adapter now emits the names visible at each seat, a
conservative list of seats showing cards, and Hero's call/raise button values
when the action controls are present. Two agreeing frames are required before
these values enter live state. Names are indexed locally; similar new names
are queued for identity review rather than silently merged. The app does not
claim to reconstruct every intervening bet, fold, or raise from sparse frames.
The live chat sends explicitly stated facts to `POST /api/coach/chat` and
receives a reply plus per-hand context. The browser stores the visible chat
transcript locally. The configured Azure `gpt-6-luna` adapter (or optional
local Gemma 3 1B adapter) receives at most eight recent turns and a small
recorded-state summary for open-ended conversation;
it does not receive video frames or mutate game state. A future adapter can cause an automatic reply only when
it supplies a complete high-confidence observation including Hero as actor,
pot, cards, street, and action; raw frames do not meet that gate.

## Repository map

| Path | Responsibility | Key details |
| --- | --- | --- |
| `backend/main.py` | FastAPI composition and routes | Creates shared services, serves `/` and `/advanced`, and exposes state, input, chat, coaching, player, and WebSocket APIs. |
| `backend/coach_chat.py` | Conversational fact parser | Extracts explicitly labelled cards, pot and price; asks for missing facts and keeps each hand's chat context separate. |
| `backend/local_coach_llm.py` | Optional local conversation | Calls only Ollama on loopback, bounds prompt and output size, and falls back cleanly if unavailable. |
| `backend/azure_coach_llm.py` | Azure conversation provider | Uses the OpenAI Responses API against the configured Azure inference endpoint. Reads a server-side key from the environment or one-time local connection form; never persists the key, sends frames, or controls the game. |
| `backend/models.py` | Pydantic contracts | Defines cards/actions, review items, hand state, solver input, generic vision observations, and coach requests. |
| `backend/storage.py` | Hand-state service | Applies actions/corrections, calculates positions, active/folded players, legal choices, side-pot summaries, archives hands, and writes final player aggregates. |
| `backend/player_database.py` | Opponent memory | SQLite schema and indexed `player_id` retrieval. Stores hands, VPIP/PFR counters, calls, raises, AF inputs, and pot totals. |
| `backend/input_stream.py` | Video/CNN boundary | Keeps the newest preview data URL and ignores duplicate external action IDs. It knows nothing about a specific CNN. |
| `backend/engine/hand_evaluator.py` | Hand-ranking adapter | Validates cards and wraps `phevaluator`, the HenryRLee perfect-hash hand evaluator. |
| `backend/engine/cfr_solver.py` | Quantitative decision model | Builds opponent ranges, estimates equity by sampling legal runouts, applies rake in cash games, then uses regret matching across legal choices. |
| `backend/engine/plo5.py` | PLO5 baseline | Computes exact-two-hole/exact-three-board hand ranks and samples heads-up equity; it is not a PLO5 equilibrium solver. |
| `backend/engine/open_study.py` | Open-data study reader | Validates the pinned artifact and exact game assumptions, then looks up HU NLH shove/call frequencies by hand class and stack depth. |
| `backend/engine/texas_solver_study.py` | Native export study adapter | Validates one TexasSolver node and its solve manifest; exact-match combo queries never enter the live decision path. |
| `backend/engine/opponent_adjustment.py` | Contextual player-read overlay | Summarizes trusted completed-hand JSON actions, Bayesian-shrinks faced-size fold rates, and caps a study-only policy mixture at 35% player weight. |
| `backend/engine/research_frameworks.py` | Optional CFR lab | Runs OpenSpiel CFR+ on Kuhn or RLCard CFR on Leduc for offline algorithm checks only. |
| `backend/engine/tournament.py` | Tournament payout model | Computes exact small-field Independent Chip Model equity with a subset dynamic program. |
| `backend/engine/coach.py` | Explanation layer | Converts live state, opponent memory, and solver result into readable coaching notes and practice prompts. |
| `backend/engine/recommend.py` | Lightweight fallback | Existing simple rule-based recommendation path used when full coaching is not requested. |
| `backend/speech/action_parser.py` | Speech parser | Parses phrases such as “Alex raises to 10” into structured actions. |
| `backend/vision/camera.py` | Local camera placeholder | Basic OpenCV availability/capture interface; it is not a production card detector. |
| `backend/vision/variant.py` | Variant cue | Conservatively separates green NLH from blue PLO5 felt in a single shared window, with three-frame agreement. |
| `backend/vision/clubgg_holdem.py` | Calibrated NLH reader | Uses local RapidOCR for card ranks/text plus suit-shape masks from the supplied recording; reads visible cards, pot, seat names, and an opened profile panel, abstaining outside the calibrated layout. |
| `backend/play_recordings.py` | Local recording store | Appends opt-in WebM chunks, samples JPEG frames, keeps confirmed action events, and stores manually verified labels under `data/play/`. |
| `backend/vision/play_training.py` | Post-game calibration | Builds extra suit masks from verified frame labels, checks held-out frames, and promotes only a no-new-error improvement. Raw video or OCR guesses alone are never training truth. |
| `frontend/play.html`, `frontend/play.js`, `frontend/play.css` | Play review | Lists local sessions, plays WebM, shows sampled frames and predictions, saves verified labels, and runs the calibration gate. |
| `frontend/index.html`, `frontend/focus.js`, `frontend/focus.css`, `frontend/focus-layout.css` | Focused live view | Black table preview, viewport-contained chat, top Setup overlay, game toggles, optional spoken replies, and local window-frame transport. |
| `frontend/advanced.html`, `frontend/app.js`, `frontend/styles.css`, `frontend/advanced-theme.css` | Detailed dashboard | Seat and format setup, corrections, review queue, hand archive, profiles, and on-demand coaching; dark theme matched to the live view. |
| `tests/` | Regression coverage | Validates API flows, action parsing, database aggregation, input-event idempotency, and coaching responses. |
| `data/` | Runtime and study data | Contains current state, session log, hand archive, and `player_stats.sqlite3`; `data/open_study/` is a separately attributed immutable third-party asset. |

## Input and event model

### Generic input contract

The system accepts `POST /api/input/frame`. A source may submit any combination
of a latest visual frame and structured observations:

```json
{
  "source_id": "my-cnn",
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
        "event_id": "table-7-action-42",
        "player": "Alex",
        "action": "raise",
        "amount": 6,
        "amount_type": "total",
        "street": "preflop"
      }
    ]
  }
}
```

The `event_id` is crucial. Vision models usually observe the same action across
many frames; the input hub records each ID once, preventing a detected raise
from being appended repeatedly.

### Confidence and correction workflow

`InputObservation` accepts both an overall `confidence` and per-field
`field_confidences`. The current auto-apply threshold is 85%.

- High-confidence fields update the live state.
- Low-confidence cards, pot values, street, actor, active-player list, or
  actions enter `GameState.review_queue` instead.
- The dashboard exposes each item with **Apply** and **Dismiss** controls.
- Applying an action uses the same action ledger as a manual entry; dismissing
  it leaves the hand untouched so the user can enter a correction manually.

This prevents one uncertain OCR/camera frame from silently changing the pot or
creating a false raise. Every queued/approved/dismissed item is logged locally.

### Current implementation versus future vision

The browser can already produce a preview stream using a camera or screen
share. It uploads compressed JPEG frames approximately every 1.2 seconds.

The browser uploads JPEGs (up to 1600 pixels wide). The reader accepts the
calibrated game-window shape directly or isolates one green ClubGG table from
a larger screen share; it abstains if multiple candidates are visible.
The calibrated ClubGG NLH reader uses local RapidOCR for rank/pot/name/profile
text and small suit-shape masks for cards. It recognizes the supplied table
layout only, then requires two identical reads before cards, pot, or
displayed profile values update state. The API returns a `vision` object with
read/missing fields for visible feedback. OCR runs off the FastAPI event loop.
It never emits betting actions or marks a complete verified spot merely from
pixels: history, amount to call, hidden information, aliases, and obscured
fields need confirmation. Other layouts, including PLO, retain the generic
observation seam for a future adapter; manual corrections remain available.

The live page can also start an explicit, visible MediaRecorder session for the
single shared window. Two-second WebM chunks are appended locally; sampled
JPEGs and reader predictions are retained every few seconds for post-game
review. Confirmed actions entered through the app are timestamped in the same
session. Training currently affects **suit recognition only** and requires
independently verified cards. No betting-action model is fitted from unlabeled
video, and no generated recommendation is treated as a target label.

The contract also accepts `game_format`, `game_variant`, and a partial `tournament` object, such
as `{"players_remaining": 7}`. A partial update preserves saved payouts and
other fields. The browser's Stop Input control clears its own preview and
does not erase a newer feed from another source.

## State and opponent memory

### Live state

`GameState` is the current-hand source of truth. It includes:

- seats, stacks, blind configuration, and the Hero seat;
- street, pot, betting level, street contributions, and action history;
- hero cards and board cards;
- optional vision-observed pot, current actor, active players, and confidence;
- VPIP/PFR flags for the current hand.

The service recomputes stacks, contributions, and pot from recorded actions.
An observed pot from a camera is kept separately so the UI can surface the
vision value without overwriting the accounting trail.

It also calculates these real-time truth-layer fields:

- dealer-based positions (`BTN`, `SB`, `BB`, `UTG`, `CO`, and related labels);
- active and folded players from the action ledger;
- Hero's price to call and legal action set;
- per-player total contribution across all streets;
- a side-pot summary based on contribution levels, including contributors and
  eligible non-folded players.

The current side-pot output is a correct contribution/eligibility summary. A
future payout allocator still needs explicit showdown-hand and winner-by-pot
input before it can distribute each side pot automatically.

### Persistent player database

`data/player_stats.sqlite3` stores hand and action aggregates at hand end, so
corrections made while the hand is live cannot pollute those aggregates. A
displayed ClubGG profile snapshot is saved immediately in a separate table.

| Table | Granularity | Purpose |
| --- | --- | --- |
| `player_stats` | Player lifetime | Fast global profile used when no useful contextual sample exists. |
| `player_table_stats` | Player × table-size bucket | Fast contextual profile for heads-up, short-handed, middle, and full-ring games. |
| `player_hand_records` | Player × completed hand | One row per seated player with stack, contribution, result, street, and action-count context. |
| `player_action_records` | Individual action | Every call, raise, bet, fold, check, or all-in with amount, contribution, street, source, confidence, and timestamp. |
| `player_aliases` | Confirmed detector spelling | Maps a human-approved alternate spelling, such as `Fantamj`, to a canonical local player, such as `Fanta`. |
| `player_observed_stats` | Visible profile snapshot | Keeps displayed hand count, VPIP, PFR, AF, confidence, source, and capture time apart from locally recorded hands. |

`player_id` is the primary key for lifetime lookup; contextual and history
tables have player/bucket and player/hand indexes. At hand end, the system
inserts raw hand/action records and updates aggregate rows in one transaction.

| Metric | Definition |
| --- | --- |
| VPIP | `vpip_count / hands_seen`: voluntarily put money in preflop |
| PFR | `pfr_count / hands_seen`: raised preflop |
| Calls | Total calls across completed hands |
| Raises | Total raises/bets/all-ins across completed hands |
| AF | `aggressive_actions / call_count` (uses aggressive count if calls are zero) |

The contextual table-size buckets are:

| Active players represented in the hand | Bucket |
| --- | --- |
| 1–2 | `1-2` |
| 3–4 | `3-4` |
| 5–6 | `5-6` |
| 7–9 or more | `7-9` |

For each bucket, the app separately tracks hands, VPIP, PFR, pot totals,
contributions, calls, raises, bets, folds, checks, all-ins, aggressive actions,
showdowns, wins, and total action count. The raw rows preserve the full action
timeline so richer statistics can be computed later without losing history.
Each per-hand row also stores table position, preflop context (`limp`, `open`,
`call_raise`, `three_bet`, or `four_bet`), and effective stack depth in big
blinds. Each hand and action also stores game format, game variant, and
tournament phase. The coach reads only same-variant hand rows and displayed
baselines, so NLH and PLO5 observations for one username are not blended.
The phase endpoint aggregates a username's measured cash, tournament, bubble,
and final-table behavior. A phase sample changes the range prior only after ten
completed hands in that phase; otherwise the broader profile is used.

### Player-identity safety gate

Vision actions resolve names in this order: exact seated-player match, an exact
human-confirmed local alias, then a fuzzy match against currently seated names.
A fuzzy match never writes an action automatically. For example, when a
detector sees `Fantamj raises $30` and the table has `Fanta`, the review queue
asks whether they are the same person. Approval stores `Fantamj → Fanta` in
`player_aliases` and applies the raise to Fanta's history; the alias then works
automatically in later games.

An unknown/missing player, or a raise/bet with no recognized amount, is queued
as ambiguous instead of being written to the action ledger.

The update occurs after a hand ends, not after each detected frame. This means
manual corrections and action undo remain safe during the hand.

## Decision model: what the solver actually does

The application does **not** claim to solve the complete game of no-limit
Hold'em. A complete game-tree equilibrium requires massive abstraction,
billions of states, stack-dependent bet sizing, and iterative solving far
beyond an on-demand web request.

Instead, this implementation is a **bounded one-decision abstraction**. For a
known Hero hand, board, pot, price to call, raise size, and selected opponent,
it estimates which of `fold`, `call`, or `raise` is strongest in the current
modeled spot.

### Step 1: Construct a Bayesian opponent range

The database gives observed counts, but raw percentages are noisy. A player who
plays 1 of 2 hands should not be treated as a true 50% VPIP player. The solver
uses a Beta-binomial posterior mean with population priors:

```text
posterior VPIP = (observed_vpip_count + 20 × 0.25) / (hands_seen + 20)
posterior PFR  = (observed_pfr_count  + 20 × 0.18) / (hands_seen + 20)
```

The `20` is the prior strength: before enough history exists, the model leans
toward 25% VPIP and 18% PFR. As hands accumulate, the player's own data
dominates.

Every legal two-card combination is then assigned a relative weight based on:

- rank strength, pairs, and suitedness;
- posterior VPIP — wider VPIP increases weak-combo probability;
- posterior PFR and AF — aggressive profiles receive a modest strength/continue
  adjustment;
- an explicit loose-player adjustment that increases low-quality holdings,
  including 72o and J4s when VPIP is high.
- opponent position — early positions start tighter; cutoff/button positions
  begin wider;
- preflop sequence — limps are wider, while calls of raises, 3-bets, and 4-bets
  receive stronger priors;
- effective stack depth — short stacks concentrate more weight in strong cards
  and pairs;
- table size — multiway spots tighten the continuing-range prior.

Weights are normalized to sum to one. The solver uses the selected opponent's
same-variant stage sample once it has ten local hands; otherwise it uses that
variant's full local/displayed profile. NLH and PLO5 observations never mix.
This is a probability distribution over legal opponent hole-card combinations,
not a hard “range chart.”

### Step 2: Estimate equity by weighted Monte Carlo simulation

For each sample, the solver:

1. draws a legal opponent combination from the weighted range;
2. draws any missing turn/river cards without replacement;
3. ranks Hero and opponent seven-card hands;
4. counts a win, tie, or loss.

The equity estimator is:

```text
equity = (wins + 0.5 × ties) / total_samples
```

Hand rankings use `phevaluator`, a Python distribution of HenryRLee’s
perfect-hash evaluator. Perfect-hash evaluation is efficient for one known
5–7-card hand, while complete spot equity remains slower because it must
evaluate many sampled opponent/runout pairs.

The default is 3,000 samples, bounded to 100–20,000. The response returns its
measured `latency_ms`; on the development machine, a representative 3,000
sample flop calculation was around 141 ms. That is a measurement, not a
hardware-independent promise.

### Step 3: Build action utilities

The model assigns a simplified expected value to each action:

```text
fold utility = 0

call utility = equity × pot − (1 − equity) × amount_to_call

raise utility = fold_equity × pot
              + (1 − fold_equity) × [equity × (pot + call_cost)
                                      − (1 − equity) × raise_cost]
```

Fold equity is heuristic rather than observed directly. It increases with
raise size relative to the pot, and decreases against loose or high-AF
opponents. This is one of the major assumptions to calibrate from your own
home-game histories.

Cash mode subtracts estimated rake from winning branches using a configured
percentage and cap. The no-flop-no-drop option omits rake on a preflop raise
that ends the hand. The estimate does not reconstruct every site's rake rules.

Tournament mode labels the state as regular, bubble, or final table. When the
entire remaining final table and all paid prizes are entered, it computes ICM
expected payout for each simplified fold/call/raise outcome. For an ordering
of remaining players, the next finishing-place probability is proportional to
each stack's share of remaining chips. A subset dynamic program sums those
probabilities over all orders in at most nine players. This produces prize EV
and a loss-to-gain risk ratio, so a call that gains chip EV can still lose prize
EV near payout jumps. Equal ticket prizes model a satellite. Missing stacks,
prizes, or a field larger than the configured final table cause an explicit
chip-EV fallback. ICM itself does not predict an opponent's response to an
elimination; the phase-specific observed history informs their range only when
there is a usable sample.

### Step 4: Regret matching

Counterfactual Regret Minimization normally updates a strategy at many
information sets throughout a large game tree. Here it is applied to one
decision node with three actions.

For each iteration:

```text
strategy = normalized positive regrets
node value = strategy · action_utilities
regret[action] += action_utility − node_value
strategy_sum += strategy
```

The returned action frequencies are:

```text
average strategy = strategy_sum / sum(strategy_sum)
```

With fixed action utilities, regret matching naturally concentrates on the
best modeled action. The important output is not a guaranteed poker
equilibrium; it is an explicit, inspectable mixture created from your selected
range, equity estimate, pot odds, and raise assumption.

### PLO5 coaching boundary

PLO5 does not enter the Hold'em CFR path. The separate PLO5 estimator draws a
legal five-card opponent hand and missing board cards, then compares the best
rank using exactly two hole cards and three board cards. It is capped at 200
samples per coaching request and reports actual elapsed time. The opponent is
uniformly random, so this is a learning baseline, not a calibrated player
range or pot-limit equilibrium. In a multiway spot, tournament, or close
pot-odds comparison, the coach declines to recommend an action. It never
models a PLO5 raise and does not yet subtract rake. A proper PLO5 range,
multiway equity, pot-limit sizing, and payout model remain future work. It
only names a check/call/fold candidate on a heads-up cash-game river with a
wide equity margin; on earlier streets, raw equity is shown for study only.

### Open solver study data boundary

The bundled [MIT-licensed study artifact](data/open_study/README.md) contains
eight solved *simplified* heads-up NLH push/fold games at exactly 2, 3, 5, 8,
10, 12, 15, and 20bb. A checked SHA-256 and 169-hand-class validation run on
first load. The read-only API returns the source quality flag, model-game
exploitability, and action-value error bound. It performs no nearest-stack
matching or interpolation. The assumed game has equal stacks, 0.5/1 blinds,
no ante/rake/ICM, and only SB shove/fold plus BB call/fold decisions. Monte
Carlo equities and a restricted betting tree mean this is not full-game GTO.
The catalog is absent from the live coaching and player-memory paths. No
qualified open PLO5/PLO6 strategy dataset was found or bundled; PLO6 itself
is not implemented in this MVP.

### Optional external solver frameworks

`texas_solver_study.py` accepts native TexasSolver action-node exports with an
operator-supplied solve manifest. Import validates action vectors and legal
combos, records source SHA-256, and writes a local exact-spot study record.
`POST /api/study/texas-solver` returns a combo's native frequencies only if
board, pot, price, stack, rake, position, and action path all match. It is
post-hand study, not a live solve. The manifest's range assumptions still
need human verification; metadata cannot be inferred from the export alone.

`research_frameworks.py` provides optional OpenSpiel CFR+ on Kuhn (with
reported exploitability) and RLCard chance-sampling CFR on Leduc. They are
useful for checking an algorithm workflow but cannot transfer a policy to
no-limit Texas Hold'em. None of the three frameworks is treated as a
drop-in cure for incomplete vision or missing action history.

The opponent overlay uses the saved `hand_archive.json` action ledger, now
including pot-before and price-to-call-before values. Evidence is partitioned
by format, variant, phase, and table-size bucket; old records lacking these
fields do not silently contribute. It measures how often the player folded,
called, or raised when facing a bet, including conditional price-size buckets.
It separately reports their own small/medium/large aggressive-size mix.
Posterior fold rates use a population prior; a minimum of 12 distinct facing
hands is required for a nonzero blend, and weight never exceeds 35%. Rare
current bet sizes *reduce* confidence; they do not imply specific hidden
cards. No showdown-strength tell is claimed. The imported TexasSolver
distribution stays separately visible from the approximate player model and
the study-only blended frequencies.

## What the coach thinks about

The coach deliberately separates facts, model implications, and caution:

1. **What I see** — Hero cards, board, pot, price, player count, and the
   selected opponent profile.
2. **Why this matters** — estimated equity, number of samples, fold/call/raise
   frequencies, fold-equity assumption, and selected raise size.
3. **Watch out for** — missing information, vision uncertainty, small samples,
   loose or aggressive opponent profiles, and the risk of treating a model as
   certainty.
4. **Next repetition** — a post-hand learning prompt: compare the actual line
   or showdown to the assumed range and update the read.

No large language model is required for the numerical coach brief. Specific
move suggestions and core explanations remain deterministic templates tied to
the same numerical values that produced the suggested line. The configured
Azure `gpt-6-luna` model or optional local 1B-parameter model handles
open-ended conversation and follow-up teaching.
It is never used to enter actions, change game state, invent observed facts, or
replace the quantitative decision model. If the model is missing, the API
reports that honestly and the rule-based spot path stays available.

The browser’s **Read Aloud** button uses the local browser Speech Synthesis API
to speak the coaching brief. It does not upload audio to a third party.

## API surface

| Endpoint | Purpose |
| --- | --- |
| `GET /api/state` | Current live hand state |
| `POST /api/config/seats` | Seats, stacks, blinds, Hero |
| `POST /api/config/game-format` | NLH/PLO5 variant plus cash rake or tournament context |
| `POST /api/action` | Manual structured action |
| `POST /api/correction` | Correct or undo the latest action |
| `POST /api/speech/transcript` | Parse a spoken-action transcript |
| `POST /api/vision/card_manual` | Manual cards endpoint retained for compatibility |
| `POST /api/input/frame` | Generic browser/camera/CNN frame and observation |
| `GET /api/input/latest` | Latest in-memory preview/status |
| `POST /api/input/stop` | Clear a stopped source's preview if still current |
| `GET /api/reviews` | Low-confidence vision observations awaiting a decision |
| `POST /api/reviews/{review_id}` | Apply or dismiss one review item |
| `GET /api/players` | All persistent player profiles |
| `GET /api/players/{player_id}` | One persistent player profile |
| `GET /api/players/{player_id}/table-breakdown` | Player aggregates by `1-2`, `3-4`, `5-6`, and `7-9` table-size buckets |
| `GET /api/players/{player_id}/history` | Recent normalized hand and action records, for review/calibration |
| `GET /api/players/{player_id}/phase-breakdown` | Stage-specific measured behavior for a username |
| `POST /api/solver/recommend` | Raw quantitative solver response |
| `POST /api/coach/chat` | Parsed spot facts or bounded local-model conversation with recent history |
| `GET /api/coach/model-status` | Whether the selected conversation model is configured and verified |
| `POST /api/coach/connect-azure` | Same-origin, loopback-only key entry; tests Azure and holds the key only in server memory |
| `GET /api/study/push-fold` | Exact available stack depths and post-hand study provenance |
| `POST /api/study/push-fold` | One HU NLH push/fold hand-class lookup, outside the live coach |
| `POST /api/study/texas-solver` | Exact-match imported HU NLH postflop study lookup; 404 if absent |
| `POST /api/study/texas-solver/adjusted` | Exact-tree baseline plus bounded player-read blend and evidence |
| `GET /api/players/{player_id}/read` | Contextual completed-hand JSON action and sizing summary |
| `POST /api/coach/brief` | Explanation-first coaching response |
| `POST /api/hand/end` | Archive and commit per-player aggregates |
| `GET /ws` | Real-time state updates to the browser |

## Performance and scaling plan

### Current behavior

- Browser preview capture is throttled to roughly 1.2 seconds.
- Input memory retains only one preview frame, preventing unbounded frame
  storage.
- Player lookup is an indexed SQLite query.
- Solver work is bounded by sample and iteration caps.
- NumPy manages probability vectors and regret operations; the evaluator ranks
  individual sampled hands.

### Practical next engineering steps

1. **Vision generalization** — collect consented labeled examples of other
   window sizes/themes and action states; measure accuracy before enabling
   automatic action logging. Keep manual corrections prominent.
2. **Hand lifecycle** — detect new hands, blinds, folded players, all-ins, side
   pots, and showdown automatically.
3. **Data quality** — store confidence and human correction outcomes per
   observed field, then measure card/action-recognition accuracy.
4. **Range calibration** — compare predicted ranges to showdown hands; fit the
   loose/aggressive weighting coefficients using your own logged data.
5. **Strategy abstraction** — add position, effective stack, legal bet sizes,
   multiway ranges, and street-specific action models. A true larger CFR engine
   should run asynchronously or from pre-solved tables, not inside a short HTTP
   request.
6. **Operational hardening** — add migration tooling, database backups,
   structured logs, health endpoints, authentication if the app is ever exposed
   beyond localhost, and privacy retention controls.

## Developer workflow

```powershell
cd poker-coach-mvp
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Run regression tests:

```powershell
python -m pytest -q
```

The baseline test suite covers API flow, parser behavior, player-stat updates,
generic input de-duplication, and a coach brief response.

## Current limits and decision-risk posture

- The system supports a single opponent range in the quantitative solver;
  multiway hands can be displayed in context but are not yet solved as a true
  multi-opponent equity/range model.
- Player identity is a manually supplied familiar name; there is no facial
  recognition.
- Chip counting, side-pot resolution, legal bet-sizing enforcement, and real
  CNN recognition remain future work.
- The raise model uses a heuristic fold-equity estimate. It requires empirical
  calibration before being treated as anything more than a learning aid.
- Every input may be wrong. The intended workflow is observe → review → request
  coaching → decide, with visible context and correction controls.

This posture is deliberate: fast, local, explainable, and extensible is more
valuable than an opaque system that appears more certain than its data.
