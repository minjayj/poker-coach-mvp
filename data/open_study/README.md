# Open study data

`hu_push_fold.json` is a verbatim copy of
[`data/preflop/solved-scenarios.json`](https://github.com/davidvayn/pokersolver/blob/0036c954f411984e87298e75ed63d7e5554bcf83/data/preflop/solved-scenarios.json)
from David Vayntrub's `pokersolver`, commit
`0036c954f411984e87298e75ed63d7e5554bcf83` (2026-09-28).

- License: MIT. See [LICENSE-MIT.txt](LICENSE-MIT.txt).
- SHA-256 of the copied JSON: `6fb89ce7b5ca3919480570f6e14188fb95e400b1dd5fade32d6d01d9d743b59c`.
- Contents: eight approximate heads-up NLH push/fold models at exactly 2, 3, 5, 8, 10, 12, 15, and 20 big blinds. Each has 169 hand classes for SB shove and BB call-versus-shove, with model-estimated action values.
- Assumptions: equal effective stacks, 0.5/1 blinds, no ante, no rake, no other first action besides SB fold or shove. These are **not** full-game GTO, cash-game charts, multiway ranges, or ICM solutions. The source uses Monte Carlo showdown equities and reports model-game exploitability, not exact real-game exploitability. The upstream `quality` flag is preserved in each result.
- Use: post-hand study only. This file is deliberately not connected to live capture, opponent profiles, or automatic coaching recommendations.

No PLO5 or PLO6 strategy data is bundled. Omaha equity and hand evaluation must not be labeled GTO.
