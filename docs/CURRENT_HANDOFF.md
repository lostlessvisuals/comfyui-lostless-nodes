# Current Handoff

## Active Objective
- Add a per-node lock for `Lostless Random Image` so broadcast randomization can skip selected nodes while others keep randomizing.

## Current Repo Reality
- Branch: `main`.
- Random-image behavior lives in `nodes.py` and `web/js/lostless_nodes.js`.
- `README.md` documents the random image and randomize button flow.
- Implemented but not fully verified yet: `lock_randomize` boolean input on `Lostless Random Image`, with frontend broadcast skip logic for locked nodes.

## Decisions That Matter Right Now
- Decision: lock applies only to broadcast (`Lostless Randomize Button`) randomization, not to manual `Randomize Image` or `Load Image` on each node.

## Verification State
- Passed: none yet in this session.
- Still needed: `python3 -m compileall .`, `node --check web/js/lostless_nodes.js`, and a ComfyUI smoke confirming one locked node stays fixed while unlocked nodes randomize.

## Next Steps
1. Run syntax checks for Python and JS.
2. Run a local ComfyUI smoke with one button wired to multiple random-image nodes and one lock enabled.
3. Commit and push after verification.

## Risks Or Blockers
- No automated UI integration test covers broadcast randomization lock behavior; manual ComfyUI smoke is needed for runtime confirmation.

## Pointers
- Stable spec / architecture doc: `README.md`.
- Supporting work log / status doc: `docs/LEARNINGS.md`.
- Related artifact or handoff: this file.
