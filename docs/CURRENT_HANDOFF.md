# Current Handoff

## Active Objective
- Review and harden mask point rendering, editing, and session restore. The user authorized follow-up improvements and a GitHub push. Combined fixes are ready to push; live ComfyUI validation remains a documented caveat.

## Current Repo Reality
- Branch: `main`.
- Runtime entrypoints remain `__init__.py`, `nodes.py`, and `web/js/lostless_nodes.js`.
- The intended public node surface is five nodes: `Lostless Random Image`, `Lostless Randomize Button`, `Lostless Buffer`, `Lostless Mask Editor`, and `Lostless Image To Mask`.
- Root packaging now uses Comfy Registry metadata in `pyproject.toml`, root `requirements.txt`, root `LICENSE`, and `.comfyignore`.
- Publish assets live under `assets/registry/`.
- The GitHub publish workflow under `.github/workflows/publish_action.yml` now uses direct `comfy-cli` publishing with `actions/checkout@v6` and `actions/setup-python@v6`, rather than the `Comfy-Org/publish-node-action` wrapper.

## Decisions That Matter Right Now
- Decision: the Comfy Registry-facing package id is `lostless-nodes`, while the Git repository and manual install folder remain `comfyui-lostless-nodes`.
- Decision: root `requirements.txt` is the install-facing dependency surface for ComfyUI/Manager and must stay aligned with the embedded editor requirements.
- Decision: internal docs, smoke evidence, and embedded upstream example workflows are kept in git for development continuity but excluded from the published archive via `.comfyignore`.

## Mask Review — 2026-09-04
- Combined fixes: Shape default, visible mode indicator, debug-render import, direct point dragging, cache/undo correctness, zoom-aware picking, mode/frame restore, and blank-frame/vector-timeline preservation.
- Added `tests/test_mask_editor.py` and `tests/launcher_smoke.py`; 23 tests pass with normal and 2× Qt scaling on the MacBook. Python compilation and diff checks pass.
- Full findings and remaining raster-fidelity limitations: `docs/MASK_EDITOR_REVIEW_2026-09-04.md`.
- The four-path live ComfyUI smoke remains pending on Windows; automated Qt carry subcases pass. Evidence: `docs/evidence/deferred-carry/deferred-carry-smoke-2026-09-04-mask-review.md`.
- Follow-up: imported pixel masks now remain intact until explicit, confirmed, undoable Create Points conversion. Autosave retains pixel-mask data. Reuse Last Edit retains original editable points/settings alongside pixels. Windows/Linux regression CI is prepared locally; the configured GitHub token lacks workflow scope, so its file is excluded from the code push.
- GitHub push is authorized; no version bump or Registry publication is included.

## Verification State
- Passed: `python3 -m compileall .`, browser-JS parse checks for `web/js/lostless_nodes.js` and `Lostless-Mask-Editor/web/js/wan_mask_editor.js`, registry asset validation, initial Registry publish of `lostless-nodes@0.2.0`, YAML validation for `.github/workflows/publish_action.yml`, and removal of the accidental `Lostless VACE Strength Schedule` node from code plus publish-facing copy.
- Still needed: a local ComfyUI startup smoke using the five-node surface and the next publish pass for `0.2.1`.

## Next Steps
1. Run a live Windows ComfyUI smoke: new Shape mask, point drag/undo, legacy reopen, exported blank frames, and the deferred-carry checklist.
2. Install the prepared regression workflow when GitHub credentials permit workflow writes; the local branch `codex/mask-editor-ci-20260904` preserves the CI candidate.
3. Obtain separate authorization for any Registry release/version bump. Prior release-prep checks for the other four nodes remain applicable.

## Risks Or Blockers
- The publish workflow still depends on the Registry publisher id being available as `lostlessvisuals`; change `PublisherId` in `pyproject.toml` if the actual Registry handle differs.
- No automated ComfyUI integration test covers the browser-side widget flows, so runtime confirmation is still manual.

## Pointers
- Stable spec / install doc: `README.md`.
- Publish metadata: `pyproject.toml`, `requirements.txt`, `.comfyignore`.
- Supporting work log / durable lessons: `docs/LEARNINGS.md`.
