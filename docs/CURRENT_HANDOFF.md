# Current Handoff

## Active Objective
- Publish-prep the Lostless node pack for the Comfy Registry and ComfyUI-Manager.

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

## Verification State
- Passed: `python3 -m compileall .`, browser-JS parse checks for `web/js/lostless_nodes.js` and `Lostless-Mask-Editor/web/js/wan_mask_editor.js`, registry asset validation, initial Registry publish of `lostless-nodes@0.2.0`, YAML validation for `.github/workflows/publish_action.yml`, and removal of the accidental `Lostless VACE Strength Schedule` node from code plus publish-facing copy.
- Still needed: a local ComfyUI startup smoke using the five-node surface and the next publish pass for `0.2.1`.

## Next Steps
1. Run a local ComfyUI startup smoke and one random-image broadcast-lock smoke.
2. Publish `lostless-nodes@0.2.1` so the Registry matches the intended five-node surface.
3. Use the direct GitHub Actions workflow or local `comfy node publish` path for future version bumps.

## Risks Or Blockers
- The publish workflow still depends on the Registry publisher id being available as `lostlessvisuals`; change `PublisherId` in `pyproject.toml` if the actual Registry handle differs.
- No automated ComfyUI integration test covers the browser-side widget flows, so runtime confirmation is still manual.

## Pointers
- Stable spec / install doc: `README.md`.
- Publish metadata: `pyproject.toml`, `requirements.txt`, `.comfyignore`.
- Supporting work log / durable lessons: `docs/LEARNINGS.md`.
