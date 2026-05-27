# Current Handoff

## Active Objective
- Publish-prep the Lostless node pack for the Comfy Registry and ComfyUI-Manager.

## Current Repo Reality
- Branch: `main`.
- Runtime entrypoints remain `__init__.py`, `nodes.py`, and `web/js/lostless_nodes.js`.
- Root packaging now uses Comfy Registry metadata in `pyproject.toml`, root `requirements.txt`, root `LICENSE`, and `.comfyignore`.
- Publish assets live under `assets/registry/`.
- A GitHub publish workflow is expected under `.github/workflows/publish_action.yml`.

## Decisions That Matter Right Now
- Decision: the Comfy Registry-facing package id is `lostless-nodes`, while the Git repository and manual install folder remain `comfyui-lostless-nodes`.
- Decision: root `requirements.txt` is the install-facing dependency surface for ComfyUI/Manager and must stay aligned with the embedded editor requirements.
- Decision: internal docs, smoke evidence, and embedded upstream example workflows are kept in git for development continuity but excluded from the published archive via `.comfyignore`.

## Verification State
- Passed: none yet in this handoff.
- Still needed: `python3 -m compileall .`, `node --check web/js/lostless_nodes.js`, and a local ComfyUI startup smoke using the publish-prep tree.

## Next Steps
1. Run syntax checks for Python and JS.
2. Run a local ComfyUI startup smoke and one random-image broadcast-lock smoke.
3. Create the `lostlessvisuals` publisher and API key on the Comfy Registry if it does not exist yet.
4. Push the repo, then publish the new version through `comfy node publish` or the GitHub action after confirming assets resolve from `main`.

## Risks Or Blockers
- The publish workflow still depends on the Registry publisher id being available as `lostlessvisuals`; change `PublisherId` in `pyproject.toml` if the actual Registry handle differs.
- No automated ComfyUI integration test covers the browser-side widget flows, so runtime confirmation is still manual.

## Pointers
- Stable spec / install doc: `README.md`.
- Publish metadata: `pyproject.toml`, `requirements.txt`, `.comfyignore`.
- Supporting work log / durable lessons: `docs/LEARNINGS.md`.
