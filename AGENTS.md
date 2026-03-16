# ComfyUI Lostless Nodes

Custom ComfyUI nodes focused on random image selection, sequence buffering, and mask editing.

## Constraints
- Keep the package compatible with ComfyUI custom-node loading.
- Be careful around the embedded `Lostless-Mask-Editor` boundary and avoid accidental upstream-wide rewrites.
- Prefer targeted validation over sweeping refactors.

## How To Work
- Read `README.md`, `docs/LEARNINGS.md`, and the embedded `Lostless-Mask-Editor/README.md` when work touches the mask-editor surface.
- Keep package-local lessons in `docs/LEARNINGS.md`.
- Record cross-project or ComfyUI-wide patterns in shared guidance instead of duplicating them here.
- Inherit the package-workspace skill inventory from `/Volumes/T7/Dropbox/Codex/packages/AGENTS.md` unless this project explicitly needs a narrower set.

## Build / Run / Test
- Python syntax smoke after touching Python modules: `python3 -m compileall .`
- Dependency refresh when embedded editor requirements change: `python3 -m pip install -r Lostless-Mask-Editor/requirements.txt`
- Prefer a local ComfyUI startup smoke when node registration or import paths change.

## Coding Conventions
- Keep node registration explicit and easy to diff.
- Minimize surprises in filesystem, subprocess, or browser-launch behavior.
