# Deferred Carry Smoke Result Template

Use with `docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE.md` after deferred-carry edits in `Lostless-Mask-Editor/nodes/mask_editor.py`.
Legacy alias (redirect only): `docs/MASK_EDITOR_DEFERRED_CARRY_RESULTS_TEMPLATE.md`.

- Date/time: 2026-09-04 17:24:41 PDT
- Branch/commit: main @ eae4ccf
- Editor build/version: r2026.09.04.1 against eae4ccf; Python 3.13 / PyQt5 5.15.11 / Qt 5.15.19 / OpenCV 4.10.0 / NumPy 2.2.6
- Edited file paths: Lostless-Mask-Editor/nodes/mask_editor.py

## End-Path Results

1. Key release flush
- Result: blocked
- Notes: Manual ComfyUI UI/device validation unavailable on this machine; run this checklist on the target host and update this artifact with pass/fail outcomes.

2. Click-jump boundary
- Result: blocked
- Notes: Manual ComfyUI UI/device validation unavailable on this machine; run this checklist on the target host and update this artifact with pass/fail outcomes.

3. Mode/tool switch boundary
- Result: blocked
- Notes: Manual ComfyUI UI/device validation unavailable on this machine; run this checklist on the target host and update this artifact with pass/fail outcomes.

4. Direction reversal boundary
- Result: blocked
- Notes: Manual ComfyUI UI/device validation unavailable on this machine; run this checklist on the target host and update this artifact with pass/fail outcomes.

## Outcome

- Ambiguous behavior observed: unknown
- Follow-up required: Execute docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE.md on the ComfyUI host and replace each blocked result with pass/fail plus notes.

## Automated Qt evidence

`test_deferred_carry_end_paths_apply_once_and_stay_stable_on_idle` passes all five subcases (key release, click jump, mode switch, tool switch, direction reversal) using real Qt mouse/key events at 1× and 2× scaling. Traversed masks contain the stroke, untouched frames remain blank, and repeated release/idle does not add changes or undo entries. The complete 23-test suite also exercises the real launcher and mask export.

The blocked statuses above refer specifically to the required live Windows ComfyUI workflow check. This session ran on Porters-MacBook-Pro.local in an isolated offscreen Qt environment; no Windows desktop or live ComfyUI workflow was exercised. Rerun the canonical checklist on the ComfyUI machine before release.

Runtime edits: `Lostless-Mask-Editor/__init__.py`, `Lostless-Mask-Editor/nodes/mask_editor.py`, `Lostless-Mask-Editor/nodes/comfyui_mask_launcher.py`. Full scope and limits: `docs/MASK_EDITOR_REVIEW_2026-09-04.md`.

Follow-up before the authorized push: the expanded 23-test suite covers explicit raster conversion and undo, exact raster launcher export/autosave, and preservation of geometry in Reuse Last Edit. Live Windows ComfyUI checklist remains outstanding; automated Windows/Linux CI is prepared locally but excluded from the push because the configured GitHub token lacks workflow scope.
