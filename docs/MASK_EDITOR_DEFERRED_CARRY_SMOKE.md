# Deferred Carry Smoke Pass (Mask Editor)

Use this checklist after any deferred carry change in `Lostless-Mask-Editor/nodes/mask_editor.py`.

## Goal

Confirm deferred carry sessions end intentionally for each navigation end-path and do not reinterpret pending carry when tool or mode changes mid-run.

## Preconditions

- Open a workflow with a multi-frame sequence and `Lostless Mask Editor`.
- Start on a frame with visible mask content.
- Use brush mode with a clearly visible stroke color/shape.

## End-Path Checks

1. Key release flush
- Paint a stroke, hold right arrow for at least 2 frame moves, then release the arrow key.
- Expected: queued carry applies once across traversed frames and no extra apply happens on idle.

2. Click-jump boundary
- Start another carry run with arrow navigation, then click a non-adjacent frame thumbnail before key release.
- Expected: carry session closes deterministically (flush or cancel by design) and does not double-apply after the jump.

3. Mode/tool switch boundary
- Start carry with brush + arrow navigation, then switch tool or editor mode before ending navigation.
- Expected: pending carry does not get reinterpreted under the new tool/mode and session closes without ambiguous extra writes.

4. Direction reversal boundary
- Start carry with right-arrow movement, then reverse direction with left-arrow before ending the run.
- Expected: reversal closes or restarts carry by design, with no duplicated or skipped frame application.

## Closeout

- Generate a prefilled artifact first: `./scripts/new_deferred_carry_smoke_result.sh`.
- If the current machine cannot run the ComfyUI UI flow, first try reusing recent blocked evidence with `./scripts/new_deferred_carry_smoke_result.sh --blocked --reuse-latest-blocked` (default 24h window, override with `--max-age-hours <hours>`).
- Reuse is valid only when host, toolchain, device, credentials, script version, and target command are unchanged since the reused artifact.
- If reuse succeeds, capture the printed `REUSED_BLOCKED_ARTIFACT:<path>` line in notes and append a short delta stating environment fields were unchanged.
- If no recent blocked artifact is reusable, generate a new blocked handoff artifact with `./scripts/new_deferred_carry_smoke_result.sh --blocked`.
- Optional: stamp outcomes directly while generating the artifact, for example `./scripts/new_deferred_carry_smoke_result.sh --key-release pass --click-jump pass --mode-switch pass --direction-reversal pass --ambiguous no`.
- Record pass/fail for each end-path using `docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE_RESULT_TEMPLATE.md`.
- Use `blocked` only for machine/runtime blockers, then hand off with exact environment and rerun expectations in the Notes and Follow-up fields.
- Legacy path note: `docs/MASK_EDITOR_DEFERRED_CARRY_RESULTS_TEMPLATE.md` is redirect-only for older links.
- Paste the completed block into the implementation thread (or commit/PR note) so same-run evidence includes explicit end-path outcomes.
- If any path is ambiguous, do not treat the change as complete.
