# ComfyUI Lostless Nodes Learnings

Use this file for durable repo-local lessons that should change future work in this package.

Rules:
- Keep entries short and factual.
- Capture lessons that should alter node behavior, integration boundaries, validation, or release handling in future sessions.
- Put cross-project patterns in shared skills, root guidance, or root memory instead of duplicating them here.

Entry template:
- Date:
- Context:
- Lesson:
- Action:

## Entries

- Date: 2026-03-09
- Context: Multiple recent mask-editor passes had to clean up node/editor contract drift (`project_data` legacy ports, stale docs, and restore-path mismatches that hid editable keyframes).
- Lesson: Any change to `MaskEditor` inputs/outputs or restore behavior needs a single contract-parity pass across `nodes.py`, frontend slot scrubbers, launcher/session restore logic, and root docs before closeout.
- Action: For future mask-editor surface changes, verify `python3 -m compileall .`, `node --check web/*.js`, and one reopen smoke on an older workflow to confirm stale ports scrub and editable handles/keyframes remain visible.

- Date: 2026-03-17
- Context: Deferred brush-carry navigation changes repeatedly required follow-up review because pending batches behaved ambiguously when arrow runs ended via mode/tool changes instead of key release.
- Lesson: When mask propagation is deferred across frame navigation, carry-session lifetime rules (release, click-jump, mode switch, direction reversal) are part of feature correctness, not polish.
- Action: For future deferred-carry edits in `Lostless-Mask-Editor/nodes/mask_editor.py`, run `docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE.md` and record the result with `docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE_RESULT_TEMPLATE.md` before closeout.

- Date: 2026-05-27
- Context: Preparing the pack for Comfy Registry publication exposed that ComfyUI install and publish tooling expects clean root-level metadata even when runtime dependencies and broader code live inside an embedded subpackage.
- Lesson: For this repo, the publish-facing source of truth lives at the root: `pyproject.toml`, `requirements.txt`, `LICENSE`, and `.comfyignore` must stay accurate even if the embedded `Lostless-Mask-Editor` structure evolves.
- Action: When embedded dependencies, public node surface, or shipped assets change, update the root publish files in the same pass and re-check what the published archive will include.

- Date: 2026-05-27
- Context: The first Registry release accidentally described and shipped a temporary `Lostless VACE Strength Schedule` helper that was not intended to be part of the permanent public pack identity.
- Lesson: This repo's release surface should stay intentionally small; `NODE_CLASS_MAPPINGS`, the README node list, and the root package description must agree on the exact public node count before each publish.
- Action: Before future releases, compare `nodes.py` exports against the README and `pyproject.toml` description to confirm the pack still matches the intended five-node surface.

- Date: 2026-09-04
- Context: Mask point review reproduced a debug-render exception, stale point caches, after-edit undo snapshots, and startup/restore mode drift.
- Lesson: Verify point visibility with a real Qt paint plus mouse events; syntax checks cannot catch missing names inside paintEvent. Restore must set the delayed startup mode as well as the visible widget, and point mutations must invalidate geometry caches.
- Action: Run `python -m unittest discover -s tests -v` and its 2× Qt-scale variant for mask-editor changes. Keep explicit empty keyframes and do not regenerate an existing vector timeline from its raster export. Contour conversion is approximate; restore must preserve pixel data and offer explicit, undoable Create Points conversion. Pixel-mask autosaves must encode raster data because there may be no vector keyframes to recover from. Reuse Last Edit must cache geometry/settings with pixels while rebinding source images to the current input batch.
