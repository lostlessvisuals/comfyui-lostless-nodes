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
