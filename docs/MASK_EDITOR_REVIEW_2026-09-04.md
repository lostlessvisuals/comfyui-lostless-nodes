# Mask editor robustness review — 2026-09-04

PR Review. Scope: the public MaskEditor launch path, Qt point rendering and editing, project restore, raster recovery, export, and navigation boundaries. This is a targeted review of a large embedded editor, not an exhaustive audit of unrelated video nodes.

Baseline: `main` at `eae4ccf`, matching the remote HEAD at review time. The checkout was clean before this pass. The user authorized additional fixes and a GitHub push after the initial review. This report describes the combined change prepared for that push; Registry metadata and publication are unchanged.

## Findings and fixes

| Priority | Reproducible problem | Consequence | Local fix |
| --- | --- | --- | --- |
| P1 | Debug painting referenced `QPolygon` without importing it. Pressing D reached the error before point drawing. | Outlines could retain their raster overlay while all points disappeared. | Import the polygon type; render a real Qt image and assert visible cyan handles. |
| P1 | New ComfyUI configurations explicitly selected Pixel mode. | Fresh users painted masks with no vector points. | Start blank edits in Shape mode and expose the mode indicator. Preserve supplied pixel masks in Pixel mode, with an explicit Create Points action. |
| P1 | Restored mode, delayed startup mode, and widget mode could disagree; saved frame was assigned before a frame-change method that immediately returned. | Reopen could hide points or display the wrong frame beneath them. | Restore mode through one method, invalidate restored geometry caches, and navigate to the saved frame without preassigning it. |
| P1 | Point dragging did not invalidate cached geometry and saved undo after the edit. The old warp gesture had no active toolbar control. | Points appeared stuck; undo could not recover the original position. | Directly drag visible points, cache invalidation, a single snapshot before movement, and editable interpolated frames. |
| P2 | Hit testing used image pixels, selected the first nearby point, and included hidden shapes. | Zoomed-out points were hard to grab; hidden or neighboring points intercepted clicks. | Pick the nearest visible point within ten screen pixels. Reject a stale drag after frame navigation. |
| P1 | Reuse Last Edit cached only pixels, discarding geometry and settings. | Reused shape masks lost their original editable points. | Store a detached copy of geometry/settings with the matching raster cache; retain current input image paths and clear both forms together. |
| P1 | Raster bootstrap ignored blank frames and could add extra keyframes to an existing vector timeline. | Export filled blank frames or altered animation on reopen. | Preserve blank raster frames as empty keyframes; retain existing vector timelines. |

The recurring design issue is duplicated state and cache ownership: a single user action must update the editor mode, widget mode, pending initialization, and cached geometry together. The focused restore method and regression tests reduce this drift without rewriting the embedded package. This follows the Brooks review guide's knowledge-duplication check and Feathers' characterization-test approach.

## Verification

- Seven initial regression tests failed against the original code, confirming the rendering, mode, selection, interpolation, and undo failures.
- Final 23-test suite passes on Python 3.13, PyQt5 5.15.11 / Qt 5.15.19, OpenCV 4.10.0, NumPy 2.2.6, with normal and 2× Qt scaling.
- Earlier 14-test revision also passed with OpenCV 5.0.0 and NumPy 2.5.2. That combination emitted an existing complex-to-real warning in shape geometry analysis; OpenCV 4.10 / NumPy 2.2 did not.
- Tests use real offscreen Qt widgets and mouse/key events, including Shape strokes, debug pixels, direct dragging, undo/redo, interpolated frames, and interrupted drags.
- Memory tests use real PyTorch tensors and substitute only the interactive subprocess boundary; they verify retained geometry, current input paths, stale dimensions, and cache clearing/replacement.
- A disposable subprocess runs the real launcher, loads a legacy project, waits for startup timers, verifies current frame and tool, accepts the session, and checks the exported `.npy` masks and project JSON.
- Five carry subcases pass: key release, click jump, mode switch, tool switch, and direction reversal; repeated release/idle does not change masks or add undo entries.
- Python compilation and `git diff --check` pass. The public node count and ports are unchanged. Tests are excluded from the Registry archive.
- A dense 150-point outline was visually inspected in the actual Qt editor rendering. The mode badge is placed above the canvas controls so it fits outside the narrow tool strip.

Run: `python -m unittest discover -s tests -v` after installing root requirements plus PyTorch (normally supplied by ComfyUI). Use `QT_SCALE_FACTOR=2` for the high-DPI pass. Qt settings and launcher files are isolated in temporary directories.

## Remaining limits and next improvements

1. **Silent raster conversion is fixed in the follow-up.** Imported pixel masks stay in Pixel mode and export unchanged. Create Points is explicit, confirmed, and undoable; switching to Shape/Liquify requests the same conversion. Tests verify exact preservation of grayscale, holes, and isolated pixels through the real launcher and export, cancellation, and conversion undo/redo. Conversion itself remains approximate, as described in its confirmation dialog. Autosave now includes encoded pixel masks and replaces its prior JSON atomically.
2. **Live Windows/ComfyUI validation remains outstanding.** Testing ran on the MacBook with an isolated Qt runtime. Offscreen tests do not certify Windows DPI settings, remote desktop behavior, the ComfyUI browser extension, or interaction with other node packs. Run the canonical deferred-carry checklist and a new-mask / reopen / export smoke on the user's ComfyUI machine before publishing. See `docs/evidence/deferred-carry/deferred-carry-smoke-2026-09-04-mask-review.md`.
3. Pixel strokes deliberately have no outline points; use Create Points for supplied pixel masks, or Shape mode for new blank masks, and Show Mask if overlays are hidden. No affected user's actual workflow, dependency versions, or logs were available, so the fixes address confirmed causes rather than claiming a single proven cause for every report.

## File impact and rollback

- `Lostless-Mask-Editor/__init__.py`: new-session brush default and editable geometry/settings retained in the existing memory cache.
- `Lostless-Mask-Editor/nodes/mask_editor.py`: rendering import, mode state, direct point editing, cache/undo behavior, raster bootstrap boundaries, and mode indicator.
- `Lostless-Mask-Editor/nodes/comfyui_mask_launcher.py`: consistent restore and frame navigation.
- `tests/`: Qt regressions and disposable launcher smoke; `.comfyignore` excludes these from releases.
- README and local continuity docs record behavior, verification, and limitations.

Rollback is a revert of the mask-hardening commit(s) after `eae4ccf`, preserving later work. No dependencies or release versions were changed in the repository.

## Push readiness

Recommendation: ready to push with the documented live-ComfyUI validation caveat. Prepared `.github/workflows/mask-editor-tests.yml` to exercise normal and 2× scaling on Windows and Linux. GitHub rejected adding that file with the configured token because it lacks workflow scope. The CI candidate is preserved on local branch `codex/mask-editor-ci-20260904`; the code/test push excludes the workflow file. No remote CI result is claimed. No `pyproject.toml` change is included, so the Registry publishing workflow is not triggered by this code push.
