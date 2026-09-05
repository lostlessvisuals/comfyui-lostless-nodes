# Lostless Nodes

Lostless custom nodes for ComfyUI focused on random image selection, sequence buffering, mask editing, and image-to-mask conversion.

## Included Nodes

- `Lostless Random Image`
  Picks an image from a folder and outputs `IMAGE` plus the selected filename. Supports direct image picking, preview restore on reopen, and a per-node `Broadcast Lock` toggle for shared randomize pulses.
- `Lostless Randomize Button`
  Broadcast trigger for connected `Lostless Random Image` nodes so one button can randomize every unlocked Lostless random-image node in the graph.
- `Lostless Buffer`
  Pads image sequences by duplicating the last frame to satisfy `LTX (8n+1)` or `WAN (4n+1)` batch requirements.
- `Lostless Mask Editor`
  Interactive mask editor for batched image and mask sequences with reusable in-memory sessions, clear-memory control, and frame-to-frame carry while painting.
- `Lostless Image To Mask`
  Converts black-and-white or RGB images into `MASK` plus a preview `IMAGE`, with grayscale-preserving and thresholded conversion modes.

## Installation

### Manual install

1. Clone this repository into `ComfyUI/custom_nodes/comfyui-lostless-nodes`.
2. Install the node-pack dependencies using the same Python environment that runs ComfyUI.
3. Restart ComfyUI.

```bash
python -m pip install -r ComfyUI/custom_nodes/comfyui-lostless-nodes/requirements.txt
```

### ComfyUI-Manager / Registry

Once this pack is published to the Comfy Registry, install it through ComfyUI-Manager by searching for `Lostless Nodes`.

## Notes

- New blank-mask edits start in Shape mode, which creates editable outline points. Drag a point to move it; use `Ctrl+Z` / `Ctrl+Shift+Z` to undo/redo. `Shift+B` switches between Shape and Pixel brushes; the toolbar shows the current mode. Pixel strokes do not create outline points. Use **Show Mask** if the overlay and points are hidden.
- Imported pixel masks open in Pixel mode and retain their grayscale, holes, and small details. Click **Create Points** to convert the sequence to editable outlines. Conversion is approximate and requires confirmation; undo restores the original pixels. Switching a pixel mask to Shape or Liquify mode offers the same conversion rather than changing it silently.
- **Reuse Last Edit** preserves editable points and tool settings together with the saved mask pixels. **Clear Memory** clears both.
- The repo contains an embedded `Lostless-Mask-Editor` package for compatibility, but this node pack intentionally exposes only the focused Lostless surface documented above.
- If you modify deferred-carry behavior in `Lostless-Mask-Editor/nodes/mask_editor.py`, run `docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE.md` before shipping changes.

## Development checks

Install the root requirements and PyTorch (already supplied by ComfyUI) in a disposable Python environment, then run `python -m unittest discover -s tests -v`. The tests use real Qt widgets with an offscreen display, exercise point rendering and dragging, restore a project through the launcher, and verify exported masks. Repeat with `QT_SCALE_FACTOR=2` for high-DPI coverage. A live ComfyUI smoke on the target operating system is still required before release.

## License

This repository is licensed under Apache License 2.0. Embedded upstream-derived files retain their own notices where applicable.
