# Lostless Nodes

Lostless custom nodes for ComfyUI focused on random image selection, sequence buffering, mask editing, and VACE strength scheduling.

## Included Nodes

- `Lostless Random Image`
  Picks an image from a folder and outputs `IMAGE` plus the selected filename. Supports direct image picking, preview restore on reopen, and a per-node `Broadcast Lock` toggle for shared randomize pulses.
- `Lostless Randomize Button`
  Broadcast trigger for connected `Lostless Random Image` nodes so one button can randomize every unlocked Lostless random-image node in the graph.
- `Lostless Buffer`
  Pads image sequences by duplicating the last frame to satisfy `LTX (8n+1)` or `WAN (4n+1)` batch requirements.
- `Lostless VACE Strength Schedule`
  Builds a repeated float schedule for VACE-style strength control from compact text input such as `0.90, 0.64#10, 0.80, 1.00, 0.64#2`.
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

- The repo contains an embedded `Lostless-Mask-Editor` package for compatibility, but this node pack intentionally exposes only the focused Lostless surface documented above.
- If you modify deferred-carry behavior in `Lostless-Mask-Editor/nodes/mask_editor.py`, run `docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE.md` before shipping changes.

## License

This repository is licensed under Apache License 2.0. Embedded upstream-derived files retain their own notices where applicable.
