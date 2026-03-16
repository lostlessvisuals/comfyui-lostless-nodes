# ComfyUI Lostless Nodes

Lostless custom nodes for ComfyUI focused on image selection, sequence buffering, and mask editing.

## Included Nodes

Only these nodes are exposed by this package:

- `Lostless Random Image`
  - Picks an image from a folder and outputs `IMAGE` plus selected filename (`STRING`).
- `Lostless Randomize Button`
  - Broadcast trigger for connected `Lostless Random Image` nodes.
- `Lostless Buffer`
  - Pads image sequences by duplicating the last frame to satisfy `LTX (8n+1)` or `WAN (4n+1)`.
- `Lostless Mask Editor`
  - Interactive mask editor for batched image/mask sequences.
  - Uses `edit_mode` to open the external editor and `reuse_last_edit` to optionally reuse the node's in-memory session edit.
  - Exposes `masks` plus a black/white `mask_image` preview.
- `Lostless Image To Mask`
  - Converts black/white (or RGB) images into `MASK` plus a mask preview `IMAGE`.
  - Supports grayscale-preserving conversion as well as thresholded conversion.

## Embedded Boundary

- The embedded `Lostless-Mask-Editor` folder still contains broader upstream/editor code for compatibility.
- This package intentionally exposes only the focused Lostless surface:
  - random image selection
  - sequence buffering
  - loop-cut planning/apply nodes
  - `Lostless Mask Editor`
  - `Lostless Image To Mask`

## Installation

1. Place this repository in your ComfyUI custom nodes folder:
   - `ComfyUI/custom_nodes/comfyui-lostless-nodes`
2. Install embedded package requirements (recommended) using your ComfyUI Python environment:

```powershell
# Example: ComfyUI venv install
C:\Users\porte\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install -r C:\Users\porte\Documents\ComfyUI\custom_nodes\comfyui-lostless-nodes\Lostless-Mask-Editor\requirements.txt
```

3. Restart ComfyUI.
4. Add nodes from:
   - `lostless/nodes`
   - `lostless/mask`

## License

- The embedded `Lostless-Mask-Editor` package includes an `Apache License 2.0` license file at `Lostless-Mask-Editor/LICENSE`.
- Review upstream and embedded component licenses if you redistribute modified versions.
