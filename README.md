# ComfyUI Lostless Nodes

Lostless custom nodes for ComfyUI focused on random image selection, sequence buffering, loop planning, and mask editing.

## Included Nodes

This package exposes these Lostless nodes:

- `Lostless Random Image`
  - Picks an image from a folder and outputs `IMAGE` plus the selected filename (`STRING`).
  - Supports direct image picking in the node UI and preview restore when workflows reopen.
- `Lostless Randomize Button`
  - Broadcast trigger for connected `Lostless Random Image` nodes.
  - Lets one button randomize every connected Lostless random-image node in the graph.
- `Lostless Buffer`
  - Pads image sequences by duplicating the last frame to satisfy `LTX (8n+1)` or `WAN (4n+1)`.
- `Lostless Loop Cut Planner`
  - Opens an interactive planner for reviewing loop transitions and deciding which frames to cut.
- `Lostless Loop Cut Task`
  - Stores an individual cut instruction for later application.
- `Lostless Loop Apply Cut Plan`
  - Applies a saved cut plan to an image sequence.
- `Lostless Loop Cutter`
  - Combined loop-cut workflow node for planning and applying cuts from one surface.
- `Lostless Mask Editor`
  - Interactive mask editor for batched image and mask sequences.
  - Supports reusable in-memory sessions, clear-memory control, and brush/shape carry while moving through frames.
  - Exposes `masks` plus a black/white `mask_image` preview.
- `Lostless Image To Mask`
  - Converts black/white or RGB images into `MASK` plus a mask preview `IMAGE`.
  - Supports grayscale-preserving conversion as well as thresholded conversion.

## Recent Highlights

- Random-image workflows now support manual image picking, preview restore, and connected-node randomization from the broadcast button node.
- The mask editor now includes reusable session memory, memory clearing from the node UI, improved frame-to-frame mask carry while painting, and a more compact toolbar layout.
- Loop-cut workflows include an in-canvas planner flow for reviewing video frames before applying a cut plan.

## Embedded Boundary

- The embedded `Lostless-Mask-Editor` folder still contains broader upstream/editor code for compatibility.
- This package intentionally exposes only the focused Lostless surface:
  - random image selection
  - sequence buffering
  - loop-cut planning and apply nodes
  - `Lostless Mask Editor`
  - `Lostless Image To Mask`

## Installation

1. Place this repository in your ComfyUI custom nodes folder:
   - `ComfyUI/custom_nodes/comfyui-lostless-nodes`
2. Install the embedded package requirements using your ComfyUI Python environment:

```powershell
# Example: ComfyUI venv install
C:\Users\porte\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install -r C:\Users\porte\Documents\ComfyUI\custom_nodes\comfyui-lostless-nodes\Lostless-Mask-Editor\requirements.txt
```

3. Restart ComfyUI.

## License

- `comfyui-lostless-nodes` is licensed under Apache License 2.0.
- The current license text lives at `Lostless-Mask-Editor/LICENSE` and applies to the full Lostless node pack, not only the embedded mask editor folder.
- Review any embedded upstream notices if you redistribute modified versions.
