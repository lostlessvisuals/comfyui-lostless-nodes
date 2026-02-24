# ComfyUI Lostless Nodes

Lostless custom nodes for ComfyUI, including random image utilities, frame buffering utilities, and an embedded Lostless mask editor/video toolkit.

## Included Nodes

### Lostless Core Nodes (`lostless/nodes`)

- `Lostless Random Image`
  - Picks an image from a folder and outputs `IMAGE` plus selected filename (`STRING`).
- `Lostless Randomize Button`
  - Broadcast trigger for connected `Lostless Random Image` nodes.
- `Lostless Buffer`
  - Pads image sequences by duplicating the last frame to satisfy `LTX (8n+1)` or `WAN (4n+1)`.

### Lostless Mask / Editor Nodes (embedded package)

- `Mask Editor`
  - Interactive mask editor for batched image/mask sequences.
  - Supports node-side project load/save (`project_data` in/out plus optional file path actions).
- `Lostless Image To Mask`
  - Converts black/white (or RGB) images into binary `MASK` plus a mask preview `IMAGE`.
- `Lostless Outpainting Editor`
  - Interactive outpainting canvas workflow node.

### Embedded Lostless Video / Timeline / Utility Nodes

These are loaded from `Lostless-Mask-Editor` and may vary slightly depending on optional dependencies:

- `Lostless Load Video`
- `Lostless Save Video`
- `Lostless Join Videos`
- `Lostless Video Extension`
- `Lostless Frame Interpolator`
- `Lostless Keyframe Timeline`
- `Lostless Frame Injector`
- `Lostless Outpainting Prep`
- `Lostless Fast Image Batch Processor`
- `Lostless Fast Depth Anything V2`
- `Lostless Fast DWPose Estimator`
- `Lostless Mask Viewer` (if available)
- `Lostless Test Mask` (if available)
- `Lostless WAN Inpaint Conditioning` (if available)
- `Lostless WAN Video Sampler Inpaint` (if available)
- `Lostless WAN Tiled Sampler` (if available)
- `Lostless Match Batch Size` (if available)

Experimental/disabled-by-default embedded nodes exist in the embedded package but are not enabled unless that package is configured to expose them.

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
   - embedded `Lostless ...` entries in the ComfyUI node picker

## License

- The embedded `Lostless-Mask-Editor` package includes an `Apache License 2.0` license file at `Lostless-Mask-Editor/LICENSE`.
- Review upstream and embedded component licenses if you redistribute modified versions.
