# ComfyUI Lostless Nodes

This repository contains Lostless custom nodes for ComfyUI, including random image utilities and an embedded advanced mask editor/video toolkit.

## Included Nodes

- `Lostless Random Image`
  - Pick an image from a folder using a UI randomize action.
  - Outputs `IMAGE` and the selected filename (`STRING`).

- `Lostless Randomize Button`
  - Broadcast randomize trigger to connected `Lostless Random Image` nodes.

- Embedded `Lostless Mask Editor Pipeline` node set
  - Loaded from `Lostless-Mask-Editor-Pipeline`.
  - Includes timeline, mask editor, and WAN video helper nodes exposed by the embedded package.

## Usage

1. Put this repo in `ComfyUI/custom_nodes/comfyui-lostless-nodes`.
2. Restart ComfyUI.
3. Add nodes from `lostless/nodes` and the embedded WanVace/Lostless categories.

## Notes

- The embedded package is versioned in this repo under `Lostless-Mask-Editor-Pipeline`.
- Keep dependencies from that package installed if prompted by ComfyUI.
