# ComfyUI Lostless Nodes

Nodes included:

- `Lostless Random Image`
  - Pick a random image from a folder via button press before queueing.
  - Outputs `IMAGE` and selected file path (`STRING`).

- `Lostless Randomize Button`
  - Standalone button node.
  - Connect its `trigger` output to one or more `Lostless Random Image` nodes.
  - Button press randomizes all connected image nodes at once.

## Notes

- In each random image node, set `folder_path` and optional `allowed_extensions`.
- Press `Randomize Image` on an individual node, or press `Randomize Connected` on the broadcaster node.
- Queue the workflow after selection to load the chosen images.
