import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const RANDOM_IMAGE_NODE = "LostlessRandomImage";
const RANDOMIZE_BUTTON_NODE = "LostlessRandomizeButton";

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

async function fetchRandomImage(node) {
  const folderPath = getWidget(node, "folder_path")?.value || "";
  const recursive = !!getWidget(node, "recursive")?.value;
  const allowedExtensions = getWidget(node, "allowed_extensions")?.value || "";

  const response = await api.fetchApi("/lostless/random-image", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      folder_path: folderPath,
      recursive,
      allowed_extensions: allowedExtensions,
    }),
  });

  const data = await response.json();
  if (!response.ok || !data?.ok) {
    throw new Error(data?.error || "Failed to select random image.");
  }

  return {
    path: data.path,
    filename: data.filename,
  };
}

async function fetchPreviewBlob(path) {
  const response = await api.fetchApi("/lostless/image-preview", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ path }),
  });

  if (!response.ok) {
    throw new Error("Failed to fetch preview image.");
  }

  return await response.blob();
}

function applyPreview(node, blob) {
  if (node.__lostlessPreviewUrl) {
    URL.revokeObjectURL(node.__lostlessPreviewUrl);
    node.__lostlessPreviewUrl = null;
  }

  const objectUrl = URL.createObjectURL(blob);
  node.__lostlessPreviewUrl = objectUrl;

  const img = new Image();
  img.src = objectUrl;
  img.onload = () => {
    node.imgs = [img];
    node.imageIndex = 0;
    node.setSizeForImage?.();
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  };
}

function markNodeDirty(node) {
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function setWidgetValue(widget, value, node) {
  if (!widget) {
    return;
  }
  widget.value = value;
  widget.callback?.(value, app.canvas, node, node.pos, {});
}

function makeReadOnly(widget) {
  if (!widget) {
    return;
  }
  widget.options = widget.options || {};
  widget.options.readOnly = true;
  if (widget.inputEl) {
    widget.inputEl.readOnly = true;
    widget.inputEl.style.opacity = "0.9";
  }
}

function hideWidget(widget) {
  if (!widget) {
    return;
  }
  widget.hidden = true;
}

async function randomizeImageNode(node) {
  const selectedFilenameWidget = getWidget(node, "selected_filename");
  const selectedPathWidget = getWidget(node, "selected_path");
  if (!selectedFilenameWidget || !selectedPathWidget) {
    return;
  }

  const selected = await fetchRandomImage(node);
  setWidgetValue(selectedFilenameWidget, selected.filename, node);
  setWidgetValue(selectedPathWidget, selected.path, node);

  const previewBlob = await fetchPreviewBlob(selected.path);
  applyPreview(node, previewBlob);
  markNodeDirty(node);
}

async function randomizeConnectedNodes(buttonNode) {
  const graph = app.graph;
  if (!graph) {
    return;
  }

  const seenTargetIds = new Set();
  const outputs = buttonNode.outputs || [];

  for (const output of outputs) {
    for (const linkId of output.links || []) {
      const link = graph.links[linkId];
      if (!link || seenTargetIds.has(link.target_id)) {
        continue;
      }
      seenTargetIds.add(link.target_id);

      const targetNode = graph.getNodeById(link.target_id);
      if (targetNode && typeof targetNode.lostlessRandomize === "function") {
        try {
          await targetNode.lostlessRandomize();
        } catch (error) {
          console.error("Lostless randomize failed:", error);
        }
      }
    }
  }
}

app.registerExtension({
  name: "comfyui.lostless.nodes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === RANDOM_IMAGE_NODE) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);
        if (this.__lostless_random_image_ready) {
          return result;
        }
        this.__lostless_random_image_ready = true;

        makeReadOnly(getWidget(this, "selected_filename"));
        hideWidget(getWidget(this, "selected_path"));

        this.lostlessRandomize = async () => {
          await randomizeImageNode(this);
        };

        this.addWidget(
          "button",
          "Randomize Image",
          "",
          async () => {
            try {
              await this.lostlessRandomize();
            } catch (error) {
              console.error("Lostless random image error:", error);
            }
          },
          { serialize: false }
        );

        return result;
      };
    }

    if (nodeData.name === RANDOMIZE_BUTTON_NODE) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);
        if (this.__lostless_broadcast_ready) {
          return result;
        }
        this.__lostless_broadcast_ready = true;

        this.addWidget(
          "button",
          "Randomize Connected",
          "",
          async () => {
            const pulseWidget = getWidget(this, "pulse");
            if (pulseWidget && typeof pulseWidget.value === "number") {
              pulseWidget.value += 1;
              pulseWidget.callback?.(pulseWidget.value, app.canvas, this, this.pos, {});
            }

            await randomizeConnectedNodes(this);
            markNodeDirty(this);
          },
          { serialize: false }
        );

        return result;
      };
    }
  },
});
