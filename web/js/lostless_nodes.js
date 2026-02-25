import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const RANDOM_IMAGE_NODE = "LostlessRandomImage";
const RANDOMIZE_BUTTON_NODE = "LostlessRandomizeButton";
const LOOP_CUT_PLANNER_NODE = "LostlessLoopCutPlanner";
const LOOP_CUTTER_NODE = "LostlessLoopCutter";

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
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

function getConnectedInputNode(node, inputName) {
  const graph = app.graph;
  if (!graph) {
    return null;
  }
  const input = (node.inputs || []).find((entry) => entry.name === inputName);
  const linkId = input?.link;
  if (!linkId) {
    return null;
  }
  const link = graph.links?.[linkId];
  if (!link) {
    return null;
  }
  return graph.getNodeById?.(link.origin_id) || null;
}

function getConnectedVhsVideoPath(node) {
  const upstream = getConnectedInputNode(node, "images");
  if (!upstream || upstream.type !== "VHS_LoadVideo") {
    return "";
  }

  const widgetsValues = upstream.widgets_values || {};
  if (widgetsValues && typeof widgetsValues === "object" && !Array.isArray(widgetsValues)) {
    const direct = widgetsValues.video || widgetsValues?.videopreview?.params?.filename;
    if (typeof direct === "string" && direct.trim()) {
      return direct.trim();
    }
  }

  const widget = getWidget(upstream, "video");
  if (typeof widget?.value === "string" && widget.value.trim()) {
    return widget.value.trim();
  }
  return "";
}

function parseJsonSafe(value, fallback) {
  try {
    const parsed = JSON.parse(value || "");
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
}

async function postJson(path, body) {
  const response = await api.fetchApi(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  let data = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    data = await response.json();
  }

  if (!response.ok || (data && data.ok === false)) {
    throw new Error(data?.error || `Request failed: ${response.status}`);
  }

  return { response, data };
}

async function fetchRandomImage(node) {
  const folderPath = getWidget(node, "folder_path")?.value || "";
  const recursive = !!getWidget(node, "recursive")?.value;
  const allowedExtensions = getWidget(node, "allowed_extensions")?.value || "";

  const { data } = await postJson("/lostless/random-image", {
    folder_path: folderPath,
    recursive,
    allowed_extensions: allowedExtensions,
  });

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

async function fetchVideoMeta(path) {
  const { data } = await postJson("/lostless/video-meta", { path });
  return data;
}

async function fetchVideoFrameBlob(path, frameIndex, maxWidth = 960) {
  const response = await api.fetchApi("/lostless/video-frame", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      path,
      frame_index: frameIndex,
      max_width: maxWidth,
    }),
  });

  if (!response.ok) {
    let message = "Failed to load video frame.";
    try {
      const data = await response.json();
      message = data?.error || message;
    } catch {
      // no-op
    }
    throw new Error(message);
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

function ensureLoopPlannerStyles() {
  if (document.getElementById("lostless-loop-planner-styles")) {
    return;
  }

  const style = document.createElement("style");
  style.id = "lostless-loop-planner-styles";
  style.textContent = `
    .lostless-loop-overlay {
      position: fixed;
      inset: 0;
      background: rgba(10, 12, 16, 0.75);
      z-index: 100000;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
    }
    .lostless-loop-modal {
      width: min(1180px, 96vw);
      max-height: 92vh;
      overflow: hidden;
      background: #15181e;
      color: #e5eaf3;
      border: 1px solid #2b3445;
      border-radius: 12px;
      box-shadow: 0 20px 80px rgba(0,0,0,0.45);
      display: grid;
      grid-template-rows: auto 1fr auto;
      font: 12px/1.35 ui-sans-serif, system-ui, sans-serif;
    }
    .lostless-loop-header {
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid #273041;
      background: linear-gradient(180deg, #1b2230, #161b26);
    }
    .lostless-loop-header input {
      flex: 1;
      min-width: 180px;
      background: #0f1218;
      border: 1px solid #2a3342;
      color: #e5eaf3;
      border-radius: 6px;
      padding: 6px 8px;
    }
    .lostless-loop-btn {
      background: #253047;
      color: #e5eaf3;
      border: 1px solid #384862;
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }
    .lostless-loop-btn:hover { background: #2d3a55; }
    .lostless-loop-btn.danger { background: #4a2020; border-color: #6a3030; }
    .lostless-loop-btn.primary { background: #1f4c7a; border-color: #2c6dad; }
    .lostless-loop-main {
      display: grid;
      grid-template-columns: minmax(520px, 1.3fr) minmax(360px, 1fr);
      gap: 10px;
      padding: 10px;
      min-height: 0;
    }
    .lostless-loop-pane {
      border: 1px solid #273041;
      border-radius: 10px;
      background: #11151d;
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    .lostless-loop-preview-wrap {
      padding: 10px;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 280px;
      background: #0d1016;
      border-bottom: 1px solid #252e3e;
    }
    .lostless-loop-preview-wrap img {
      max-width: 100%;
      max-height: 48vh;
      object-fit: contain;
      border-radius: 6px;
      border: 1px solid #2a3342;
      background: #0a0c10;
    }
    .lostless-loop-controls {
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .lostless-loop-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .lostless-loop-row input[type="number"] {
      width: 84px;
      background: #0f1218;
      border: 1px solid #2a3342;
      color: #e5eaf3;
      border-radius: 6px;
      padding: 5px 6px;
    }
    .lostless-loop-row input[type="range"] { flex: 1; min-width: 180px; }
    .lostless-loop-label { color: #9fb0c8; min-width: 58px; }
    .lostless-loop-status {
      color: #bdd0ea;
      background: #141b27;
      border: 1px solid #263247;
      border-radius: 6px;
      padding: 6px 8px;
      min-height: 18px;
      white-space: pre-wrap;
    }
    .lostless-loop-cuts {
      padding: 10px;
      overflow: auto;
      min-height: 0;
      display: grid;
      gap: 8px;
      align-content: start;
    }
    .lostless-loop-cut-row {
      display: grid;
      grid-template-columns: 34px 1fr 1fr 1fr auto;
      gap: 6px;
      align-items: center;
      border: 1px solid #273041;
      border-radius: 8px;
      padding: 6px;
      background: #141a24;
    }
    .lostless-loop-cut-row input {
      width: 100%;
      min-width: 0;
      background: #0f1218;
      border: 1px solid #2a3342;
      color: #e5eaf3;
      border-radius: 6px;
      padding: 4px 6px;
      box-sizing: border-box;
    }
    .lostless-loop-cut-head {
      display: grid;
      grid-template-columns: 34px 1fr 1fr 1fr auto;
      gap: 6px;
      color: #8fa2bf;
      padding: 0 6px;
    }
    .lostless-loop-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-top: 1px solid #273041;
      background: #141924;
    }
    .lostless-loop-footer .left,
    .lostless-loop-footer .right { display: flex; gap: 8px; align-items: center; }
    .lostless-loop-muted { color: #98a9c3; }
    @media (max-width: 980px) {
      .lostless-loop-main { grid-template-columns: 1fr; }
      .lostless-loop-preview-wrap img { max-height: 34vh; }
    }
  `;
  document.head.appendChild(style);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, Number(value) || 0));
}

function buildDefaultCut(frame, transitionFrames, removeBufferEachSide = 0) {
  return {
    center: Math.max(0, Number(frame) || 0),
    remove_buffer_each_side: Math.max(0, Number(removeBufferEachSide) || 0),
    transition_frames: Math.max(0, Number(transitionFrames) || 0),
  };
}

function cutRange(cut) {
  const center = Math.max(0, Number(cut?.center ?? cut?.cut_frame ?? cut?.frame) || 0);
  const buffer = Math.max(
    0,
    Number(cut?.remove_buffer_each_side ?? cut?.buffer_each_side ?? cut?.buffer) || 0
  );
  const start = Math.max(0, center - buffer);
  const end = Math.max(start, center + buffer);
  return { center, buffer, start, end };
}

async function openLoopCutPlanner(node) {
  ensureLoopPlannerStyles();

  const videoPathWidget = getWidget(node, "video_path");
  const cutsJsonWidget = getWidget(node, "cuts_json");
  const videoInfoWidget = getWidget(node, "video_info_json");
  const refreshWidget = getWidget(node, "ui_refresh");
  const defaultTransitionWidget = getWidget(node, "default_transition_frames");
  const defaultSourceWidget = getWidget(node, "default_source_frames");
  const defaultOverlapWidget = getWidget(node, "default_overlap_frames");

  if (!videoPathWidget || !cutsJsonWidget || !defaultTransitionWidget) {
    throw new Error("Loop cutter widgets are missing.");
  }

  const inferredVideoPath = getConnectedVhsVideoPath(node);
  if ((!videoPathWidget.value || !String(videoPathWidget.value).trim()) && inferredVideoPath) {
    setWidgetValue(videoPathWidget, inferredVideoPath, node);
  }

  const loadedCuts = parseJsonSafe(String(cutsJsonWidget.value || "[]"), []);
  const normalizedCuts = Array.isArray(loadedCuts)
    ? loadedCuts
        .filter((c) => c && typeof c === "object")
        .map((c) => {
          const r = cutRange(c);
          return {
            center: r.center,
            remove_buffer_each_side: r.buffer,
            transition_frames: Math.max(
              0,
              Number(c.transition_frames ?? defaultTransitionWidget.value ?? 0) || 0
            ),
          };
        })
    : [];

  const state = {
    path: String(videoPathWidget.value || inferredVideoPath || ""),
    meta: parseJsonSafe(String(videoInfoWidget?.value || "{}"), {}),
    cuts: normalizedCuts,
    currentFrame: 0,
    selectedCutIndex: normalizedCuts.length ? 0 : -1,
    frameUrl: null,
    frameRequestId: 0,
    frameCache: new Map(),
    frameCacheOrder: [],
    maxCachedFrames: 80,
    sliderTimer: null,
    keyHandler: null,
    timelineCanvas: null,
  };

  const overlay = document.createElement("div");
  overlay.className = "lostless-loop-overlay";

  const modal = document.createElement("div");
  modal.className = "lostless-loop-modal";
  overlay.appendChild(modal);

  modal.innerHTML = `
    <div class="lostless-loop-header">
      <span class="lostless-loop-muted">Loop Cutter</span>
      <span class="lostless-loop-muted ll-source-label" style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"></span>
      <button class="lostless-loop-btn ll-reload">Reload Video</button>
      <button class="lostless-loop-btn ll-close">Close</button>
    </div>
    <div class="lostless-loop-main" style="grid-template-columns: 1fr;">
      <div class="lostless-loop-pane">
        <div class="lostless-loop-preview-wrap"><img class="ll-preview" alt="Video frame preview" /></div>
        <div class="lostless-loop-controls">
          <div class="lostless-loop-status ll-status"></div>
          <div class="lostless-loop-row">
            <span class="lostless-loop-label">Frame</span>
            <button class="lostless-loop-btn ll-step" data-step="-10">Prev 10</button>
            <button class="lostless-loop-btn ll-step" data-step="-1">Prev</button>
            <input type="number" class="ll-frame-input" min="0" step="1" value="0" />
            <button class="lostless-loop-btn ll-step" data-step="1">Next</button>
            <button class="lostless-loop-btn ll-step" data-step="10">Next 10</button>
            <button class="lostless-loop-btn primary ll-add-cut">Add Cut @ Current</button>
          </div>
          <div class="lostless-loop-row">
            <input type="range" class="ll-frame-slider" min="0" max="0" step="1" value="0" />
          </div>
          <div class="lostless-loop-row">
            <canvas class="ll-timeline" width="1000" height="86" style="width:100%; height:86px; background:#0d1118; border:1px solid #273041; border-radius:8px; cursor:pointer;"></canvas>
          </div>
          <div class="lostless-loop-row" style="justify-content:space-between;">
            <span class="lostless-loop-muted">Timeline: ticks = frames, bars = removed region, yellow line = cut frame</span>
            <span class="lostless-loop-muted">Keys: Left/Right, Shift+Left/Right, C add cut</span>
          </div>
          <div class="lostless-loop-row" style="border-top:1px solid #252e3e; padding-top:8px;">
            <span class="lostless-loop-muted">Selected Cut</span>
            <button class="lostless-loop-btn ll-prev-cut">Prev Cut</button>
            <button class="lostless-loop-btn ll-next-cut">Next Cut</button>
            <button class="lostless-loop-btn danger ll-delete-cut">Delete Cut</button>
          </div>
          <div class="lostless-loop-row">
            <span class="lostless-loop-label">Cut Frame</span>
            <input type="number" class="ll-cut-center" min="0" step="1" value="0" />
            <span class="lostless-loop-label">Buffer/Side</span>
            <input type="number" class="ll-cut-buffer" min="0" step="1" value="0" />
            <span class="lostless-loop-label">Transition</span>
            <input type="number" class="ll-cut-transition" min="0" step="1" value="0" />
            <button class="lostless-loop-btn ll-jump-cut">Jump To Cut</button>
          </div>
          <div class="lostless-loop-row" style="border-top:1px solid #252e3e; padding-top:8px;">
            <span class="lostless-loop-muted">Defaults (saved inside node but edited here)</span>
          </div>
          <div class="lostless-loop-row">
            <span class="lostless-loop-label">Def Buffer</span>
            <input type="number" class="ll-default-buffer" min="0" step="1" value="0" />
            <span class="lostless-loop-label">Def Transition</span>
            <input type="number" class="ll-default-transition" min="0" step="1" value="0" />
            <span class="lostless-loop-label">Source</span>
            <input type="number" class="ll-default-source" min="1" step="1" value="16" />
            <span class="lostless-loop-label">Overlap</span>
            <input type="number" class="ll-default-overlap" min="0" step="1" value="8" />
          </div>
        </div>
      </div>
    </div>
    <div class="lostless-loop-footer">
      <div class="left">
        <span class="lostless-loop-muted ll-summary">No video loaded</span>
      </div>
      <div class="right">
        <button class="lostless-loop-btn ll-cancel">Cancel</button>
        <button class="lostless-loop-btn primary ll-save">Save To Node</button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  const previewImg = modal.querySelector(".ll-preview");
  const statusEl = modal.querySelector(".ll-status");
  const summaryEl = modal.querySelector(".ll-summary");
  const sourceLabelEl = modal.querySelector(".ll-source-label");
  const sliderEl = modal.querySelector(".ll-frame-slider");
  const frameInputEl = modal.querySelector(".ll-frame-input");
  const timelineCanvas = modal.querySelector(".ll-timeline");
  const cutCenterEl = modal.querySelector(".ll-cut-center");
  const cutBufferEl = modal.querySelector(".ll-cut-buffer");
  const cutTransitionEl = modal.querySelector(".ll-cut-transition");
  const defaultBufferEl = modal.querySelector(".ll-default-buffer");
  const defaultTransitionEl = modal.querySelector(".ll-default-transition");
  const defaultSourceEl = modal.querySelector(".ll-default-source");
  const defaultOverlapEl = modal.querySelector(".ll-default-overlap");
  const reloadButton = modal.querySelector(".ll-reload");
  const closeButton = modal.querySelector(".ll-close");
  const cancelButton = modal.querySelector(".ll-cancel");
  const saveButton = modal.querySelector(".ll-save");

  state.timelineCanvas = timelineCanvas;
  defaultBufferEl.value = String(Math.max(0, Number(defaultOverlapWidget?.value) || 0));
  defaultTransitionEl.value = String(Math.max(0, Number(defaultTransitionWidget.value) || 0));
  defaultSourceEl.value = String(Math.max(1, Number(defaultSourceWidget?.value) || 16));
  defaultOverlapEl.value = String(Math.max(0, Number(defaultOverlapWidget?.value) || 8));

  function setStatus(message) {
    statusEl.textContent = String(message || "");
  }

  function getFrameCount() {
    return Math.max(0, Number(state.meta?.frame_count) || 0);
  }

  function cleanup() {
    if (state.frameUrl) {
      URL.revokeObjectURL(state.frameUrl);
      state.frameUrl = null;
    }
    if (state.sliderTimer) clearTimeout(state.sliderTimer);
    if (state.keyHandler) window.removeEventListener("keydown", state.keyHandler);
    overlay.remove();
  }

  function syncSourceLabel() {
    sourceLabelEl.textContent = state.path
      ? `Source: ${state.path}`
      : "Source: connect VHS_LoadVideo (IMAGE) to auto-detect video";
  }

  function syncFrameControls() {
    const frameCount = getFrameCount();
    const maxFrame = Math.max(0, frameCount - 1);
    state.currentFrame = clamp(state.currentFrame, 0, maxFrame);
    sliderEl.max = String(maxFrame);
    sliderEl.value = String(state.currentFrame);
    frameInputEl.max = String(maxFrame);
    frameInputEl.value = String(state.currentFrame);
  }

  function cacheFrameBlob(frameIndex, blob) {
    state.frameCache.set(frameIndex, blob);
    state.frameCacheOrder = state.frameCacheOrder.filter((n) => n !== frameIndex);
    state.frameCacheOrder.push(frameIndex);
    while (state.frameCacheOrder.length > state.maxCachedFrames) {
      const oldest = state.frameCacheOrder.shift();
      if (oldest != null) state.frameCache.delete(oldest);
    }
  }

  async function getCachedFrameBlob(frameIndex) {
    if (state.frameCache.has(frameIndex)) return state.frameCache.get(frameIndex);
    const blob = await fetchVideoFrameBlob(state.path, frameIndex, 640);
    cacheFrameBlob(frameIndex, blob);
    return blob;
  }

  async function prefetchNeighborFrames(centerIndex) {
    const frameCount = getFrameCount();
    if (!frameCount) return;
    const candidates = [centerIndex - 1, centerIndex + 1, centerIndex - 10, centerIndex + 10]
      .map((n) => clamp(n, 0, frameCount - 1))
      .filter((n, i, arr) => arr.indexOf(n) === i && !state.frameCache.has(n));
    for (const idx of candidates.slice(0, 2)) {
      getCachedFrameBlob(idx).catch(() => {});
    }
  }

  function frameToX(frame, width, frameCount) {
    if (frameCount <= 1) return 0;
    return (frame / (frameCount - 1)) * (width - 1);
  }

  function drawTimeline() {
    if (!timelineCanvas) return;
    const ctx = timelineCanvas.getContext("2d");
    if (!ctx) return;
    const frameCount = getFrameCount();
    const w = timelineCanvas.clientWidth || 900;
    const h = timelineCanvas.clientHeight || 86;
    if (timelineCanvas.width !== w) timelineCanvas.width = w;
    if (timelineCanvas.height !== h) timelineCanvas.height = h;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0d1118";
    ctx.fillRect(0, 0, w, h);

    if (!frameCount) {
      ctx.fillStyle = "#95a7c4";
      ctx.font = "12px ui-sans-serif, system-ui";
      ctx.fillText("Connect/load a video to draw timeline", 12, 22);
      return;
    }

    const tickY0 = 46;
    const tickY1 = h - 10;
    const maxTicks = Math.min(frameCount, 1200);
    const tickStep = Math.max(1, Math.ceil(frameCount / maxTicks));
    for (let f = 0; f < frameCount; f += tickStep) {
      const x = frameToX(f, w, frameCount);
      const major = (f % 10) === 0;
      ctx.strokeStyle = major ? "#33465f" : "#263343";
      ctx.beginPath();
      ctx.moveTo(x + 0.5, major ? tickY0 - 8 : tickY0);
      ctx.lineTo(x + 0.5, tickY1);
      ctx.stroke();
    }

    // Removed regions bars
    state.cuts.forEach((cut, idx) => {
      const r = cutRange(cut);
      const start = clamp(r.start, 0, frameCount - 1);
      const end = clamp(r.end, 0, frameCount - 1);
      const x0 = frameToX(start, w, frameCount);
      const x1 = frameToX(end, w, frameCount);
      const selected = idx === state.selectedCutIndex;
      ctx.fillStyle = selected ? "rgba(55, 195, 255, 0.35)" : "rgba(236, 93, 80, 0.28)";
      ctx.fillRect(Math.min(x0, x1), 30, Math.max(2, Math.abs(x1 - x0)), 14);
      ctx.strokeStyle = selected ? "rgba(90, 220, 255, 0.9)" : "rgba(255, 132, 120, 0.8)";
      ctx.strokeRect(Math.min(x0, x1), 30.5, Math.max(2, Math.abs(x1 - x0)), 13);

      const cx = frameToX(r.center, w, frameCount);
      ctx.strokeStyle = selected ? "#ffe178" : "#d9b24d";
      ctx.beginPath();
      ctx.moveTo(cx + 0.5, 14);
      ctx.lineTo(cx + 0.5, h - 6);
      ctx.stroke();
    });

    // Current frame indicator
    const curX = frameToX(state.currentFrame, w, frameCount);
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(curX + 0.5, 0);
    ctx.lineTo(curX + 0.5, h);
    ctx.stroke();

    ctx.fillStyle = "#c7d6ec";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.fillText(`Current: ${state.currentFrame}`, 10, 14);
  }

  function updateSummary() {
    const frameCount = getFrameCount();
    const totalRemoved = state.cuts.reduce((sum, cut) => {
      const r = cutRange(cut);
      return sum + (r.end - r.start + 1);
    }, 0);
    const finalCount = frameCount ? Math.max(0, frameCount - totalRemoved) : "?";
    const fps = Number(state.meta?.fps || 0);
    const wh = `${state.meta?.width || 0}x${state.meta?.height || 0}`;
    summaryEl.textContent = frameCount
      ? `${frameCount} frames | ${fps.toFixed(3)} fps | ${wh} | removed ${totalRemoved} | final ${finalCount}`
      : "No video loaded";
    drawTimeline();
  }

  function clampCutToBounds(cut) {
    const frameCount = getFrameCount();
    if (!frameCount) return cut;
    let center = clamp(Number(cut.center) || 0, 0, frameCount - 1);
    let buffer = Math.max(0, Number(cut.remove_buffer_each_side) || 0);
    buffer = Math.min(buffer, center, (frameCount - 1) - center);
    return {
      center,
      remove_buffer_each_side: buffer,
      transition_frames: Math.max(0, Number(cut.transition_frames) || 0),
    };
  }

  function setSelectedCut(index) {
    if (!state.cuts.length) {
      state.selectedCutIndex = -1;
      cutCenterEl.value = "0";
      cutBufferEl.value = String(Math.max(0, Number(defaultBufferEl.value) || 0));
      cutTransitionEl.value = String(Math.max(0, Number(defaultTransitionEl.value) || 0));
      drawTimeline();
      return;
    }
    state.selectedCutIndex = clamp(index, 0, state.cuts.length - 1);
    const cut = clampCutToBounds(state.cuts[state.selectedCutIndex]);
    state.cuts[state.selectedCutIndex] = cut;
    cutCenterEl.value = String(cut.center);
    cutBufferEl.value = String(cut.remove_buffer_each_side);
    cutTransitionEl.value = String(cut.transition_frames);
    drawTimeline();
  }

  function sortCutsAndPreserveSelection() {
    const selected = state.selectedCutIndex >= 0 ? state.cuts[state.selectedCutIndex] : null;
    state.cuts.sort((a, b) => {
      const ac = Number(a.center) || 0;
      const bc = Number(b.center) || 0;
      if (ac !== bc) return ac - bc;
      return (Number(a.remove_buffer_each_side) || 0) - (Number(b.remove_buffer_each_side) || 0);
    });
    if (selected) {
      state.selectedCutIndex = state.cuts.indexOf(selected);
    }
    if (state.selectedCutIndex < 0 && state.cuts.length) state.selectedCutIndex = 0;
  }

  function applySelectedCutInputs() {
    if (state.selectedCutIndex < 0 || state.selectedCutIndex >= state.cuts.length) return;
    state.cuts[state.selectedCutIndex] = clampCutToBounds({
      center: Number(cutCenterEl.value) || 0,
      remove_buffer_each_side: Number(cutBufferEl.value) || 0,
      transition_frames: Number(cutTransitionEl.value) || 0,
    });
    sortCutsAndPreserveSelection();
    setSelectedCut(state.selectedCutIndex);
    updateSummary();
  }

  function validateCutsForSave() {
    const frameCount = getFrameCount();
    const cuts = state.cuts
      .map((cut) => clampCutToBounds(cut))
      .sort((a, b) => (a.center - b.center) || (a.remove_buffer_each_side - b.remove_buffer_each_side));

    for (let i = 0; i < cuts.length; i++) {
      const a = cutRange(cuts[i]);
      if (frameCount && (a.end >= frameCount || a.start < 0)) {
        throw new Error(`Cut ${i + 1} exceeds frame bounds.`);
      }
      if (i > 0) {
        const prev = cutRange(cuts[i - 1]);
        if (a.start <= prev.end) {
          throw new Error(`Cuts overlap (${i} and ${i + 1}). Reduce buffer on one cut.`);
        }
      }
    }
    return cuts;
  }

  function syncDefaultsToWidgets() {
    setWidgetValue(defaultTransitionWidget, Math.max(0, Number(defaultTransitionEl.value) || 0), node);
    if (defaultSourceWidget) setWidgetValue(defaultSourceWidget, Math.max(1, Number(defaultSourceEl.value) || 1), node);
    if (defaultOverlapWidget) setWidgetValue(defaultOverlapWidget, Math.max(0, Number(defaultOverlapEl.value) || 0), node);
  }

  async function renderFrame(frameIndex) {
    const frameCount = getFrameCount();
    if (!frameCount || !state.path) return;

    state.currentFrame = clamp(frameIndex, 0, frameCount - 1);
    syncFrameControls();
    drawTimeline();
    setStatus(`Loading frame ${state.currentFrame}...`);

    const reqId = ++state.frameRequestId;
    try {
      const blob = await getCachedFrameBlob(state.currentFrame);
      if (reqId !== state.frameRequestId) return;
      if (state.frameUrl) URL.revokeObjectURL(state.frameUrl);
      state.frameUrl = URL.createObjectURL(blob);
      previewImg.src = state.frameUrl;
      setStatus(`Frame ${state.currentFrame} / ${frameCount - 1}`);
      prefetchNeighborFrames(state.currentFrame);
      drawTimeline();
    } catch (error) {
      if (reqId !== state.frameRequestId) return;
      setStatus(error?.message || String(error));
    }
  }

  async function loadVideo() {
    const inferred = getConnectedVhsVideoPath(node);
    if (inferred && inferred !== state.path) {
      state.path = inferred;
      setWidgetValue(videoPathWidget, inferred, node);
    }

    syncSourceLabel();
    if (!state.path) {
      setStatus("Connect VHS_LoadVideo IMAGE to this node so the cutter can detect the source video.");
      updateSummary();
      return;
    }

    setStatus("Loading video metadata...");
    try {
      const meta = await fetchVideoMeta(state.path);
      state.meta = {
        path: meta.path,
        frame_count: Number(meta.frame_count) || 0,
        fps: Number(meta.fps) || 0,
        width: Number(meta.width) || 0,
        height: Number(meta.height) || 0,
      };
      state.path = meta.path || state.path;
      state.frameCache.clear();
      state.frameCacheOrder = [];
      syncSourceLabel();
      syncFrameControls();
      sortCutsAndPreserveSelection();
      setSelectedCut(state.selectedCutIndex);
      updateSummary();
      await renderFrame(state.currentFrame);
    } catch (error) {
      setStatus(error?.message || String(error));
      updateSummary();
    }
  }

  function addCutAtCurrent() {
    const cut = clampCutToBounds(
      buildDefaultCut(
        state.currentFrame,
        Number(defaultTransitionEl.value) || 0,
        Number(defaultBufferEl.value) || 0
      )
    );
    state.cuts.push(cut);
    sortCutsAndPreserveSelection();
    setSelectedCut(state.cuts.indexOf(cut));
    updateSummary();
  }

  function deleteSelectedCut() {
    if (state.selectedCutIndex < 0 || state.selectedCutIndex >= state.cuts.length) return;
    state.cuts.splice(state.selectedCutIndex, 1);
    if (!state.cuts.length) {
      state.selectedCutIndex = -1;
    } else if (state.selectedCutIndex >= state.cuts.length) {
      state.selectedCutIndex = state.cuts.length - 1;
    }
    setSelectedCut(state.selectedCutIndex);
    updateSummary();
  }

  function saveToNode() {
    const normalizedCuts = validateCutsForSave();
    const videoInfo = {
      path: String(state.path || ""),
      frame_count: Number(state.meta?.frame_count) || 0,
      fps: Number(state.meta?.fps) || 0,
      width: Number(state.meta?.width) || 0,
      height: Number(state.meta?.height) || 0,
    };

    syncDefaultsToWidgets();
    setWidgetValue(videoPathWidget, videoInfo.path, node);
    setWidgetValue(cutsJsonWidget, JSON.stringify(normalizedCuts, null, 2), node);
    if (videoInfoWidget) setWidgetValue(videoInfoWidget, JSON.stringify(videoInfo), node);
    if (refreshWidget) setWidgetValue(refreshWidget, (Number(refreshWidget.value) || 0) + 1, node);

    markNodeDirty(node);
    cleanup();
  }

  function frameFromTimelineClick(event) {
    const rect = timelineCanvas.getBoundingClientRect();
    const x = clamp(event.clientX - rect.left, 0, rect.width);
    const frameCount = getFrameCount();
    if (!frameCount) return 0;
    return Math.round((x / Math.max(1, rect.width - 1)) * Math.max(0, frameCount - 1));
  }

  timelineCanvas.addEventListener("click", (event) => {
    const frame = frameFromTimelineClick(event);
    const frameCount = getFrameCount();
    if (!frameCount) return;

    // Prefer selecting an existing cut when clicking near its center.
    let bestIndex = -1;
    let bestDist = Number.POSITIVE_INFINITY;
    state.cuts.forEach((cut, idx) => {
      const d = Math.abs(cutRange(cut).center - frame);
      if (d < bestDist) {
        bestDist = d;
        bestIndex = idx;
      }
    });
    if (bestIndex >= 0 && bestDist <= Math.max(1, Math.round(frameCount * 0.02))) {
      setSelectedCut(bestIndex);
    }
    renderFrame(frame);
  });

  timelineCanvas.addEventListener("dblclick", (event) => {
    const frame = frameFromTimelineClick(event);
    state.currentFrame = frame;
    addCutAtCurrent();
    renderFrame(frame);
  });

  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) cleanup();
  });

  closeButton.addEventListener("click", cleanup);
  cancelButton.addEventListener("click", cleanup);
  reloadButton.addEventListener("click", () => loadVideo());

  modal.querySelectorAll(".ll-step").forEach((button) => {
    button.addEventListener("click", () => renderFrame(state.currentFrame + Number(button.dataset.step || 0)));
  });

  modal.querySelector(".ll-add-cut")?.addEventListener("click", addCutAtCurrent);
  modal.querySelector(".ll-delete-cut")?.addEventListener("click", deleteSelectedCut);
  modal.querySelector(".ll-prev-cut")?.addEventListener("click", () => setSelectedCut(state.selectedCutIndex - 1));
  modal.querySelector(".ll-next-cut")?.addEventListener("click", () => setSelectedCut(state.selectedCutIndex + 1));
  modal.querySelector(".ll-jump-cut")?.addEventListener("click", () => {
    if (state.selectedCutIndex < 0) return;
    renderFrame(cutRange(state.cuts[state.selectedCutIndex]).center);
  });

  sliderEl.addEventListener("input", () => {
    if (state.sliderTimer) clearTimeout(state.sliderTimer);
    const target = Number(sliderEl.value) || 0;
    state.sliderTimer = setTimeout(() => renderFrame(target), 30);
  });
  frameInputEl.addEventListener("change", () => renderFrame(Number(frameInputEl.value) || 0));

  [cutCenterEl, cutBufferEl, cutTransitionEl].forEach((el) => {
    el.addEventListener("change", applySelectedCutInputs);
    el.addEventListener("input", () => {
      if (el === cutCenterEl || el === cutBufferEl || el === cutTransitionEl) {
        applySelectedCutInputs();
      }
    });
  });

  [defaultBufferEl, defaultTransitionEl, defaultSourceEl, defaultOverlapEl].forEach((el) => {
    el.addEventListener("change", () => {
      syncDefaultsToWidgets();
      markNodeDirty(node);
    });
  });

  saveButton.addEventListener("click", () => {
    try {
      saveToNode();
    } catch (error) {
      setStatus(error?.message || String(error));
    }
  });

  state.keyHandler = (event) => {
    if (!overlay.isConnected) return;
    const tag = event.target?.tagName?.toLowerCase?.() || "";
    if (tag === "input" || tag === "textarea") return;

    if (event.key === "ArrowLeft") {
      event.preventDefault();
      renderFrame(state.currentFrame + (event.shiftKey ? -10 : -1));
      return;
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      renderFrame(state.currentFrame + (event.shiftKey ? 10 : 1));
      return;
    }
    if (event.key.toLowerCase() === "c") {
      event.preventDefault();
      addCutAtCurrent();
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      if (state.selectedCutIndex >= 0) {
        event.preventDefault();
        deleteSelectedCut();
      }
    }
  };
  window.addEventListener("keydown", state.keyHandler);

  syncSourceLabel();
  syncFrameControls();
  setSelectedCut(state.selectedCutIndex);
  updateSummary();

  if (state.meta && Number(state.meta.frame_count) > 0) {
    state.currentFrame = clamp(state.currentFrame, 0, Number(state.meta.frame_count) - 1);
    syncFrameControls();
    renderFrame(state.currentFrame);
  } else {
    loadVideo();
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

    if (nodeData.name === LOOP_CUT_PLANNER_NODE || nodeData.name === LOOP_CUTTER_NODE) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);
        if (this.__lostless_loop_planner_ready) {
          return result;
        }
        this.__lostless_loop_planner_ready = true;

        makeReadOnly(getWidget(this, "video_info_json"));
        hideWidget(getWidget(this, "video_info_json"));
        hideWidget(getWidget(this, "video_path"));
        hideWidget(getWidget(this, "default_transition_frames"));
        hideWidget(getWidget(this, "default_source_frames"));
        hideWidget(getWidget(this, "default_overlap_frames"));
        hideWidget(getWidget(this, "cuts_json"));
        hideWidget(getWidget(this, "ui_refresh"));

        this.addWidget(
          "button",
          nodeData.name === LOOP_CUTTER_NODE ? "Select Frames To Remove" : "Select Loop Cuts",
          "",
          async () => {
            try {
              await openLoopCutPlanner(this);
            } catch (error) {
              console.error("Lostless loop cut planner error:", error);
            }
          },
          { serialize: false }
        );

        return result;
      };
    }
  },
});

