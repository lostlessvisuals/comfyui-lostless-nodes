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

function buildDefaultCut(frame, transitionFrames) {
  return {
    start: Math.max(0, Number(frame) || 0),
    end: Math.max(0, Number(frame) || 0),
    transition_frames: Math.max(0, Number(transitionFrames) || 0),
  };
}

async function openLoopCutPlanner(node) {
  ensureLoopPlannerStyles();

  const videoPathWidget = getWidget(node, "video_path");
  const cutsJsonWidget = getWidget(node, "cuts_json");
  const videoInfoWidget = getWidget(node, "video_info_json");
  const refreshWidget = getWidget(node, "ui_refresh");
  const defaultTransitionWidget = getWidget(node, "default_transition_frames");

  if (!videoPathWidget || !cutsJsonWidget || !defaultTransitionWidget) {
    throw new Error("Loop planner widgets are missing.");
  }

  const inferredVideoPath = getConnectedVhsVideoPath(node);
  if ((!videoPathWidget.value || !String(videoPathWidget.value).trim()) && inferredVideoPath) {
    setWidgetValue(videoPathWidget, inferredVideoPath, node);
  }

  const state = {
    path: String(videoPathWidget.value || inferredVideoPath || ""),
    meta: parseJsonSafe(String(videoInfoWidget?.value || "{}"), {}),
    cuts: parseJsonSafe(String(cutsJsonWidget.value || "[]"), []),
    currentFrame: 0,
    pendingStart: null,
    pendingEnd: null,
    frameUrl: null,
    frameRequestId: 0,
    loadingFrame: false,
    frameCache: new Map(),
    frameCacheOrder: [],
    maxCachedFrames: 80,
    sliderTimer: null,
    keyHandler: null,
  };

  if (!Array.isArray(state.cuts)) {
    state.cuts = [];
  }
  state.cuts = state.cuts
    .filter((c) => c && typeof c === "object")
    .map((c) => ({
      start: Math.max(0, Number(c.start) || 0),
      end: Math.max(0, Number(c.end) || 0),
      transition_frames: Math.max(
        0,
        Number(c.transition_frames ?? defaultTransitionWidget.value ?? 0) || 0
      ),
    }));

  const overlay = document.createElement("div");
  overlay.className = "lostless-loop-overlay";

  const modal = document.createElement("div");
  modal.className = "lostless-loop-modal";
  overlay.appendChild(modal);

  modal.innerHTML = `
    <div class="lostless-loop-header">
      <span class="lostless-loop-muted">Video</span>
      <input type="text" class="ll-video-path" placeholder="C:/path/to/video.mp4" />
      <button class="lostless-loop-btn ll-load">Load</button>
      <button class="lostless-loop-btn ll-close">Close</button>
    </div>
    <div class="lostless-loop-main">
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
          </div>
          <div class="lostless-loop-row">
            <input type="range" class="ll-frame-slider" min="0" max="0" step="1" value="0" />
          </div>
          <div class="lostless-loop-row">
            <button class="lostless-loop-btn ll-mark-start">Set Start = Current</button>
            <button class="lostless-loop-btn ll-mark-end">Set End = Current</button>
            <button class="lostless-loop-btn primary ll-add-cut">Add Cut</button>
            <button class="lostless-loop-btn ll-add-single">Add 1-Frame Cut</button>
          </div>
          <div class="lostless-loop-row">
            <span class="lostless-loop-muted">Pending Start: <strong class="ll-pending-start">-</strong></span>
            <span class="lostless-loop-muted">Pending End: <strong class="ll-pending-end">-</strong></span>
            <span class="lostless-loop-muted">Keys: Left/Right = frame, Shift+Left/Right = 10, [ = start, ] = end</span>
          </div>
        </div>
      </div>
      <div class="lostless-loop-pane">
        <div class="lostless-loop-cuts">
          <div class="lostless-loop-row">
            <span class="lostless-loop-muted">Cuts</span>
            <button class="lostless-loop-btn ll-sort">Sort</button>
            <button class="lostless-loop-btn danger ll-clear">Clear All</button>
          </div>
          <div class="lostless-loop-cut-head">
            <div>#</div>
            <div>Start</div>
            <div>End</div>
            <div>Transition</div>
            <div></div>
          </div>
          <div class="ll-cut-list"></div>
        </div>
      </div>
    </div>
    <div class="lostless-loop-footer">
      <div class="left">
        <span class="lostless-loop-muted ll-summary">No video loaded</span>
      </div>
      <div class="right">
        <button class="lostless-loop-btn ll-cancel">Cancel</button>
        <button class="lostless-loop-btn primary ll-save">Save Cuts To Node</button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  const pathInput = modal.querySelector(".ll-video-path");
  const loadButton = modal.querySelector(".ll-load");
  const closeButton = modal.querySelector(".ll-close");
  const cancelButton = modal.querySelector(".ll-cancel");
  const saveButton = modal.querySelector(".ll-save");
  const previewImg = modal.querySelector(".ll-preview");
  const statusEl = modal.querySelector(".ll-status");
  const summaryEl = modal.querySelector(".ll-summary");
  const sliderEl = modal.querySelector(".ll-frame-slider");
  const frameInputEl = modal.querySelector(".ll-frame-input");
  const cutListEl = modal.querySelector(".ll-cut-list");
  const pendingStartEl = modal.querySelector(".ll-pending-start");
  const pendingEndEl = modal.querySelector(".ll-pending-end");

  pathInput.value = state.path;

  function cleanup() {
    if (state.frameUrl) {
      URL.revokeObjectURL(state.frameUrl);
      state.frameUrl = null;
    }
    if (state.sliderTimer) {
      clearTimeout(state.sliderTimer);
      state.sliderTimer = null;
    }
    if (state.keyHandler) {
      window.removeEventListener("keydown", state.keyHandler);
      state.keyHandler = null;
    }
    overlay.remove();
  }

  function setStatus(message) {
    statusEl.textContent = String(message || "");
  }

  function getFrameCount() {
    return Math.max(0, Number(state.meta?.frame_count) || 0);
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

  function refreshPendingLabels() {
    pendingStartEl.textContent = state.pendingStart == null ? "-" : String(state.pendingStart);
    pendingEndEl.textContent = state.pendingEnd == null ? "-" : String(state.pendingEnd);
  }

  function sortCutsInPlace() {
    state.cuts.sort((a, b) => {
      const sa = Number(a.start) || 0;
      const sb = Number(b.start) || 0;
      if (sa !== sb) return sa - sb;
      return (Number(a.end) || 0) - (Number(b.end) || 0);
    });
  }

  function updateSummary() {
    const frameCount = getFrameCount();
    const totalRemoved = state.cuts.reduce((sum, cut) => {
      const start = Number(cut.start) || 0;
      const end = Number(cut.end) || 0;
      return sum + Math.max(0, end - start + 1);
    }, 0);
    const metaText = frameCount
      ? `${frameCount} frames | ${Number(state.meta?.fps || 0).toFixed(3)} fps | ${state.meta?.width || 0}x${state.meta?.height || 0}`
      : "No video loaded";
    const finalCount = frameCount ? Math.max(0, frameCount - totalRemoved) : "?";
    summaryEl.textContent = `${metaText} | cuts: ${state.cuts.length} | removed: ${totalRemoved} | final: ${finalCount}`;
  }

  function cacheFrameBlob(frameIndex, blob) {
    state.frameCache.set(frameIndex, blob);
    state.frameCacheOrder = state.frameCacheOrder.filter((n) => n !== frameIndex);
    state.frameCacheOrder.push(frameIndex);
    while (state.frameCacheOrder.length > state.maxCachedFrames) {
      const oldest = state.frameCacheOrder.shift();
      if (oldest != null) {
        state.frameCache.delete(oldest);
      }
    }
  }

  async function getCachedFrameBlob(frameIndex) {
    if (state.frameCache.has(frameIndex)) {
      return state.frameCache.get(frameIndex);
    }
    const blob = await fetchVideoFrameBlob(pathInput.value, frameIndex, 640);
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

  function renderCutRows() {
    cutListEl.innerHTML = "";
    state.cuts.forEach((cut, index) => {
      const row = document.createElement("div");
      row.className = "lostless-loop-cut-row";
      row.innerHTML = `
        <div>${index + 1}</div>
        <input type="number" min="0" step="1" value="${Math.max(0, Number(cut.start) || 0)}" data-field="start" />
        <input type="number" min="0" step="1" value="${Math.max(0, Number(cut.end) || 0)}" data-field="end" />
        <input type="number" min="0" step="1" value="${Math.max(0, Number(cut.transition_frames) || 0)}" data-field="transition_frames" />
        <button class="lostless-loop-btn danger" data-action="delete">Delete</button>
      `;

      row.querySelectorAll("input").forEach((input) => {
        input.addEventListener("change", () => {
          const field = input.dataset.field;
          cut[field] = Math.max(0, Number(input.value) || 0);
          updateSummary();
        });
      });

      row.querySelector("[data-action='delete']")?.addEventListener("click", () => {
        state.cuts.splice(index, 1);
        renderCutRows();
        updateSummary();
      });

      cutListEl.appendChild(row);
    });
  }

  async function renderFrame(frameIndex) {
    const frameCount = getFrameCount();
    if (!frameCount) {
      return;
    }

    state.currentFrame = clamp(frameIndex, 0, frameCount - 1);
    syncFrameControls();
    setStatus(`Loading frame ${state.currentFrame}...`);

    const reqId = ++state.frameRequestId;
    try {
      const blob = await getCachedFrameBlob(state.currentFrame);
      if (reqId !== state.frameRequestId) {
        return;
      }
      if (state.frameUrl) {
        URL.revokeObjectURL(state.frameUrl);
      }
      state.frameUrl = URL.createObjectURL(blob);
      previewImg.src = state.frameUrl;
      setStatus(`Frame ${state.currentFrame} / ${frameCount - 1}`);
      prefetchNeighborFrames(state.currentFrame);
    } catch (error) {
      if (reqId !== state.frameRequestId) {
        return;
      }
      setStatus(error?.message || String(error));
    }
  }

  async function loadVideo() {
    const path = String(pathInput.value || "").trim();
    if (!path) {
      setStatus("Enter a video path first.");
      return;
    }

    setStatus("Loading video metadata...");
    try {
      const meta = await fetchVideoMeta(path);
      state.meta = {
        path: meta.path,
        frame_count: Number(meta.frame_count) || 0,
        fps: Number(meta.fps) || 0,
        width: Number(meta.width) || 0,
        height: Number(meta.height) || 0,
      };
      state.path = path;
      syncFrameControls();
      refreshPendingLabels();
      updateSummary();
      await renderFrame(state.currentFrame);
    } catch (error) {
      setStatus(error?.message || String(error));
    }
  }

  function addPendingCut(singleFrame = false) {
    const defaultTransition = Math.max(0, Number(defaultTransitionWidget.value) || 0);
    if (singleFrame) {
      state.cuts.push(buildDefaultCut(state.currentFrame, defaultTransition));
      renderCutRows();
      updateSummary();
      return;
    }

    if (state.pendingStart == null || state.pendingEnd == null) {
      setStatus("Set both Start and End before adding a cut.");
      return;
    }

    const start = Math.min(state.pendingStart, state.pendingEnd);
    const end = Math.max(state.pendingStart, state.pendingEnd);
    state.cuts.push({
      start,
      end,
      transition_frames: defaultTransition,
    });
    state.pendingStart = null;
    state.pendingEnd = null;
    refreshPendingLabels();
    renderCutRows();
    updateSummary();
    setStatus(`Added cut [${start}, ${end}]`);
  }

  function validateCutsForSave() {
    const frameCount = getFrameCount();
    const normalized = state.cuts.map((cut, i) => {
      const start = Math.max(0, Number(cut.start) || 0);
      const end = Math.max(0, Number(cut.end) || 0);
      const transition = Math.max(0, Number(cut.transition_frames) || 0);
      if (end < start) {
        throw new Error(`Cut ${i + 1}: end must be >= start`);
      }
      if (frameCount && (start >= frameCount || end >= frameCount)) {
        throw new Error(`Cut ${i + 1}: frame range [${start}, ${end}] exceeds 0..${frameCount - 1}`);
      }
      return { start, end, transition_frames: transition };
    });

    normalized.sort((a, b) => (a.start - b.start) || (a.end - b.end));
    for (let i = 1; i < normalized.length; i++) {
      if (normalized[i].start <= normalized[i - 1].end) {
        throw new Error(
          `Cuts overlap: [${normalized[i - 1].start}, ${normalized[i - 1].end}] and [${normalized[i].start}, ${normalized[i].end}]`
        );
      }
    }
    return normalized;
  }

  function saveToNode() {
    const normalizedCuts = validateCutsForSave();
    const videoInfo = {
      path: String(pathInput.value || "").trim(),
      frame_count: Number(state.meta?.frame_count) || 0,
      fps: Number(state.meta?.fps) || 0,
      width: Number(state.meta?.width) || 0,
      height: Number(state.meta?.height) || 0,
    };

    setWidgetValue(videoPathWidget, videoInfo.path, node);
    setWidgetValue(cutsJsonWidget, JSON.stringify(normalizedCuts, null, 2), node);
    if (videoInfoWidget) {
      setWidgetValue(videoInfoWidget, JSON.stringify(videoInfo), node);
    }

    if (refreshWidget) {
      const nextValue = (Number(refreshWidget.value) || 0) + 1;
      setWidgetValue(refreshWidget, nextValue, node);
    }

    markNodeDirty(node);
    cleanup();
  }

  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) {
      cleanup();
    }
  });

  closeButton.addEventListener("click", cleanup);
  cancelButton.addEventListener("click", cleanup);
  loadButton.addEventListener("click", () => {
    loadVideo();
  });

  modal.querySelectorAll(".ll-step").forEach((button) => {
    button.addEventListener("click", () => {
      const step = Number(button.dataset.step || 0);
      renderFrame(state.currentFrame + step);
    });
  });

  sliderEl.addEventListener("input", () => {
    if (state.sliderTimer) {
      clearTimeout(state.sliderTimer);
    }
    const target = Number(sliderEl.value) || 0;
    state.sliderTimer = setTimeout(() => {
      renderFrame(target);
    }, 40);
  });

  frameInputEl.addEventListener("change", () => {
    renderFrame(Number(frameInputEl.value) || 0);
  });

  modal.querySelector(".ll-mark-start")?.addEventListener("click", () => {
    state.pendingStart = state.currentFrame;
    refreshPendingLabels();
    setStatus(`Pending start set to ${state.pendingStart}`);
  });

  modal.querySelector(".ll-mark-end")?.addEventListener("click", () => {
    state.pendingEnd = state.currentFrame;
    refreshPendingLabels();
    setStatus(`Pending end set to ${state.pendingEnd}`);
  });

  modal.querySelector(".ll-add-cut")?.addEventListener("click", () => addPendingCut(false));
  modal.querySelector(".ll-add-single")?.addEventListener("click", () => addPendingCut(true));

  modal.querySelector(".ll-clear")?.addEventListener("click", () => {
    state.cuts = [];
    renderCutRows();
    updateSummary();
  });

  modal.querySelector(".ll-sort")?.addEventListener("click", () => {
    sortCutsInPlace();
    renderCutRows();
    updateSummary();
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
    if (event.key === "[") {
      event.preventDefault();
      state.pendingStart = state.currentFrame;
      refreshPendingLabels();
      setStatus(`Pending start set to ${state.pendingStart}`);
      return;
    }
    if (event.key === "]") {
      event.preventDefault();
      state.pendingEnd = state.currentFrame;
      refreshPendingLabels();
      setStatus(`Pending end set to ${state.pendingEnd}`);
      return;
    }
    if (event.key.toLowerCase() === "a") {
      event.preventDefault();
      addPendingCut(false);
    }
  };
  window.addEventListener("keydown", state.keyHandler);

  refreshPendingLabels();
  renderCutRows();
  updateSummary();

  if (state.meta && Number(state.meta.frame_count) > 0) {
    state.currentFrame = clamp(state.currentFrame, 0, Number(state.meta.frame_count) - 1);
    syncFrameControls();
    renderFrame(state.currentFrame);
  } else if (state.path) {
    loadVideo();
  } else {
    syncFrameControls();
    setStatus("Load a video to start selecting cut ranges.");
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

