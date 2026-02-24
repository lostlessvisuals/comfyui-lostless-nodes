import importlib.util
import json
import os
import random
import sys
from io import BytesIO
from mimetypes import guess_type
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from PIL import Image, ImageOps
from aiohttp import web

from server import PromptServer

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

DEFAULT_EXTENSIONS = ".png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff"
LOOP_PLAN_VERSION = 1


def _load_embedded_lostless_mappings():
    """Load node mappings from the embedded Lostless Mask Editor package."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(base_dir, "Lostless-Mask-Editor")
    init_file = os.path.join(package_dir, "__init__.py")
    module_name = "lostless_embedded_mask_editor"

    if not os.path.isfile(init_file):
        return {}, {}

    try:
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(
                module_name,
                init_file,
                submodule_search_locations=[package_dir],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to create import spec for {init_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        class_mappings = getattr(module, "NODE_CLASS_MAPPINGS", {}) or {}
        display_mappings = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}

        allowed_embedded_nodes = {"MaskEditor", "WANVaceImageToMask"}
        filtered_class_mappings = {
            key: value for key, value in class_mappings.items() if key in allowed_embedded_nodes
        }
        filtered_display_mappings = {
            key: value for key, value in display_mappings.items() if key in allowed_embedded_nodes
        }

        print(
            "[Lostless Nodes] Loaded embedded Lostless mask editor mappings: "
            f"{len(filtered_class_mappings)} exposed / {len(class_mappings)} available"
        )
        return filtered_class_mappings, filtered_display_mappings
    except Exception as e:
        print(f"[Lostless Nodes] Failed to load embedded Lostless mask editor package: {e}")
        return {}, {}


def _normalize_extensions(raw_extensions: str) -> Set[str]:
    values = [x.strip().lower() for x in raw_extensions.split(",") if x.strip()]
    normalized = set()
    for value in values:
        if not value.startswith("."):
            value = f".{value}"
        normalized.add(value)
    return normalized or {".png", ".jpg", ".jpeg"}


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path.strip())))


def _is_image_file(file_path: str, extensions: Set[str]) -> bool:
    _, ext = os.path.splitext(file_path)
    return ext.lower() in extensions


def _list_image_files(folder_path: str, recursive: bool, extensions: Set[str]) -> List[str]:
    folder = _expand_path(folder_path)
    if not os.path.isdir(folder):
        return []

    found = []
    if recursive:
        for root, _, files in os.walk(folder):
            for name in files:
                file_path = os.path.join(root, name)
                if _is_image_file(file_path, extensions):
                    found.append(file_path)
    else:
        for name in os.listdir(folder):
            file_path = os.path.join(folder, name)
            if os.path.isfile(file_path) and _is_image_file(file_path, extensions):
                found.append(file_path)

    found.sort()
    return found


def _load_image_tensor(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    if image.mode == "I":
        image = image.point(lambda value: value * (1 / 255))
    image = image.convert("RGB")

    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array)[None,]


def _ensure_image_batch(images: torch.Tensor) -> torch.Tensor:
    if images is None:
        raise ValueError("Expected IMAGE input, got None.")
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Expected IMAGE batch [B,H,W,C], got {tuple(images.shape)}")
    if int(images.shape[0]) < 1:
        raise ValueError("IMAGE batch must contain at least one frame.")
    return images


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))


def _load_video_meta(video_path: str) -> Dict[str, Any]:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for Lostless loop cut video preview routes. Install opencv-python in the ComfyUI venv."
        )

    path = _expand_path(video_path)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if frame_count <= 0:
            counted = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                counted += 1
            frame_count = counted
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        duration_seconds = (frame_count / fps) if fps and frame_count > 0 else 0.0
        return {
            "path": path,
            "frame_count": int(frame_count),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "duration_seconds": float(duration_seconds),
        }
    finally:
        cap.release()


def _read_video_frame_png(video_path: str, frame_index: int, max_width: int = 960) -> bytes:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for Lostless loop cut video preview routes. Install opencv-python in the ComfyUI venv."
        )

    meta = _load_video_meta(video_path)
    frame_count = int(meta["frame_count"])
    if frame_count < 1:
        raise RuntimeError("Video has no frames.")

    index = _clamp_int(frame_index, 0, frame_count - 1)
    path = str(meta["path"])
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {index} from video.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        max_width = max(64, int(max_width))
        if w > max_width:
            scale = max_width / float(w)
            new_size = (max_width, max(1, int(round(h * scale))))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Failed to encode preview frame.")
        return bytes(encoded.tobytes())
    finally:
        cap.release()


def _parse_json_string(value: str, default: Any) -> Any:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def _normalize_cuts(
    cuts: Any,
    frame_count: Optional[int] = None,
    default_transition_frames: int = 0,
) -> List[Dict[str, int]]:
    if cuts is None:
        return []
    if not isinstance(cuts, list):
        raise ValueError("cuts must be a JSON array")

    normalized: List[Dict[str, int]] = []
    for idx, item in enumerate(cuts):
        if not isinstance(item, dict):
            raise ValueError(f"Cut #{idx + 1} must be an object")

        start = _safe_int(item.get("start"), -1)
        end = _safe_int(item.get("end"), -1)
        if start < 0 or end < 0:
            raise ValueError(f"Cut #{idx + 1} must have non-negative start/end")
        if end < start:
            raise ValueError(f"Cut #{idx + 1} end must be >= start")

        if frame_count is not None and frame_count > 0:
            if start >= frame_count or end >= frame_count:
                raise ValueError(
                    f"Cut #{idx + 1} [{start}, {end}] is outside video frame range 0..{frame_count - 1}"
                )

        transition_frames = item.get("transition_frames")
        if transition_frames is None or transition_frames == "":
            transition_value = int(default_transition_frames)
        else:
            transition_value = max(0, _safe_int(transition_frames, int(default_transition_frames)))

        normalized.append(
            {
                "start": int(start),
                "end": int(end),
                "transition_frames": int(transition_value),
                "remove_frames": int(end - start + 1),
            }
        )

    normalized.sort(key=lambda x: (x["start"], x["end"]))
    for i in range(1, len(normalized)):
        prev = normalized[i - 1]
        cur = normalized[i]
        if cur["start"] <= prev["end"]:
            raise ValueError(
                "Cuts must not overlap. "
                f"Found overlap between [{prev['start']}, {prev['end']}] and [{cur['start']}, {cur['end']}]."
            )

    return normalized


def _coerce_cut_plan(cut_plan: Any) -> Dict[str, Any]:
    if isinstance(cut_plan, dict):
        return cut_plan
    if isinstance(cut_plan, str):
        parsed = _parse_json_string(cut_plan, {})
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("cut_plan must be a LOSTLESS_CUT_PLAN object or JSON string")


def _build_loop_plan(
    video_path: str,
    cuts_json: str,
    video_info_json: str,
    default_transition_frames: int,
    default_source_frames: int,
    default_overlap_frames: int,
) -> Dict[str, Any]:
    info = _parse_json_string(video_info_json, {})
    if info is None:
        info = {}
    if not isinstance(info, dict):
        raise ValueError("video_info_json must be a JSON object")

    if video_path:
        info["path"] = _expand_path(video_path)

    frame_count = info.get("frame_count")
    frame_count_int: Optional[int] = None
    if frame_count is not None:
        frame_count_int = _safe_int(frame_count, 0)
        if frame_count_int <= 0:
            frame_count_int = None

    raw_cuts = _parse_json_string(cuts_json, [])
    normalized_cuts = _normalize_cuts(
        raw_cuts,
        frame_count=frame_count_int,
        default_transition_frames=max(0, _safe_int(default_transition_frames, 0)),
    )

    total_removed = sum(int(c["remove_frames"]) for c in normalized_cuts)
    schedule: List[Dict[str, int]] = []
    removed_before = 0
    for i, cut in enumerate(normalized_cuts):
        remove_frames = int(cut["remove_frames"])
        schedule.append(
            {
                "cut_index": i,
                "original_start": int(cut["start"]),
                "original_end": int(cut["end"]),
                "remove_frames": remove_frames,
                "reindexed_start_after_prior_removals": int(cut["start"] - removed_before),
                "reindexed_end_after_prior_removals": int(cut["end"] - removed_before),
                "transition_frames": int(cut["transition_frames"]),
            }
        )
        removed_before += remove_frames

    final_frame_count = None
    if frame_count_int is not None:
        final_frame_count = max(0, frame_count_int - total_removed)

    return {
        "kind": "lostless_loop_cut_plan",
        "version": LOOP_PLAN_VERSION,
        "video": {
            "path": str(info.get("path", "") or ""),
            "frame_count": frame_count_int,
            "fps": float(info.get("fps", 0.0) or 0.0),
            "width": _safe_int(info.get("width"), 0),
            "height": _safe_int(info.get("height"), 0),
        },
        "defaults": {
            "transition_frames": max(0, _safe_int(default_transition_frames, 0)),
            "source_frames": max(1, _safe_int(default_source_frames, 16)),
            "overlap_frames": max(0, _safe_int(default_overlap_frames, 0)),
        },
        "cuts": normalized_cuts,
        "stats": {
            "cut_count": len(normalized_cuts),
            "total_removed_frames": int(total_removed),
            "final_frame_count": final_frame_count,
        },
        "schedule": schedule,
    }


def _plan_to_json(plan: Dict[str, Any]) -> str:
    return json.dumps(plan, indent=2, sort_keys=False)


def _plan_summary(plan: Dict[str, Any]) -> str:
    stats = plan.get("stats", {}) if isinstance(plan, dict) else {}
    video = plan.get("video", {}) if isinstance(plan, dict) else {}
    cut_count = _safe_int(stats.get("cut_count"), 0)
    total_removed = _safe_int(stats.get("total_removed_frames"), 0)
    frame_count = video.get("frame_count")
    final_frame_count = stats.get("final_frame_count")
    frame_part = ""
    if isinstance(frame_count, int) and frame_count > 0:
        frame_part = f" | frames {frame_count} -> {final_frame_count if final_frame_count is not None else '?'}"
    return f"cuts={cut_count} | removed={total_removed}{frame_part}"


def _safe_slice_or_boundary(images: torch.Tensor, start: int, end: int, fallback_index: int) -> torch.Tensor:
    frame_count = int(images.shape[0])
    start = max(0, min(frame_count, int(start)))
    end = max(0, min(frame_count, int(end)))
    if end > start:
        return images[start:end].contiguous()
    idx = _clamp_int(fallback_index, 0, frame_count - 1)
    return images[idx : idx + 1].contiguous()


class LostlessRandomImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "C:/path/to/images",
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Scan Subfolders: ON",
                        "label_off": "Scan Subfolders: OFF",
                    },
                ),
                "allowed_extensions": (
                    "STRING",
                    {
                        "default": DEFAULT_EXTENSIONS,
                        "multiline": False,
                    },
                ),
                "selected_filename": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Set by Randomize Image button",
                    },
                ),
                "trigger": ("INT", {"default": 0}),
                "selected_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "selected_filename")
    FUNCTION = "load_image"
    CATEGORY = "lostless/nodes"

    def load_image(
        self,
        folder_path: str,
        recursive: bool,
        allowed_extensions: str,
        selected_filename: str,
        trigger: int,
        selected_path: str,
    ):
        del selected_filename
        del trigger

        extensions = _normalize_extensions(allowed_extensions)
        images = _list_image_files(folder_path, recursive, extensions)

        if not images:
            raise FileNotFoundError(
                f"No images found in '{_expand_path(folder_path)}' with extensions: {sorted(extensions)}"
            )

        chosen = _expand_path(selected_path) if selected_path else ""
        if not chosen or not os.path.isfile(chosen) or not _is_image_file(chosen, extensions):
            chosen = images[0]

        image_tensor = _load_image_tensor(chosen)
        return (image_tensor, os.path.basename(chosen))


class LostlessRandomizeButton:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pulse": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "emit"
    CATEGORY = "lostless/nodes"

    def emit(self, pulse: int):
        return (pulse,)


class LostlessBufferNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["LTX (8n+1)", "WAN (4n+1)"], {"default": "LTX (8n+1)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "buffer"
    CATEGORY = "lostless/nodes"

    def buffer(self, images: torch.Tensor, mode: str):
        images = _ensure_image_batch(images)

        frame_count = int(images.shape[0])
        step = 8 if mode.startswith("LTX") else 4
        remainder = (frame_count - 1) % step
        target_count = frame_count if remainder == 0 else frame_count + (step - remainder)
        pad_count = target_count - frame_count

        if pad_count <= 0:
            return (images.contiguous(),)

        last_frame = images[-1:].repeat(pad_count, 1, 1, 1)
        buffered = torch.cat([images, last_frame], dim=0)
        return (buffered.contiguous(),)


class LostlessLoopCutPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "C:/path/to/video.mp4",
                    },
                ),
                "default_transition_frames": ("INT", {"default": 12, "min": 0, "max": 9999}),
                "default_source_frames": ("INT", {"default": 16, "min": 1, "max": 9999}),
                "default_overlap_frames": ("INT", {"default": 8, "min": 0, "max": 9999}),
                "cuts_json": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "placeholder": "Set via Select Loop Cuts button",
                    },
                ),
                "video_info_json": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": False,
                    },
                ),
                "ui_refresh": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("LOSTLESS_CUT_PLAN", "STRING", "INT", "STRING")
    RETURN_NAMES = ("cut_plan", "cut_plan_json", "cut_count", "summary")
    FUNCTION = "build_plan"
    CATEGORY = "lostless/loop"

    def build_plan(
        self,
        video_path: str,
        default_transition_frames: int,
        default_source_frames: int,
        default_overlap_frames: int,
        cuts_json: str,
        video_info_json: str,
        ui_refresh: int,
    ):
        del ui_refresh
        plan = _build_loop_plan(
            video_path=video_path,
            cuts_json=cuts_json,
            video_info_json=video_info_json,
            default_transition_frames=default_transition_frames,
            default_source_frames=default_source_frames,
            default_overlap_frames=default_overlap_frames,
        )
        return (plan, _plan_to_json(plan), int(len(plan.get("cuts", []))), _plan_summary(plan))


class LostlessLoopCutTask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "cut_plan": ("LOSTLESS_CUT_PLAN",),
                "cut_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "start_frame",
        "end_frame",
        "pre_context",
        "post_context",
        "remove_frames",
        "transition_frames",
        "source_frames",
        "overlap_frames",
        "cut_start",
        "cut_end",
        "cut_json",
    )
    FUNCTION = "extract_cut"
    CATEGORY = "lostless/loop"

    def extract_cut(self, images: torch.Tensor, cut_plan: Any, cut_index: int):
        images = _ensure_image_batch(images)
        plan = _coerce_cut_plan(cut_plan)
        cuts = plan.get("cuts") or []
        if not isinstance(cuts, list) or not cuts:
            raise ValueError("Cut plan contains no cuts.")

        idx = _safe_int(cut_index, 0)
        if idx < 0 or idx >= len(cuts):
            raise ValueError(f"cut_index {idx} is out of range for {len(cuts)} cuts")

        cut = cuts[idx]
        if not isinstance(cut, dict):
            raise ValueError("Cut entry is invalid")

        frame_count = int(images.shape[0])
        start = _safe_int(cut.get("start"), -1)
        end = _safe_int(cut.get("end"), -1)
        if start < 0 or end < start or end >= frame_count:
            raise ValueError(
                f"Cut [{start}, {end}] is invalid for IMAGE batch with {frame_count} frames"
            )

        defaults = plan.get("defaults") or {}
        source_frames = max(1, _safe_int(defaults.get("source_frames"), 16))
        overlap_frames = max(0, _safe_int(defaults.get("overlap_frames"), 0))
        transition_frames = max(
            0,
            _safe_int(cut.get("transition_frames"), _safe_int(defaults.get("transition_frames"), 0)),
        )
        context_span = source_frames + overlap_frames

        pre_context = _safe_slice_or_boundary(images, start - context_span, start, start)
        post_context = _safe_slice_or_boundary(images, end + 1, end + 1 + context_span, end)

        start_frame = images[start : start + 1].contiguous()
        end_frame = images[end : end + 1].contiguous()
        cut_payload = {
            "cut_index": idx,
            "start": start,
            "end": end,
            "remove_frames": int(end - start + 1),
            "transition_frames": transition_frames,
            "source_frames": source_frames,
            "overlap_frames": overlap_frames,
        }

        return (
            start_frame,
            end_frame,
            pre_context,
            post_context,
            int(end - start + 1),
            int(transition_frames),
            int(source_frames),
            int(overlap_frames),
            int(start),
            int(end),
            json.dumps(cut_payload),
        )


class LostlessLoopApplyCutPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "cut_plan": ("LOSTLESS_CUT_PLAN",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("images", "normalized_plan_json", "schedule_json", "final_frame_count")
    FUNCTION = "apply_plan"
    CATEGORY = "lostless/loop"

    def apply_plan(self, images: torch.Tensor, cut_plan: Any):
        images = _ensure_image_batch(images)
        plan = _coerce_cut_plan(cut_plan)

        video = plan.get("video") or {}
        defaults = plan.get("defaults") or {}
        frame_count = int(images.shape[0])
        cuts = _normalize_cuts(
            plan.get("cuts") or [],
            frame_count=frame_count,
            default_transition_frames=_safe_int(defaults.get("transition_frames"), 0),
        )

        rebuilt_plan = {
            "kind": plan.get("kind", "lostless_loop_cut_plan"),
            "version": _safe_int(plan.get("version"), LOOP_PLAN_VERSION),
            "video": {
                "path": str(video.get("path", "") or ""),
                "frame_count": frame_count,
                "fps": float(video.get("fps", 0.0) or 0.0),
                "width": _safe_int(video.get("width"), int(images.shape[2])),
                "height": _safe_int(video.get("height"), int(images.shape[1])),
            },
            "defaults": {
                "transition_frames": max(0, _safe_int(defaults.get("transition_frames"), 0)),
                "source_frames": max(1, _safe_int(defaults.get("source_frames"), 16)),
                "overlap_frames": max(0, _safe_int(defaults.get("overlap_frames"), 0)),
            },
            "cuts": cuts,
        }

        kept_segments: List[torch.Tensor] = []
        kept_spans: List[Dict[str, int]] = []
        schedule: List[Dict[str, int]] = []
        cursor = 0
        removed_before = 0

        for idx, cut in enumerate(cuts):
            start = int(cut["start"])
            end = int(cut["end"])
            if start > cursor:
                kept_segments.append(images[cursor:start])
                kept_spans.append({"start": cursor, "end": start - 1, "count": start - cursor})

            remove_count = int(end - start + 1)
            schedule.append(
                {
                    "cut_index": idx,
                    "original_start": start,
                    "original_end": end,
                    "remove_frames": remove_count,
                    "transition_frames": int(cut.get("transition_frames", 0)),
                    "reindexed_start_after_prior_removals": start - removed_before,
                    "reindexed_end_after_prior_removals": end - removed_before,
                }
            )
            removed_before += remove_count
            cursor = end + 1

        if cursor < frame_count:
            kept_segments.append(images[cursor:frame_count])
            kept_spans.append({"start": cursor, "end": frame_count - 1, "count": frame_count - cursor})

        if not kept_segments:
            raise ValueError("Cut plan removes all frames. At least one frame must remain.")

        pruned = torch.cat([seg for seg in kept_segments if int(seg.shape[0]) > 0], dim=0).contiguous()
        rebuilt_plan["stats"] = {
            "cut_count": len(cuts),
            "total_removed_frames": int(sum(int(c["remove_frames"]) for c in cuts)),
            "final_frame_count": int(pruned.shape[0]),
        }
        rebuilt_plan["schedule"] = schedule

        meta = {
            "kept_spans": kept_spans,
            "removed_cuts": cuts,
            "frame_count_before": frame_count,
            "frame_count_after": int(pruned.shape[0]),
        }
        return (pruned, _plan_to_json(rebuilt_plan), json.dumps(meta, indent=2), int(pruned.shape[0]))


def _register_routes() -> None:
    routes = PromptServer.instance.routes

    @routes.post("/lostless/random-image")
    async def lostless_random_image(request):
        payload = await request.json()

        folder_path = str(payload.get("folder_path", ""))
        recursive = bool(payload.get("recursive", True))
        allowed_extensions = str(payload.get("allowed_extensions", DEFAULT_EXTENSIONS))

        extensions = _normalize_extensions(allowed_extensions)
        images = _list_image_files(folder_path, recursive, extensions)

        if not images:
            return web.json_response(
                {
                    "ok": False,
                    "error": "No matching images were found for the provided path/extensions.",
                },
                status=404,
            )

        selected = random.choice(images)
        return web.json_response(
            {
                "ok": True,
                "path": selected,
                "filename": os.path.basename(selected),
                "count": len(images),
            }
        )

    @routes.post("/lostless/image-preview")
    async def lostless_image_preview(request):
        payload = await request.json()
        path = str(payload.get("path", ""))
        full_path = _expand_path(path)

        if not full_path or not os.path.isfile(full_path):
            return web.json_response({"ok": False, "error": "Image file not found."}, status=404)

        mime_type, _ = guess_type(full_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with Image.open(full_path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")

            buf = BytesIO()
            image.save(buf, format="PNG")
            return web.Response(body=buf.getvalue(), content_type="image/png")

    @routes.post("/lostless/video-meta")
    async def lostless_video_meta(request):
        try:
            payload = await request.json()
            path = str(payload.get("path", ""))
            meta = _load_video_meta(path)
            return web.json_response({"ok": True, **meta})
        except FileNotFoundError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=404)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)

    @routes.post("/lostless/video-frame")
    async def lostless_video_frame(request):
        try:
            payload = await request.json()
            path = str(payload.get("path", ""))
            frame_index = _safe_int(payload.get("frame_index"), 0)
            max_width = _safe_int(payload.get("max_width"), 960)
            image_bytes = _read_video_frame_png(path, frame_index, max_width=max_width)
            return web.Response(body=image_bytes, content_type="image/png")
        except FileNotFoundError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=404)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)


try:
    _register_routes()
except Exception:
    pass


NODE_CLASS_MAPPINGS = {
    "LostlessRandomImage": LostlessRandomImage,
    "LostlessRandomizeButton": LostlessRandomizeButton,
    "LostlessBufferNode": LostlessBufferNode,
    "LostlessLoopCutPlanner": LostlessLoopCutPlanner,
    "LostlessLoopCutTask": LostlessLoopCutTask,
    "LostlessLoopApplyCutPlan": LostlessLoopApplyCutPlan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LostlessRandomImage": "Lostless Random Image",
    "LostlessRandomizeButton": "Lostless Randomize Button",
    "LostlessBufferNode": "Lostless Buffer",
    "LostlessLoopCutPlanner": "Lostless Loop Cut Planner",
    "LostlessLoopCutTask": "Lostless Loop Cut Task",
    "LostlessLoopApplyCutPlan": "Lostless Loop Apply Cut Plan",
}

_EMBEDDED_CLASS_MAPPINGS, _EMBEDDED_DISPLAY_MAPPINGS = _load_embedded_lostless_mappings()
NODE_CLASS_MAPPINGS.update(_EMBEDDED_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_EMBEDDED_DISPLAY_MAPPINGS)
