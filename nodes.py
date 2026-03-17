import importlib.util
import json
import os
import random
import sys
from io import BytesIO
from mimetypes import guess_type
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from PIL import Image, ImageOps
from aiohttp import web

from server import PromptServer

DEFAULT_EXTENSIONS = ".png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff"


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
            # Tell the embedded package to skip loading its broader upstream node surface.
            module.LOSTLESS_MINIMAL_IMPORT = True
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


def _resolve_selected_image_path(
    folder_path: str,
    recursive: bool,
    extensions: Set[str],
    selected_path: str,
    selected_filename: str = "",
    images: Optional[List[str]] = None,
) -> str:
    chosen = _expand_path(selected_path) if selected_path else ""
    if chosen and os.path.isfile(chosen) and _is_image_file(chosen, extensions):
        return chosen

    image_list = images if images is not None else _list_image_files(folder_path, recursive, extensions)
    if not image_list:
        return ""

    wanted_name = os.path.basename(str(selected_filename or "").strip())
    if wanted_name:
        for image_path in image_list:
            if os.path.basename(image_path) == wanted_name:
                return image_path

    return image_list[0]


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
        del trigger

        extensions = _normalize_extensions(allowed_extensions)
        images = _list_image_files(folder_path, recursive, extensions)

        if not images:
            raise FileNotFoundError(
                f"No images found in '{_expand_path(folder_path)}' with extensions: {sorted(extensions)}"
            )

        chosen = _resolve_selected_image_path(
            folder_path=folder_path,
            recursive=recursive,
            extensions=extensions,
            selected_path=selected_path,
            selected_filename=selected_filename,
            images=images,
        )
        if not chosen:
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

    @routes.post("/lostless/resolve-image")
    async def lostless_resolve_image(request):
        payload = await request.json()

        folder_path = str(payload.get("folder_path", ""))
        recursive = bool(payload.get("recursive", True))
        allowed_extensions = str(payload.get("allowed_extensions", DEFAULT_EXTENSIONS))
        selected_path = str(payload.get("selected_path", ""))
        selected_filename = str(payload.get("selected_filename", ""))

        extensions = _normalize_extensions(allowed_extensions)
        images = _list_image_files(folder_path, recursive, extensions)
        chosen = _resolve_selected_image_path(
            folder_path=folder_path,
            recursive=recursive,
            extensions=extensions,
            selected_path=selected_path,
            selected_filename=selected_filename,
            images=images,
        )

        if not chosen:
            return web.json_response(
                {
                    "ok": False,
                    "error": "No matching image could be resolved from the saved selection.",
                },
                status=404,
            )

        return web.json_response(
            {
                "ok": True,
                "path": chosen,
                "filename": os.path.basename(chosen),
                "count": len(images),
            }
        )

    @routes.post("/lostless/select-image")
    async def lostless_select_image(request):
        try:
            payload = await request.json()
            folder_path = str(payload.get("folder_path", ""))
            selected_path = str(payload.get("selected_path", ""))
            allowed_extensions = str(payload.get("allowed_extensions", DEFAULT_EXTENSIONS))
            extensions = _normalize_extensions(allowed_extensions)

            import tkinter as tk
            from tkinter import filedialog

            initial_dir = ""
            if selected_path:
                initial_dir = os.path.dirname(_expand_path(selected_path))
            if not initial_dir:
                candidate = _expand_path(folder_path) if folder_path else ""
                if os.path.isdir(candidate):
                    initial_dir = candidate

            patterns = " ".join(f"*{ext}" for ext in sorted(extensions))
            filetypes = [("Image files", patterns)] if patterns else []
            filetypes.append(("All files", "*.*"))

            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass

            try:
                chosen = filedialog.askopenfilename(
                    title="Select image for Lostless Random Image",
                    initialdir=initial_dir or "",
                    filetypes=filetypes,
                )
            finally:
                root.destroy()

            if not chosen:
                return web.json_response({"ok": True, "cancelled": True})

            chosen = _expand_path(chosen)
            if not os.path.isfile(chosen):
                return web.json_response({"ok": False, "error": "Selected image file was not found."}, status=404)
            if not _is_image_file(chosen, extensions):
                return web.json_response(
                    {"ok": False, "error": "Selected file does not match the allowed extensions."},
                    status=400,
                )

            return web.json_response(
                {
                    "ok": True,
                    "cancelled": False,
                    "path": chosen,
                    "filename": os.path.basename(chosen),
                }
            )
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LostlessRandomImage": "Lostless Random Image",
    "LostlessRandomizeButton": "Lostless Randomize Button",
    "LostlessBufferNode": "Lostless Buffer",
}

_EMBEDDED_CLASS_MAPPINGS, _EMBEDDED_DISPLAY_MAPPINGS = _load_embedded_lostless_mappings()
NODE_CLASS_MAPPINGS.update(_EMBEDDED_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_EMBEDDED_DISPLAY_MAPPINGS)
