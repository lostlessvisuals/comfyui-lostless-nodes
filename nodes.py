import os
import random
from io import BytesIO
from mimetypes import guess_type
from typing import List, Set

import numpy as np
import torch
from PIL import Image, ImageOps
from aiohttp import web

from server import PromptServer

DEFAULT_EXTENSIONS = ".png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff"


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

        # Convert very large source images into a bounded PNG preview to keep UI responsive.
        with Image.open(full_path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")

            buf = BytesIO()
            image.save(buf, format="PNG")
            return web.Response(body=buf.getvalue(), content_type="image/png")


try:
    _register_routes()
except Exception:
    # Route may already be registered during hot reload.
    pass


NODE_CLASS_MAPPINGS = {
    "LostlessRandomImage": LostlessRandomImage,
    "LostlessRandomizeButton": LostlessRandomizeButton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LostlessRandomImage": "Lostless Random Image",
    "LostlessRandomizeButton": "Lostless Randomize Button",
}
