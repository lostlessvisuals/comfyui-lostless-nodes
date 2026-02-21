"""
ComfyUI WAN Vace Pipeline
Video frame processing nodes for AI interpolation workflows
"""

import os
import sys

# Print debug info about which file is being loaded
current_file = os.path.abspath(__file__)
print(f"[WAN Vace Pipeline] Loading from: {current_file}")
print(f"[WAN Vace Pipeline] Python version: {sys.version}")
print(f"[WAN Vace Pipeline] Module name: {__name__}")
print(f"[WAN Vace Pipeline] Package path: {os.path.dirname(current_file)}")

print("[WAN Vace Pipeline] Loading custom nodes...")

# Import server endpoints
try:
    from . import mask_editor_server
    print("[WAN Vace Pipeline] Mask editor server endpoints loaded")
except Exception as e:
    print(f"[WAN Vace Pipeline] Failed to load mask editor server: {e}")

try:
    from . import outpainting_editor_server
    print("[WAN Vace Pipeline] Outpainting editor server endpoints loaded")
except Exception as e:
    print(f"[WAN Vace Pipeline] Failed to load outpainting editor server: {e}")

# Define mask editor node directly here to ensure it loads
# TRULY PERSISTENT GLOBAL CACHE that survives module reloads
import sys
_CACHE_ATTR_NAME = "mask_editor_global_cache"

def get_persistent_cache():
    """Get cache that persists across ComfyUI module reloads"""
    if not hasattr(sys.modules[__name__], _CACHE_ATTR_NAME):
        setattr(sys.modules[__name__], _CACHE_ATTR_NAME, {})
    return getattr(sys.modules[__name__], _CACHE_ATTR_NAME)

class MaskEditor:
    NOT_IDEMPOTENT = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "edit_mode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "mask_image")
    FUNCTION = "edit_mask"
    CATEGORY = "Mask Editor"

    def _normalize_inputs(self, images, masks):
        import torch

        if images is None or masks is None:
            raise ValueError("MaskEditor requires both `images` and `masks` inputs.")

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        if images.ndim != 4:
            raise ValueError(f"`images` must be IMAGE batch [B,H,W,C], got shape {tuple(images.shape)}")
        if masks.ndim != 3:
            raise ValueError(f"`masks` must be MASK batch [B,H,W], got shape {tuple(masks.shape)}")

        image_frames = int(images.shape[0])
        mask_frames = int(masks.shape[0])
        if image_frames != mask_frames:
            raise ValueError(
                f"Frame count mismatch: images has {image_frames} frames, masks has {mask_frames} frames. "
                "Counts must match exactly."
            )

        if int(images.shape[1]) != int(masks.shape[1]) or int(images.shape[2]) != int(masks.shape[2]):
            raise ValueError(
                f"Spatial mismatch: images are {tuple(images.shape[1:3])}, masks are {tuple(masks.shape[1:3])}. "
                "Height and width must match."
            )

        if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
            raise ValueError("Unexpected non-tensor IMAGE or MASK input.")

        return images, masks

    def _mask_to_bw_image(self, masks):
        import torch

        if masks.ndim == 4:
            if int(masks.shape[-1]) >= 1:
                masks = masks[..., 0]
            else:
                raise ValueError(f"Unexpected mask tensor shape for image conversion: {tuple(masks.shape)}")

        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        elif masks.ndim != 3:
            raise ValueError(f"Expected MASK tensor [B,H,W] for image conversion, got {tuple(masks.shape)}")

        masks = masks.float()
        masks = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=0.0)
        if float(masks.max().item()) > 1.0:
            masks = masks / 255.0
        masks = torch.clamp(masks, 0.0, 1.0)
        masks = (masks >= 0.5).to(torch.float32)
        return masks.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

    def edit_mask(self, images, masks, edit_mode=True):
        import base64
        import cv2
        import json
        import numpy as np
        import subprocess
        import tempfile
        import torch

        images, masks = self._normalize_inputs(images, masks)

        if not edit_mode:
            return (masks, self._mask_to_bw_image(masks))

        frame_count = int(images.shape[0])
        height = int(images.shape[1])
        width = int(images.shape[2])

        temp_dir = tempfile.mkdtemp(prefix="mask_editor_from_comfy_")
        output_dir = os.path.join(temp_dir, "output")
        input_frames_dir = os.path.join(output_dir, "input_frames")
        os.makedirs(input_frames_dir, exist_ok=True)
        mask_frames_payload = {}
        shape_keyframes_payload = {}
        frame_files = []

        images_cpu = images.detach().cpu()
        masks_cpu = masks.detach().cpu()

        for i in range(frame_count):
            frame = images_cpu[i].numpy()
            frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            if frame_u8.ndim == 2:
                frame_u8 = np.stack([frame_u8, frame_u8, frame_u8], axis=-1)
            if frame_u8.shape[-1] == 1:
                frame_u8 = np.repeat(frame_u8, 3, axis=-1)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
            frame_name = f"frame_{i:04d}.png"
            frame_path = os.path.join(input_frames_dir, frame_name)
            if not cv2.imwrite(frame_path, frame_bgr):
                raise RuntimeError(f"Failed to write input frame {i} to {frame_path}")
            frame_files.append(os.path.join("input_frames", frame_name).replace("\\", "/"))

            mask = masks_cpu[i].numpy().astype(np.float32)
            mask_max = float(mask.max()) if mask.size > 0 else 0.0
            if mask_max <= 1.0 + 1e-6:
                mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
            else:
                mask_u8 = np.clip(mask, 0, 255).astype(np.uint8)

            ok, encoded = cv2.imencode(".png", mask_u8)
            if not ok:
                raise RuntimeError(f"Failed to encode input mask for frame {i}")
            mask_frames_payload[str(i)] = base64.b64encode(encoded.tobytes()).decode("utf-8")

            # Convert incoming raster masks to editable vector keyframes.
            binary_mask = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_shapes = []
            for contour in contours:
                if contour is None or len(contour) < 3:
                    continue
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = approx.reshape(-1, 2).tolist()
                if len(vertices) < 3:
                    continue
                frame_shapes.append({
                    "vertices": [[float(x), float(y)] for x, y in vertices],
                    "closed": True,
                    "visible": True,
                    "vertex_count": len(vertices),
                    "is_shape": True,
                })
            if frame_shapes:
                shape_keyframes_payload[str(i)] = frame_shapes

        project_data = {
            "shape_keyframes": shape_keyframes_payload,
            "mask_frames": mask_frames_payload,
            "settings": {
                "drawing_mode": "shape",
                "brush_size": 30,
                "vertex_count": 32,
            },
            "video_info": {
                "path": input_frames_dir,
                "type": "image_sequence",
                "total_frames": frame_count,
                "width": width,
                "height": height,
            },
            "source_video": {
                "path": input_frames_dir,
                "type": "image_sequence",
            },
            "frame_files": frame_files,
            "current_frame": 0,
        }

        print(f"[MaskEditor] Launching editor with frames={frame_count}, image_size={(height, width)}, masks_shape={tuple(masks.shape)}, vector_keyframes={len(shape_keyframes_payload)}")

        config = {
            "input_frames": [],
            "output_dir": output_dir,
            "project_data": json.dumps(project_data),
            "comfy_strict_mode": True,
        }
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        launcher_path = os.path.join(os.path.dirname(__file__), "nodes", "comfyui_mask_launcher.py")
        if not os.path.exists(launcher_path):
            raise RuntimeError(f"Mask editor launcher not found: {launcher_path}")

        proc = subprocess.run(
            [sys.executable, launcher_path, "--config", config_path],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip()[-1000:]
            if proc.returncode == 1:
                raise RuntimeError("Mask editor was cancelled before accepting changes.")
            raise RuntimeError(f"Mask editor failed (exit {proc.returncode}): {stderr_tail}")

        edited_masks_path = os.path.join(output_dir, "edited_masks.npy")
        if not os.path.exists(edited_masks_path):
            raise RuntimeError("Mask editor completed but did not write edited mask output.")

        edited_masks = np.load(edited_masks_path)
        if edited_masks.ndim != 3:
            raise RuntimeError(f"Edited masks must have shape [B,H,W], got {edited_masks.shape}")
        if int(edited_masks.shape[0]) != frame_count:
            raise RuntimeError(
                f"Edited mask frame count mismatch: expected {frame_count}, got {int(edited_masks.shape[0])}"
            )
        if int(edited_masks.shape[1]) != height or int(edited_masks.shape[2]) != width:
            raise RuntimeError(
                f"Edited mask size mismatch: expected {(height, width)}, got {tuple(edited_masks.shape[1:3])}"
            )

        edited_masks_tensor = torch.from_numpy(edited_masks.astype(np.float32) / 255.0)
        edited_mask_image_tensor = self._mask_to_bw_image(edited_masks_tensor)
        return (edited_masks_tensor, edited_mask_image_tensor)
class WANVaceImageToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "mask_image")
    FUNCTION = "convert"
    CATEGORY = "WAN/mask"

    def convert(self, images, threshold=0.5, invert=False):
        import torch

        if images is None:
            raise ValueError("WANVaceImageToMask requires an IMAGE input.")

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected IMAGE batch [B,H,W,C], got {tuple(images.shape)}")

        images = torch.nan_to_num(images.float(), nan=0.0, posinf=1.0, neginf=0.0)
        images = torch.clamp(images, 0.0, 1.0)

        if int(images.shape[-1]) == 1:
            gray = images[..., 0]
        else:
            # ITU-R BT.709 luminance from RGB IMAGE input.
            gray = (0.2126 * images[..., 0]) + (0.7152 * images[..., 1]) + (0.0722 * images[..., 2])

        masks = (gray >= float(threshold)).to(torch.float32)
        if invert:
            masks = 1.0 - masks

        mask_image = masks.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        return (masks.contiguous(), mask_image)

# Define outpainting editor node
class WANVaceOutpaintingEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "canvas_data": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Canvas data from outpainting editor"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "status")
    FUNCTION = "process_outpainting"
    CATEGORY = "WAN/mask"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, canvas_data=""):
        # This helps ComfyUI know when to reprocess
        import hashlib
        if canvas_data:
            return hashlib.md5(canvas_data.encode()).hexdigest()
        return ""
    
    def process_outpainting(self, canvas_data=""):
        import torch
        import json
        import cv2
        import numpy as np
        import os
        import tempfile
        from pathlib import Path
        import glob
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Optimized loading functions for ComfyUI processing
        def load_single_image_comfyui(img_path):
            """Load a single image file - for parallel processing in ComfyUI"""
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img_rgb
            except Exception as e:
                print(f"[WAN Outpainting Editor] Error loading image {img_path}: {e}")
            return None

        def load_video_optimized_comfyui(video_path):
            """Optimized video loading for ComfyUI processing"""
            frames = []
            start_time = time.time()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames
            
            # Get total frame count for progress
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[WAN Outpainting Editor] Video has {total_frames} frames")
            
            # Pre-allocate list for better performance
            frames = [None] * total_frames
            frame_idx = 0
            
            # Process in batches for better performance
            batch_size = 30
            batch_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                batch_frames.append((frame_idx, frame))
                frame_idx += 1
                
                # Process batch
                if len(batch_frames) >= batch_size or frame_idx >= total_frames:
                    for idx, frame in batch_frames:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames[idx] = frame_rgb
                    batch_frames = []
            
            cap.release()
            
            # Remove any None values (in case frame count was wrong)
            frames = [f for f in frames if f is not None]
            
            load_time = time.time() - start_time
            print(f"[WAN Outpainting Editor] Video loaded in {load_time:.2f}s ({len(frames)} frames, {len(frames)/load_time:.1f} fps)")
            
            return frames

        def load_image_sequence_optimized_comfyui(dir_path):
            """Optimized image sequence loading for ComfyUI processing"""
            frames = []
            start_time = time.time()
            
            # Get all valid image files
            image_files = sorted(glob.glob(os.path.join(dir_path, "*.*")))
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            
            valid_files = [
                img_path for img_path in image_files 
                if Path(img_path).suffix.lower() in valid_extensions
            ]
            
            if not valid_files:
                return frames
            
            print(f"[WAN Outpainting Editor] Found {len(valid_files)} image files")
            
            # Use parallel processing for faster loading
            max_workers = min(8, len(valid_files))  # Don't use too many threads
            frames = [None] * len(valid_files)
            completed = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all loading tasks
                future_to_index = {
                    executor.submit(load_single_image_comfyui, img_path): idx 
                    for idx, img_path in enumerate(valid_files)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        if result is not None:
                            frames[idx] = result
                        completed += 1
                        
                        # Progress logging
                        if completed % 100 == 0 or completed == len(valid_files):
                            print(f"[WAN Outpainting Editor] Loaded {completed}/{len(valid_files)} images...")
                            
                    except Exception as e:
                        print(f"[WAN Outpainting Editor] Error loading image {idx}: {e}")
                        completed += 1
            
            # Remove any None values
            frames = [f for f in frames if f is not None]
            
            load_time = time.time() - start_time
            print(f"[WAN Outpainting Editor] Image sequence loaded in {load_time:.2f}s ({len(frames)} images, {len(frames)/load_time:.1f} imgs/s)")
            
            return frames
        
        # Helper for cycling dots
        def get_dot_count():
            counter_file = os.path.join(tempfile.gettempdir(), "wan_outpainting_status_counter.txt")
            try:
                if os.path.exists(counter_file):
                    with open(counter_file, "r") as f:
                        count = int(f.read().strip())
                else:
                    count = 1
                count = (count % 3) + 1
                with open(counter_file, "w") as f:
                    f.write(str(count))
            except Exception:
                count = 1
            return count
        
        dot_count = get_dot_count()
        dots = "." * dot_count
        
        print(f"[WAN Outpainting Editor] process_outpainting called")
        print(f"[WAN Outpainting Editor] canvas_data: {canvas_data[:100] if canvas_data else 'empty'}...")
        
        status = f"Initializing{dots}"
        # Try to load canvas data from editor
        if canvas_data:
            try:
                data = json.loads(canvas_data)
                print(f"[WAN Outpainting Editor] Data keys: {list(data.keys())}")
                
                project_data = data.get("project_data", {})
                if not project_data:
                    print(f"[WAN Outpainting Editor] No project data found")
                    raise ValueError("No project data")
                
                # Extract video info and canvas settings
                video_info = project_data.get("video_info", {})
                canvas_settings = project_data.get("canvas_settings", {})
                
                print(f"[WAN Outpainting Editor] Video info: {video_info}")
                print(f"[WAN Outpainting Editor] Canvas settings: {canvas_settings}")
                
                # Get video properties
                video_path = video_info.get("path")
                video_type = video_info.get("type")
                total_frames = video_info.get("total_frames", 1)
                width = video_info.get("width", 512)
                height = video_info.get("height", 512)
                
                # Get canvas settings
                canvas_width = int(canvas_settings.get("canvas_width", width))
                canvas_height = int(canvas_settings.get("canvas_height", height))
                video_x = int(canvas_settings.get("video_x", 0))
                video_y = int(canvas_settings.get("video_y", 0))
                video_width = int(canvas_settings.get("video_width", width))
                video_height = int(canvas_settings.get("video_height", height))
                feather_amount = int(canvas_settings.get("feather_amount", 0))
                
                # Load video frames using optimized loading
                frames = []
                if video_path and os.path.exists(video_path):
                    if video_type == "video":
                        status = f"Loading Video File{dots}"
                        print(f"[WAN Outpainting Editor] Loading video from: {video_path}")
                        frames = load_video_optimized_comfyui(video_path)
                        if frames:
                            print(f"[WAN Outpainting Editor] Loaded {len(frames)} frames")
                        else:
                            print(f"[WAN Outpainting Editor] Failed to load video")
                    elif video_type == "image_sequence":
                        status = f"Loading Image Sequence{dots}"
                        print(f"[WAN Outpainting Editor] Loading image sequence from: {video_path}")
                        frames = load_image_sequence_optimized_comfyui(video_path)
                        if frames:
                            print(f"[WAN Outpainting Editor] Loaded {len(frames)} frames from image sequence")
                        else:
                            print(f"[WAN Outpainting Editor] Failed to load image sequence")
                else:
                    status = f"No video/image sequence found{dots}"
                
                # Process outpainting
                if frames:
                    status = f"Applying Outpainting{dots}"
                    processed_frames = []
                    masks = []
                    
                    # Get padding color (frame darkness)
                    frame_darkness = 0.0  # Default to black padding
                    
                    # Get actual frame dimensions
                    frame_height, frame_width = frames[0].shape[0], frames[0].shape[1]
                    print(f"[WAN Outpainting Editor] Frame dimensions: {frame_width}x{frame_height}")
                    print(f"[WAN Outpainting Editor] Canvas dimensions: {canvas_width}x{canvas_height}")
                    print(f"[WAN Outpainting Editor] Video position: ({video_x}, {video_y})")
                    print(f"[WAN Outpainting Editor] Video display size: {video_width}x{video_height}")
                    
                    # Pre-calculate regions once (they're the same for all frames)
                    src_x = 0 if video_x >= 0 else -video_x
                    src_y = 0 if video_y >= 0 else -video_y
                    dst_x = max(0, video_x)
                    dst_y = max(0, video_y)
                    
                    # Check if we need to resize frames
                    need_resize = (video_width != frame_width) or (video_height != frame_height)
                    
                    # Calculate copy dimensions once
                    copy_width = min(frame_width - src_x, canvas_width - dst_x)
                    copy_height = min(frame_height - src_y, canvas_height - dst_y)
                    
                    print(f"[WAN Outpainting Editor] Copy region: src=({src_x},{src_y}), dst=({dst_x},{dst_y}), size={copy_width}x{copy_height}")
                    
                    # Create base mask once (same for all frames)
                    base_mask = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255  # Start with all white
                    
                    # Use video dimensions for mask if resizing
                    mask_width = video_width if need_resize else copy_width
                    mask_height = video_height if need_resize else copy_height
                    
                    if mask_width > 0 and mask_height > 0 and dst_x >= 0 and dst_y >= 0:
                        # Calculate the actual area to mark as black (video area)
                        mask_x_end = min(dst_x + mask_width, canvas_width)
                        mask_y_end = min(dst_y + mask_height, canvas_height)
                        base_mask[dst_y:mask_y_end, dst_x:mask_x_end] = 0  # Set video area to black
                    
                    # Apply feathering once if specified
                    if feather_amount > 0:
                        # Invert mask for distance transform (need white where video is)
                        inverted_mask = 255 - base_mask
                        # Create distance transform for feathering
                        dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
                        # Normalize and apply feather (inverted - closer to video = darker)
                        feather_mask = 255 - np.clip(dist_transform / feather_amount, 0, 1) * 255
                        final_mask = feather_mask.astype(np.uint8)
                    else:
                        final_mask = base_mask
                    
                    # Convert mask to tensor once
                    mask_tensor = torch.from_numpy(final_mask).float() / 255.0
                    
                    if need_resize:
                        print(f"[WAN Outpainting Editor] Video was resized from {frame_width}x{frame_height} to {video_width}x{video_height}")
                        # Recalculate copy dimensions for resized video
                        copy_width = min(video_width - src_x, canvas_width - dst_x)
                        copy_height = min(video_height - src_y, canvas_height - dst_y)
                        print(f"[WAN Outpainting Editor] Updated copy size for resized video: {copy_width}x{copy_height}")
                    
                    # Check for GPU availability
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    print(f"[WAN Outpainting Editor] Using device: {device}")
                    
                    # Process frames with parallelism for better performance
                    print(f"[WAN Outpainting Editor] Processing {len(frames)} frames...")
                    
                    # Use optimized vectorized processing for better performance
                    if len(frames) > 50:
                        # Use vectorized batch processing for large frame counts
                        print(f"[WAN Outpainting Editor] Using vectorized batch processing for {len(frames)} frames...")
                        
                        # Process in chunks to reduce memory usage
                        chunk_size = 64  # Increased chunk size for better GPU utilization
                        processed_frames = []
                        
                        for chunk_start in range(0, len(frames), chunk_size):
                            chunk_end = min(chunk_start + chunk_size, len(frames))
                            chunk_frames = frames[chunk_start:chunk_end]
                            
                            # Convert chunk to numpy array for vectorized operations
                            chunk_array = np.stack(chunk_frames)  # Shape: (N, H, W, 3)
                            
                            # Vectorized resize if needed
                            if need_resize:
                                resized_chunk = []
                                for frame in chunk_array:
                                    resized_frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_LANCZOS4)
                                    resized_chunk.append(resized_frame)
                                chunk_array = np.stack(resized_chunk)
                            
                            # Create expanded canvas for all frames at once
                            num_frames_in_chunk = chunk_array.shape[0]
                            expanded_chunk = np.full((num_frames_in_chunk, canvas_height, canvas_width, 3), 
                                                   int(frame_darkness * 255), dtype=np.uint8)
                            
                            # Vectorized copy operation for all frames
                            if copy_width > 0 and copy_height > 0:
                                expanded_chunk[:, dst_y:dst_y+copy_height, dst_x:dst_x+copy_width] = \
                                    chunk_array[:, src_y:src_y+copy_height, src_x:src_x+copy_width]
                            
                            # Convert entire chunk to tensor in one operation
                            chunk_tensor = torch.from_numpy(expanded_chunk).float() / 255.0
                            
                            # Move to GPU if available
                            if device == 'cuda':
                                chunk_tensor = chunk_tensor.cuda()
                            
                            # Add individual frames to list (keep as tensors)
                            for frame_tensor in chunk_tensor:
                                processed_frames.append(frame_tensor)
                            
                            # Progress update
                            if chunk_end % 200 == 0 or chunk_end == len(frames):
                                print(f"[WAN Outpainting Editor] Processed {chunk_end}/{len(frames)} frames...")
                        
                        # Create masks efficiently (same for all frames) - vectorized
                        if device == 'cuda':
                            mask_tensor = mask_tensor.cuda()
                        
                        # Create all masks at once
                        mask_batch = mask_tensor.unsqueeze(0).repeat(len(processed_frames), 1, 1)
                        masks = [mask_batch[i] for i in range(len(processed_frames))]
                        
                    else:
                        # Optimized processing for small frame counts
                        print(f"[WAN Outpainting Editor] Using optimized processing for {len(frames)} frames...")
                        
                        # Process all frames at once for small counts
                        frames_array = np.stack(frames)
                        
                        # Vectorized resize if needed
                        if need_resize:
                            resized_frames = []
                            for frame in frames_array:
                                resized_frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_LANCZOS4)
                                resized_frames.append(resized_frame)
                            frames_array = np.stack(resized_frames)
                        
                        # Create expanded canvas for all frames
                        expanded_frames = np.full((len(frames), canvas_height, canvas_width, 3), 
                                               int(frame_darkness * 255), dtype=np.uint8)
                        
                        # Vectorized copy operation
                        if copy_width > 0 and copy_height > 0:
                            expanded_frames[:, dst_y:dst_y+copy_height, dst_x:dst_x+copy_width] = \
                                frames_array[:, src_y:src_y+copy_height, src_x:src_x+copy_width]
                        
                        # Convert all frames to tensors at once
                        frames_tensor = torch.from_numpy(expanded_frames).float() / 255.0
                        if device == 'cuda':
                            frames_tensor = frames_tensor.cuda()
                        
                        # Split into individual frame tensors
                        for frame_tensor in frames_tensor:
                            processed_frames.append(frame_tensor)
                        
                        # Create all masks efficiently
                        if device == 'cuda':
                            mask_tensor = mask_tensor.cuda()
                        mask_batch = mask_tensor.unsqueeze(0).repeat(len(frames), 1, 1)
                        masks = [mask_batch[i] for i in range(len(frames))]
                    
                    # Stack into tensors
                    images = torch.stack(processed_frames)
                    mask_stack = torch.stack(masks)
                    
                    # Move tensors back to CPU for ComfyUI compatibility
                    if device == 'cuda':
                        images = images.cpu()
                        mask_stack = mask_stack.cpu()
                    
                    # Set final status
                    source_str = "Image sequence" if video_type == "image_sequence" else "Video file"
                    frame_word = "frame" if len(frames) == 1 else "frames"
                    perf_info = f" | {device.upper()}"
                    status = f"{source_str} outpainted ({len(frames)} {frame_word}, canvas: {canvas_width}x{canvas_height}){perf_info}"
                    
                    return (images, mask_stack, status)
                else:
                    print(f"[WAN Outpainting Editor] No frames loaded, creating empty tensor")
                    images = torch.zeros((1, canvas_height, canvas_width, 3), dtype=torch.float32)
                    masks = torch.zeros((1, canvas_height, canvas_width), dtype=torch.float32)
                    status = "No frames loaded."
                    return (images, masks, status)
                    
            except Exception as e:
                print(f"[WAN Outpainting Editor] Error processing canvas data: {e}")
                import traceback
                traceback.print_exc()
                status = f"Error: {e}"
                return (torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1)), status)
        else:
            print(f"[WAN Outpainting Editor] No canvas_data provided")
            status = "No canvas data provided."
        return (torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1)), status)


# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add the mask editor and outpainting editor first
NODE_CLASS_MAPPINGS["MaskEditor"] = MaskEditor
NODE_DISPLAY_NAME_MAPPINGS["MaskEditor"] = "Mask Editor"

NODE_CLASS_MAPPINGS["WANVaceImageToMask"] = WANVaceImageToMask
NODE_DISPLAY_NAME_MAPPINGS["WANVaceImageToMask"] = "WanVace-pipeline Image To Mask"

NODE_CLASS_MAPPINGS["WANVaceOutpaintingEditor"] = WANVaceOutpaintingEditor
NODE_DISPLAY_NAME_MAPPINGS["WANVaceOutpaintingEditor"] = "WanVace-pipeline Outpainting Editor ðŸŽ¨"

# Try to load crop and stitch nodes first (these should work independently)
print("[WAN Vace Pipeline] Loading crop and stitch nodes...")
try:
    from .wan_cropandstitch import WanCropImproved
    from .wan_cropandstitch import WanStitchImproved
    
    NODE_CLASS_MAPPINGS["WanCropImproved"] = WanCropImproved
    NODE_CLASS_MAPPINGS["WanStitchImproved"] = WanStitchImproved
    
    NODE_DISPLAY_NAME_MAPPINGS["WanCropImproved"] = "WanVace-pipeline Crop âœ‚ï¸"
    NODE_DISPLAY_NAME_MAPPINGS["WanStitchImproved"] = "WanVace-pipeline Stitch âœ‚ï¸"
    
    print("[WAN Vace Pipeline] âœ… Successfully loaded crop and stitch nodes")
except Exception as crop_e:
    print(f"[WAN Vace Pipeline] âŒ ERROR loading crop and stitch nodes: {crop_e}")
    import traceback
    traceback.print_exc()

# Try to load main nodes
try:
    from .node_mappings import NODE_CLASS_MAPPINGS as MAIN_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MAIN_DISPLAY_MAPPINGS
    print(f"[WAN Vace Pipeline] Successfully loaded {len(MAIN_NODE_MAPPINGS)} nodes from node_mappings")
    
    # Merge the main nodes with our existing mappings
    NODE_CLASS_MAPPINGS.update(MAIN_NODE_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MAIN_DISPLAY_MAPPINGS)
    
    # List all loaded nodes
    print("[WAN Vace Pipeline] All loaded nodes:")
    for node_name in NODE_CLASS_MAPPINGS:
        print(f"  - {node_name}")
        
except Exception as e:
    print(f"[WAN Vace Pipeline] ERROR loading main nodes: {e}")
    import traceback
    traceback.print_exc()
    print("[WAN Vace Pipeline] Continuing with mask editor and crop/stitch nodes only")

import os
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]







