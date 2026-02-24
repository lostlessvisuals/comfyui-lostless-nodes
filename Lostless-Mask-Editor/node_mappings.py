"""
Node registration for Lostless nodes
This file imports nodes from separate modules and registers them with ComfyUI
"""

import os
print(f"[Lostless Nodes] Loading node_mappings.py from: {os.path.abspath(__file__)}")

# Configuration flag to control which nodes are enabled
# Set to True to enable experimental/development nodes
ENABLE_EXPERIMENTAL_NODES = False

# Import all nodes from the nodes package
from .nodes import (
    WANSaveVideo,
    WANLoadVideo,
    WANVaceSplitReferenceVideo,
    WANVaceJoinVideos,
    WANVaceVideoExtension,
    WANVaceFrameInterpolation,
    WANVaceKeyframeTimeline,
    WANVaceFrameSampler,
    WANVaceFrameInjector,
    WANVaceOutpainting,
    WANVaceBatchStartIndex,
    WANFastImageBatchProcessor,
    WANFastImageCompositeMasked,
    WANFastImageBlend,
    WANFastImageScaleBy,
    WANFastImageScaleToMegapixels,
    WANFastImageResize,
    WANFastDepthAnythingV2,
    WANFastDWPose,
    WANFastVideoEncode,
    WANFastVACEEncode,
    WANFastVideoCombine
)

# Import VACE Loop Encoder node
try:
    from .nodes.vace_loop_encoder import WANVACELoopEncoder
    print("Successfully imported WANVACELoopEncoder node")
except ImportError as e:
    print(f"WARNING: Failed to import WANVACELoopEncoder node: {e}")
    WANVACELoopEncoder = None

# Try to import mask viewer node separately
try:
    from .nodes.mask_viewer import WANVaceMaskViewer
    print("Successfully imported mask viewer node")
except ImportError as e:
    print(f"WARNING: Failed to import mask viewer node: {e}")
    WANVaceMaskViewer = None

# Try test node
try:
    from .nodes.test_mask_node import WANVaceTestMask
    print("Successfully imported test mask node")
except ImportError as e:
    print(f"WARNING: Failed to import test mask node: {e}")
    WANVaceTestMask = None

# Import WAN Inpaint Conditioning node
try:
    from .wan_inpaint_conditioning import WANInpaintConditioning
    print("Successfully imported WANInpaintConditioning node")
except ImportError as e:
    print(f"WARNING: Failed to import WANInpaintConditioning node: {e}")
    WANInpaintConditioning = None

# Import WAN Video Sampler Inpaint node
try:
    from .wan_video_sampler_inpaint import WANVideoSamplerInpaint
    print("Successfully imported WANVideoSamplerInpaint node")
except ImportError as e:
    print(f"WARNING: Failed to import WANVideoSamplerInpaint node: {e}")
    WANVideoSamplerInpaint = None

# Import WAN Tiled Sampler node
try:
    from .nodes.wan_tiled_sampler import WANTiledSampler
    print("Successfully imported WANTiledSampler node")
except ImportError as e:
    print(f"WARNING: Failed to import WANTiledSampler node: {e}")
    WANTiledSampler = None

# Import WAN Match Batch Size node
try:
    from .nodes.wan_match_batch_size import WANMatchBatchSize
    print("Successfully imported WANMatchBatchSize node")
except ImportError as e:
    print(f"WARNING: Failed to import WANMatchBatchSize node: {e}")
    WANMatchBatchSize = None

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # I/O Nodes
    "WANLoadVideo": WANLoadVideo,
    "WANSaveVideo": WANSaveVideo,
    
    # Processing Nodes
    # "WANVaceSplitReferenceVideo": WANVaceSplitReferenceVideo,  # Disabled for release
    "WANVaceJoinVideos": WANVaceJoinVideos,
    "WANVaceVideoExtension": WANVaceVideoExtension,
    "WANVaceFrameInterpolation": WANVaceFrameInterpolation,
    
    # Timeline Nodes
    "WANVaceKeyframeTimeline": WANVaceKeyframeTimeline,
    
    # Frame Utility Nodes
    # "WANVaceFrameSampler": WANVaceFrameSampler,  # Disabled for release
    "WANVaceFrameInjector": WANVaceFrameInjector,
    
    # Effects Nodes
    "WANVaceOutpainting": WANVaceOutpainting,
    # "WANVaceBatchStartIndex": WANVaceBatchStartIndex,  # Disabled for release
    
    # Fast Processing Nodes
    "WANFastImageBatchProcessor": WANFastImageBatchProcessor,
    # "WANFastImageCompositeMasked": WANFastImageCompositeMasked,  # Disabled for release
    # "WANFastImageBlend": WANFastImageBlend,  # Disabled for release
    # "WANFastImageScaleBy": WANFastImageScaleBy,  # Disabled for release
    # "WANFastImageScaleToMegapixels": WANFastImageScaleToMegapixels,  # Disabled for release
    # "WANFastImageResize": WANFastImageResize,  # Disabled for release
    
    # Fast ControlNet Processors
    "WANFastDepthAnythingV2": WANFastDepthAnythingV2,
    "WANFastDWPose": WANFastDWPose,
    
    # Fast Video Processors
    # "WANFastVideoEncode": WANFastVideoEncode,  # Disabled for release
    # "WANFastVACEEncode": WANFastVACEEncode,  # Disabled for release
    # "WANFastVideoCombine": WANFastVideoCombine  # Disabled for release
}

# Add disabled nodes if experimental mode is enabled
if ENABLE_EXPERIMENTAL_NODES:
    NODE_CLASS_MAPPINGS.update({
        # Processing Nodes
        "WANVaceSplitReferenceVideo": WANVaceSplitReferenceVideo,
        
        # Frame Utility Nodes
        "WANVaceFrameSampler": WANVaceFrameSampler,
        
        # Effects Nodes
        "WANVaceBatchStartIndex": WANVaceBatchStartIndex,
        
        # Fast Processing Nodes
        "WANFastImageCompositeMasked": WANFastImageCompositeMasked,
        "WANFastImageBlend": WANFastImageBlend,
        "WANFastImageScaleBy": WANFastImageScaleBy,
        "WANFastImageScaleToMegapixels": WANFastImageScaleToMegapixels,
        "WANFastImageResize": WANFastImageResize,
        
        # Fast Video Processors
        "WANFastVideoEncode": WANFastVideoEncode,
        "WANFastVACEEncode": WANFastVACEEncode,
        "WANFastVideoCombine": WANFastVideoCombine
    })

# Add VACE Loop Encoder node if successfully imported
if WANVACELoopEncoder is not None and ENABLE_EXPERIMENTAL_NODES:
    NODE_CLASS_MAPPINGS["WANVACELoopEncoder"] = WANVACELoopEncoder

# Add mask nodes if they were successfully imported
# Legacy mask editor mapping removed (MaskEditor is defined in __init__.py)
if WANVaceMaskViewer is not None:
    NODE_CLASS_MAPPINGS["WANVaceMaskViewer"] = WANVaceMaskViewer
if WANVaceTestMask is not None:
    NODE_CLASS_MAPPINGS["WANVaceTestMask"] = WANVaceTestMask

# Add WAN Inpaint Conditioning node if successfully imported
if WANInpaintConditioning is not None:
    NODE_CLASS_MAPPINGS["WANInpaintConditioning"] = WANInpaintConditioning

# Add WAN Video Sampler Inpaint node if successfully imported
if WANVideoSamplerInpaint is not None:
    NODE_CLASS_MAPPINGS["WANVideoSamplerInpaint"] = WANVideoSamplerInpaint

# Add WAN Tiled Sampler node if successfully imported
if WANTiledSampler is not None:
    NODE_CLASS_MAPPINGS["WANTiledSampler"] = WANTiledSampler

# Add WAN Match Batch Size node if successfully imported
if WANMatchBatchSize is not None:
    NODE_CLASS_MAPPINGS["WANMatchBatchSize"] = WANMatchBatchSize

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    # I/O Nodes
    "WANLoadVideo": "Lostless Load Video ðŸŽ¬",
    "WANSaveVideo": "Lostless Save Video ðŸ’¾",
    
    # Processing Nodes
    # "WANVaceSplitReferenceVideo": "Lostless Split Video Batch âœ‚ï¸",  # Disabled for release
    "WANVaceJoinVideos": "Lostless Join Videos ðŸ”—",
    "WANVaceVideoExtension": "Lostless Video Extension ðŸ”„",
    "WANVaceFrameInterpolation": "Lostless Frame Interpolator ðŸŽžï¸",
    
    # Timeline Nodes
    "WANVaceKeyframeTimeline": "Lostless Keyframe Timeline ðŸ“½ï¸",
    
    # Frame Utility Nodes
    # "WANVaceFrameSampler": "Lostless Frame Sampler ðŸ“Š",  # Disabled for release
    "WANVaceFrameInjector": "Lostless Frame Injector ðŸ’‰",
    
    # Effects Nodes
    "WANVaceOutpainting": "Lostless Outpainting Prep ðŸ–¼ï¸",
    # "WANVaceBatchStartIndex": "Lostless Batch Start Index ðŸ”¢",  # Disabled for release
    
    # Fast Processing Nodes
    "WANFastImageBatchProcessor": "Lostless Fast Image Batch Processor ðŸš€",
    # "WANFastImageCompositeMasked": "Lostless Fast Image Composite Masked ðŸš€",  # Disabled for release
    # "WANFastImageBlend": "Lostless Fast Image Blend ðŸš€",  # Disabled for release
    # "WANFastImageScaleBy": "Lostless Fast Image Scale By ðŸš€",  # Disabled for release
    # "WANFastImageScaleToMegapixels": "Lostless Fast Image Scale To Megapixels ðŸš€",  # Disabled for release
    # "WANFastImageResize": "Lostless Fast Image Resize ðŸš€",  # Disabled for release
    
    # Fast ControlNet Processors
    "WANFastDepthAnythingV2": "Lostless Fast Depth Anything V2 ðŸš€",
    "WANFastDWPose": "Lostless Fast DWPose Estimator ðŸš€",
    
    # Fast Video Processors
    # "WANFastVideoEncode": "Lostless Fast Video Encode ðŸš€",  # Disabled for release
    # "WANFastVACEEncode": "Lostless Fast VACE Encode ðŸš€",  # Disabled for release
    # "WANFastVideoCombine": "Lostless Fast Video Combine ðŸš€"  # Disabled for release
}

# Add disabled node display names if experimental mode is enabled
if ENABLE_EXPERIMENTAL_NODES:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        # Processing Nodes
        "WANVaceSplitReferenceVideo": "Lostless Split Video Batch âœ‚ï¸",
        
        # Frame Utility Nodes
        "WANVaceFrameSampler": "Lostless Frame Sampler ðŸ“Š",
        
        # Effects Nodes
        "WANVaceBatchStartIndex": "Lostless Batch Start Index ðŸ”¢",
        
        # Fast Processing Nodes
        "WANFastImageCompositeMasked": "Lostless Fast Image Composite Masked ðŸš€",
        "WANFastImageBlend": "Lostless Fast Image Blend ðŸš€",
        "WANFastImageScaleBy": "Lostless Fast Image Scale By ðŸš€",
        "WANFastImageScaleToMegapixels": "Lostless Fast Image Scale To Megapixels ðŸš€",
        "WANFastImageResize": "Lostless Fast Image Resize ðŸš€",
        
        # Fast Video Processors
        "WANFastVideoEncode": "Lostless Fast Video Encode ðŸš€",
        "WANFastVACEEncode": "Lostless Fast VACE Encode ðŸš€",
        "WANFastVideoCombine": "Lostless Fast Video Combine ðŸš€"
    })

# Add VACE Loop Encoder display name if node was successfully imported
if WANVACELoopEncoder is not None and ENABLE_EXPERIMENTAL_NODES:
    NODE_DISPLAY_NAME_MAPPINGS["WANVACELoopEncoder"] = "Lostless VACE Loop Encoder ðŸ”"

# Add mask node display names if they were successfully imported
# Legacy mask editor display mapping removed (MaskEditor is defined in __init__.py)
if WANVaceMaskViewer is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANVaceMaskViewer"] = "Lostless Mask Viewer ðŸ‘ï¸"
if WANVaceTestMask is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANVaceTestMask"] = "Lostless Test Mask ðŸ§ª"

# Add WAN Inpaint Conditioning display name if node was successfully imported
if WANInpaintConditioning is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANInpaintConditioning"] = "Lostless WAN Inpaint Conditioning ðŸŽ¨"

# Add WAN Video Sampler Inpaint display name if node was successfully imported
if WANVideoSamplerInpaint is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANVideoSamplerInpaint"] = "Lostless WAN Video Sampler Inpaint ðŸŽ­"

# Add WAN Tiled Sampler display name if node was successfully imported
if WANTiledSampler is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANTiledSampler"] = "Lostless WAN Tiled Sampler ðŸ”²"

# Add WAN Match Batch Size display name if node was successfully imported
if WANMatchBatchSize is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANMatchBatchSize"] = "Lostless Match Batch Size ðŸ”„"

