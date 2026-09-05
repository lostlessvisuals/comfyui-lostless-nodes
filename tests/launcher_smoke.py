"""Run the real launcher in a disposable process and accept its test session."""
import os
import json
import base64
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Lostless-Mask-Editor" / "nodes"))
import comfyui_mask_launcher as launcher
import cv2
import numpy as np
from PyQt5.QtCore import QSettings
from PyQt5.QtTest import QTest

config_path = Path(sys.argv[1])
expected_mode = sys.argv[2] if len(sys.argv) > 2 else "shape"
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(config_path.parent / "settings"))
QSettings('MaskEditor', 'VideoFrameProcessor').setValue('mask_editor_mode', 'brush')


def check_and_accept(editor):
    QTest.qWait(900)  # Include reinitialization, startup, and timeline timers.
    assert editor.drawing_mode == expected_mode, editor.drawing_mode
    assert editor.mask_widget.drawing_mode == expected_mode, editor.mask_widget.drawing_mode
    assert editor.current_frame_index == 1, editor.current_frame_index
    assert editor.timeline_widget.current_frame == 1
    assert int(editor.mask_widget.video_frame[0, 0, 0]) == 80
    if expected_mode == "shape":
        assert editor.mask_widget.get_shapes_for_frame(1), "Missing restored outline points"
    else:
        assert not editor.mask_widget.shape_keyframes, "Raster masks must not be silently converted"
        assert editor.create_points_btn.isEnabled()
        editor.auto_save_session()
        saved = json.loads((Path(editor.output_dir) / "working_autosave.json").read_text())
        saved_mask = cv2.imdecode(np.frombuffer(base64.b64decode(saved["mask_frames"]["1"]), np.uint8), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(saved_mask, editor.mask_frames[1])
    editor.grab().save(str(config_path.parent / "launcher.png"))
    editor.auto_save_timer.stop()
    editor.session_timer.stop()
    editor.playback_timer.stop()
    return editor.Accepted


launcher.EnhancedMaskEditor.exec_ = check_and_accept
sys.argv = [sys.argv[0], "--config", str(config_path)]
launcher.main()
