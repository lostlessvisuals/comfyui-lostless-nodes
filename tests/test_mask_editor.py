"""Real Qt widget regressions. Run: python -m unittest discover -s tests -v."""
import contextlib
import base64
import copy
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Lostless-Mask-Editor" / "nodes"))

import numpy as np
import cv2
from PyQt5.QtCore import QPoint, QSettings, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMessageBox
from mask_editor import InpaintingMaskEditor, MaskDrawingWidget


def square(x=20):
    return {"vertices": [[x, 20], [x + 40, 20], [x + 40, 60], [x, 60]],
            "visible": True, "closed": True}


class MaskEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings_dir = tempfile.TemporaryDirectory()
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, cls.settings_dir.name)
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls.settings_dir.cleanup()

    def setUp(self):
        QSettings('MaskEditor', 'VideoFrameProcessor').clear()
        self.widgets = []

    def tearDown(self):
        for widget in self.widgets:
            if isinstance(widget, InpaintingMaskEditor):
                widget.auto_save_timer.stop()
                widget.playback_timer.stop()
            widget.hide()
            widget.deleteLater()
        self.app.processEvents()

    def drawing_widget(self):
        widget = MaskDrawingWidget()
        widget.resize(400, 400)
        widget.set_mask(np.zeros((100, 100), np.uint8), np.zeros((100, 100, 3), np.uint8))
        widget.drawing_mode = "shape"
        widget.shape_keyframes = {0: [square()]}
        widget.invalidate_shape_cache()
        self.widgets.append(widget)
        widget.grab()  # Establish the real paint transform.
        return widget

    def editor(self, initial_mode=None, frame_count=3):
        editor = InpaintingMaskEditor([np.zeros((100, 100, 3), np.uint8)] * frame_count,
                                      initial_mode=initial_mode)
        self.widgets.append(editor)
        QTest.qWait(250)  # Exercise the actual delayed startup callback.
        return editor

    def test_debug_view_still_renders_cyan_handles(self):
        widget = self.drawing_widget()
        widget.resize(1000, 1000)
        widget.shape_keyframes = {0: [square(0)]}
        widget.invalidate_shape_cache()
        widget.show_shape_debug = True
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            image = widget.grab().toImage()
        self.assertNotIn("Error in paintEvent", output.getvalue())
        cyan_pixels = sum(image.pixelColor(x, y).name() == "#00ffff"
                          for x in range(image.width()) for y in range(image.height()))
        self.assertGreater(cyan_pixels, 0, "Shape handles must remain visible in debug view")

    def test_new_editor_defaults_to_shape_but_remembers_pixel_preference(self):
        editor = self.editor()
        self.assertEqual(editor.drawing_mode, "shape")
        editor.set_drawing_mode("brush")
        reopened = self.editor()
        self.assertEqual(reopened.drawing_mode, "brush")

    def test_explicit_pixel_mode_overrides_last_shape_brush(self):
        editor = self.editor("shape")
        editor._apply_initial_mode("brush")
        self.assertEqual(editor.drawing_mode, "brush")
        self.assertEqual(editor.mask_widget.drawing_mode, "brush")

    def test_restored_shape_mode_survives_pending_startup_timer(self):
        editor = InpaintingMaskEditor([np.zeros((100, 100, 3), np.uint8)], initial_mode="brush")
        self.widgets.append(editor)
        editor.restore_drawing_mode("shape")
        QTest.qWait(250)
        self.assertEqual(editor.drawing_mode, "shape")
        self.assertEqual(editor.mask_widget.drawing_mode, "shape")
        self.assertTrue(editor.brush_btn.isChecked())

    def test_drag_updates_cached_shape_and_undo_restores_before_drag(self):
        widget = self.drawing_widget()
        before = widget.get_shapes_for_frame(0)
        rect = widget.display_rect
        start = QPoint(rect.x() + int(rect.width() * .2), rect.y() + int(rect.height() * .2))
        end = start + QPoint(20, 20)
        widget.check_vertex_selection(start)
        widget.warp_vertex(end)
        after = widget.get_shapes_for_frame(0)
        self.assertNotEqual(before, after, "Dragging must invalidate the rendered shape cache")
        release = QMouseEvent(QMouseEvent.MouseButtonRelease, end, Qt.LeftButton,
                              Qt.NoButton, Qt.NoModifier)
        widget.mouseReleaseEvent(release)
        self.assertTrue(widget.undo())
        self.assertEqual(widget.get_shapes_for_frame(0), before)
        self.assertTrue(widget.redo())
        self.assertEqual(widget.get_shapes_for_frame(0), after)

    def test_click_without_drag_does_not_add_undo(self):
        widget = self.drawing_widget()
        rect = widget.display_rect
        point = QPoint(rect.x() + int(rect.width() * .2), rect.y() + int(rect.height() * .2))
        widget.check_vertex_selection(point)
        widget.mouseReleaseEvent(QMouseEvent(QMouseEvent.MouseButtonRelease, point,
                                            Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
        self.assertEqual(len(widget.undo_stack), 0)

    def test_zoomed_out_selection_uses_screen_distance_and_ignores_hidden_shapes(self):
        widget = self.drawing_widget()
        widget.zoom_level = .25
        widget._user_modified_view = True
        widget._auto_fit_pending = False
        widget.grab()
        rect = widget.display_rect
        point = QPoint(rect.x() + 5, rect.y() + 5)
        widget.check_vertex_selection(point + QPoint(-5, 0))
        self.assertEqual(widget.selected_vertex_index, 0, "A handle five screen pixels away is clickable")
        widget.shape_keyframes[0][0]['visible'] = False
        widget.invalidate_shape_cache()
        widget.check_vertex_selection(point)
        self.assertIsNone(widget.selected_vertex_index, "Hidden shapes must not intercept clicks")

    def test_drag_on_interpolated_frame_creates_undoable_keyframe(self):
        editor = self.editor("shape")
        widget = editor.mask_widget
        widget.shape_keyframes = {0: [square()], 2: [square(40)]}
        widget.invalidate_shape_cache()
        editor.on_frame_changed(1)
        widget.grab()
        endpoints = copy.deepcopy(widget.shape_keyframes)
        rect = widget.display_rect
        start = QPoint(rect.x() + int(rect.width() * .3), rect.y() + int(rect.height() * .2))
        widget.check_vertex_selection(start)
        widget.warp_vertex(start + QPoint(20, 20))
        self.assertIn(1, widget.shape_keyframes)
        self.assertEqual(widget.shape_keyframes[0], endpoints[0])
        self.assertEqual(widget.shape_keyframes[2], endpoints[2])
        self.assertTrue(widget.undo())
        self.assertNotIn(1, widget.shape_keyframes)

    def test_shape_handle_can_be_dragged_with_normal_mouse_gesture(self):
        widget = self.drawing_widget()
        widget.reset_view_to_default()
        widget.grab()
        start = QPoint(80, 80)
        end = QPoint(100, 100)
        QTest.mousePress(widget, Qt.LeftButton, pos=start)
        move = QMouseEvent(QMouseEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        widget.mouseMoveEvent(move)
        QTest.mouseRelease(widget, Qt.LeftButton, pos=end)
        self.assertEqual(widget.get_shapes_for_frame(0)[0]['vertices'][0], [25, 25])
        self.assertTrue(widget.undo())
        self.assertEqual(widget.get_shapes_for_frame(0)[0]['vertices'][0], [20, 20])

    def test_raster_recovery_preserves_blank_frames_on_export(self):
        editor = self.editor("shape")
        editor.mask_frames[1][20:60, 20:60] = 255
        self.assertEqual(editor.bootstrap_shape_keyframes_from_masks(), 1)
        masks = editor.get_masks()
        self.assertFalse(np.any(masks[0]), "Blank first frame must not inherit the next mask")
        self.assertTrue(np.any(masks[1]))
        self.assertFalse(np.any(masks[2]), "Blank last frame must not inherit the previous mask")

    def test_saved_vector_timeline_is_not_rebuilt_from_raster_export(self):
        editor = self.editor("shape")
        editor.mask_widget.shape_keyframes = {0: [square()], 2: []}
        before = copy.deepcopy(editor.mask_widget.shape_keyframes)
        editor.mask_frames[1][20:60, 20:60] = 255
        self.assertEqual(editor.bootstrap_shape_keyframes_from_masks(), 0)
        self.assertEqual(editor.mask_widget.shape_keyframes, before)

    def test_launcher_restores_legacy_mode_frame_and_exports_masks(self):
        self._check_launcher_roundtrip()

    def test_launcher_preserves_grayscale_holes_and_tiny_raster_components(self):
        self._check_launcher_roundtrip(raster_only=True)

    def _check_launcher_roundtrip(self, raster_only=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            encoded_masks = {}
            original_masks = []
            for i in range(3):
                cv2.imwrite(str(frames / f"frame_{i:04d}.png"), np.full((100, 100, 3), i * 80, np.uint8))
                mask = np.zeros((100, 100), np.uint8)
                if i == 1:
                    mask[20:60, 20:60] = 128 if raster_only else 255
                    if raster_only:
                        mask[30:50, 30:50] = 0
                        mask[80, 80] = 64
                original_masks.append(mask)
                ok, encoded = cv2.imencode(".png", mask)
                self.assertTrue(ok)
                encoded_masks[str(i)] = base64.b64encode(encoded).decode("ascii")
            config = {
                "output_dir": str(root / "output"), "comfy_strict_mode": True,
                "project_data": {
                    "video_info": {"path": str(frames), "type": "image_sequence"},
                    "shape_keyframes": {} if raster_only else {"0": [], "1": [square()], "2": []},
                    "mask_frames": encoded_masks, "drawing_mode": "shape", "current_frame": 1,
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("launcher_smoke.py")),
                                     str(config_path), "brush" if raster_only else "shape"], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout[-6000:] + result.stderr[-3000:])
            masks = np.load(root / "output" / "edited_masks.npy")
            self.assertEqual(masks.shape, (3, 100, 100))
            self.assertFalse(np.any(masks[0]))
            self.assertTrue(np.any(masks[1]))
            self.assertFalse(np.any(masks[2]))
            project = json.loads((root / "output" / "project_data.json").read_text())
            self.assertEqual(project["settings"]["drawing_mode"], "brush" if raster_only else "shape")
            self.assertEqual(project["current_frame"], 1)
            if raster_only:
                np.testing.assert_array_equal(masks, np.stack(original_masks))

    def raster_editor(self):
        editor = self.editor("brush")
        editor.mask_frames[1][20:60, 20:60] = 128
        editor.mask_frames[1][30:50, 30:50] = 0
        editor.mask_frames[1][80, 80] = 64
        editor.on_frame_changed(1)
        return editor

    def test_create_points_is_undoable_without_losing_original_pixels(self):
        editor = self.raster_editor()
        original = np.stack(editor.mask_frames).copy()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            editor.create_points_btn.click()
        self.assertEqual(editor.drawing_mode, "shape")
        self.assertTrue(editor.mask_widget.get_shapes_for_frame(1))
        converted = np.stack(editor.get_masks()).copy()
        self.assertFalse(np.array_equal(converted, original))
        self.assertTrue(editor.mask_widget.undo())
        self.assertEqual(editor.drawing_mode, "brush")
        np.testing.assert_array_equal(np.stack(editor.get_masks()), original)
        self.assertTrue(editor.mask_widget.redo())
        self.assertEqual(editor.drawing_mode, "shape")
        np.testing.assert_array_equal(np.stack(editor.get_masks()), converted)

    def test_declining_point_conversion_leaves_masks_and_history_unchanged(self):
        editor = self.raster_editor()
        original = np.stack(editor.mask_frames).copy()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            self.assertFalse(editor.create_points_from_masks())
        self.assertFalse(editor.mask_widget.undo_stack)
        self.assertFalse(editor.mask_widget.shape_keyframes)
        np.testing.assert_array_equal(np.stack(editor.get_masks()), original)

    def test_switching_to_shape_on_raster_masks_requires_explicit_conversion(self):
        editor = self.raster_editor()
        original = np.stack(editor.mask_frames).copy()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            editor.toggle_brush_mode()
        self.assertEqual(editor.drawing_mode, "brush")
        np.testing.assert_array_equal(np.stack(editor.get_masks()), original)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            editor.toggle_brush_mode()
        self.assertEqual(editor.drawing_mode, "shape")
        self.assertTrue(editor.mask_widget.shape_keyframes)

    def test_tiny_mask_conversion_fails_without_destroying_pixels(self):
        editor = self.editor("brush")
        editor.mask_frames[0][50, 50] = 128
        original = np.stack(editor.mask_frames).copy()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), patch.object(QMessageBox, "information"):
            self.assertFalse(editor.create_points_from_masks())
        self.assertFalse(editor.mask_widget.undo_stack)
        self.assertEqual(editor.drawing_mode, "brush")
        np.testing.assert_array_equal(np.stack(editor.get_masks()), original)

    def test_deferred_carry_end_paths_apply_once_and_stay_stable_on_idle(self):
        for boundary in ("key_release", "click_jump", "mode_switch", "tool_switch", "direction_reversal"):
            with self.subTest(boundary=boundary):
                editor = self.editor("brush", frame_count=5)
                widget = editor.mask_widget
                widget.grab()
                start = widget.display_rect.center()
                QTest.mousePress(widget, Qt.LeftButton, pos=start)
                delta = widget.mask.copy()
                self.assertTrue(np.any(delta))
                QTest.keyPress(editor, Qt.Key_Right)
                QTest.keyPress(editor, Qt.Key_Right)
                if boundary == "click_jump":
                    editor.on_timeline_frame_changed(4)
                elif boundary == "mode_switch":
                    editor.set_drawing_mode("eraser")
                elif boundary == "tool_switch":
                    editor.set_tool("zoom")
                elif boundary == "direction_reversal":
                    QTest.keyPress(editor, Qt.Key_Left)
                    QTest.keyRelease(editor, Qt.Key_Left)
                QTest.keyRelease(editor, Qt.Key_Right)
                QTest.mouseRelease(widget, Qt.LeftButton, pos=start)
                self.assertFalse(editor._recent_brush_navigation_pending_frames)
                np.testing.assert_array_equal(editor.mask_frames[1], delta)
                np.testing.assert_array_equal(editor.mask_frames[2], delta)
                self.assertFalse(np.any(editor.mask_frames[3]))
                self.assertFalse(np.any(editor.mask_frames[4]))
                snapshot = np.stack(editor.mask_frames).copy()
                undo_count = len(editor.mask_widget.undo_stack)
                QTest.keyRelease(editor, Qt.Key_Right)
                QTest.qWait(50)
                np.testing.assert_array_equal(np.stack(editor.mask_frames), snapshot)
                self.assertEqual(len(editor.mask_widget.undo_stack), undo_count)

    def test_shape_brush_stroke_creates_editable_points(self):
        widget = self.drawing_widget()
        widget.shape_keyframes.clear()
        widget.invalidate_shape_cache()
        QTest.mousePress(widget, Qt.LeftButton, pos=QPoint(180, 180))
        QTest.mouseRelease(widget, Qt.LeftButton, pos=QPoint(180, 180))
        shapes = widget.get_shapes_for_frame(0)
        self.assertTrue(shapes)
        self.assertGreaterEqual(len(shapes[0]['vertices']), 3)
        self.assertTrue(np.any(widget.mask))

    def test_changing_frame_during_point_drag_does_not_edit_the_new_frame(self):
        editor = self.editor("shape")
        widget = editor.mask_widget
        widget.shape_keyframes = {0: [square()], 1: [square(40)]}
        widget.invalidate_shape_cache()
        widget.grab()
        rect = widget.display_rect
        start = QPoint(rect.x() + int(rect.width() * .2), rect.y() + int(rect.height() * .2))
        widget.check_vertex_selection(start)
        before = copy.deepcopy(widget.shape_keyframes[1])
        editor.on_frame_changed(1)
        widget.warp_vertex(start + QPoint(20, 20))
        self.assertEqual(widget.shape_keyframes[1], before)
        self.assertIsNone(widget.selected_vertex_index)


if __name__ == "__main__":
    unittest.main()
