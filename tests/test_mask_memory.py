"""Public node memory contract; only the interactive subprocess is substituted."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1] / "Lostless-Mask-Editor"
spec = importlib.util.spec_from_file_location("mask_memory_test_package", PACKAGE / "__init__.py",
                                            submodule_search_locations=[str(PACKAGE)])
pack = importlib.util.module_from_spec(spec)
pack.LOSTLESS_MINIMAL_IMPORT = True
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)


class MaskMemoryTests(unittest.TestCase):
    def setUp(self):
        pack.clear_mask_editor_memory()
        self.node = pack.MaskEditor()
        self.masks = torch.zeros((2, 16, 16))
        self.masks[0, 3:9, 3:9] = 1
        self.state = {
            "shape_keyframes": {"0": [{"vertices": [[3, 3], [8, 3], [8, 8], [3, 8]], "visible": True}], "1": []},
            "settings": {"drawing_mode": "shape", "vertex_count": 8},
            "current_frame": 1,
            "source_video": {"path": "/obsolete/source"},
        }

    def tearDown(self):
        pack.clear_mask_editor_memory()

    def test_reuse_keeps_original_points_and_uses_current_input_images(self):
        expected = copy.deepcopy(self.state["shape_keyframes"])
        self.node._store_cached_masks("node", self.masks, self.state)
        self.state["shape_keyframes"].clear()  # Cache owns its own geometry.

        def run_editor(*, args, **kwargs):
            config = json.loads(Path(args[-1]).read_text())
            project = config["project_data"]
            self.assertEqual(project["shape_keyframes"], expected)
            self.assertEqual(project["current_frame"], 1)
            self.assertNotEqual(project["source_video"]["path"], "/obsolete/source")
            self.assertTrue(Path(project["source_video"]["path"]).is_dir())
            output = Path(config["output_dir"])
            np.save(output / "edited_masks.npy", (self.masks.numpy() * 255).astype(np.uint8))
            (output / "project_data.json").write_text(json.dumps(project))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp, patch("tempfile.mkdtemp", return_value=tmp), patch("subprocess.run", side_effect=run_editor):
            result = self.node.edit_mask(torch.zeros((2, 16, 16, 3)), reuse_last_edit=True, unique_id="node")
        torch.testing.assert_close(result["result"][0], self.masks)
        self.assertEqual(pack.get_mask_editor_memory_cache()["node"]["editor_state"]["shape_keyframes"], expected)

    def test_clear_memory_and_shape_mismatch_do_not_reuse_stale_points(self):
        self.node._store_cached_masks("node", self.masks, self.state)
        cached, status = self.node._get_cached_masks("node", (3, 16, 16))
        self.assertIsNone(cached)
        self.assertIn("stale", status)
        self.assertTrue(pack.clear_mask_editor_memory("node"))
        self.assertNotIn("node", pack.get_mask_editor_memory_cache())

    def test_new_pixel_result_replaces_previous_vector_state(self):
        self.node._store_cached_masks("node", self.masks, self.state)
        self.node._store_cached_masks("node", self.masks)
        self.assertEqual(pack.get_mask_editor_memory_cache()["node"]["editor_state"], {})


if __name__ == "__main__":
    unittest.main()
