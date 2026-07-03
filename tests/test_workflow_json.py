"""Regression guard for the bundled ComfyUI workflow JSON files.

Every tracked workflow under the repo's workflow directories must:
  (a) parse as JSON,
  (b) use only node types this pack defines plus a small comfy-core whitelist,
  (c) (UI-format only) give every Anima node a ``widgets_values`` array whose
      length matches the widget count derived from that node's INPUT_TYPES,
  (d) (UI-format only) reference only existing node ids from its ``links``.

API-format workflows (keyed by numeric id with ``class_type`` + ``inputs``)
only get the class-type check, since they carry no ``widgets_values``/``links``.
"""

import json
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from anima_mixer import NODE_CLASS_MAPPINGS  # noqa: E402

# comfy-core node types used by the bundled workflows. Kept explicit so a stray
# third-party node (which a user could not run without extra installs) fails the
# guard instead of shipping silently.
CORE_NODE_TYPES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "Note",
}

# Input entry types that render as an editable widget (and therefore consume a
# slot in widgets_values). Anything else - MODEL/CLIP/CONDITIONING/ANIMA_* and
# the "*" wildcard - is a socket and carries no widget value.
PRIMITIVE_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def _workflow_files():
    root = Path(REPO_ROOT)
    files = []
    sample = root / "sample workflow.json"
    if sample.exists():
        files.append(sample)
    for sub in ("workflow", "workflow/node_usage_showcase", "workflow/pr4_self_test_api"):
        files.extend(sorted((root / sub).glob("*.json")))
    return files


def _widget_count(input_types):
    """Number of widgets_values slots implied by an INPUT_TYPES dict."""
    count = 0
    for section in ("required", "optional"):
        for spec in (input_types.get(section) or {}).values():
            type_def = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            config = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else {}
            if isinstance(type_def, (list, tuple)):
                count += 1  # combo (list of choices) is a widget
            elif isinstance(type_def, str) and type_def in PRIMITIVE_WIDGET_TYPES:
                if not (isinstance(config, dict) and config.get("forceInput")):
                    count += 1
    return count


ANIMA_WIDGET_COUNTS = {
    name: _widget_count(cls.INPUT_TYPES()) for name, cls in NODE_CLASS_MAPPINGS.items()
}


def _iter_ui_nodes(data):
    for node in data["nodes"]:
        yield node


def _is_ui_format(data):
    return isinstance(data, dict) and isinstance(data.get("nodes"), list)


class BundledWorkflowJsonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = _workflow_files()
        cls.loaded = {}
        for path in cls.files:
            with open(path, encoding="utf-8") as fh:
                cls.loaded[path] = json.load(fh)

    def test_files_found(self):
        # Guards against the glob silently matching nothing.
        self.assertTrue(self.files, "no bundled workflow JSON files were found")

    def test_json_parses(self):
        # setUpClass already parsed every file; this makes the parse an explicit
        # assertion and reports the count.
        self.assertEqual(len(self.loaded), len(self.files))

    def test_node_types_are_known(self):
        for path, data in self.loaded.items():
            if _is_ui_format(data):
                nodes = [(n.get("id"), n.get("type")) for n in _iter_ui_nodes(data)]
            else:
                nodes = [
                    (key, node.get("class_type"))
                    for key, node in data.items()
                    if isinstance(node, dict) and "class_type" in node
                ]
            for node_id, node_type in nodes:
                with self.subTest(file=path.name, node=node_id, type=node_type):
                    self.assertTrue(
                        node_type in NODE_CLASS_MAPPINGS or node_type in CORE_NODE_TYPES,
                        f"{path.name}: node {node_id} has unknown/third-party "
                        f"type {node_type!r}",
                    )

    def test_ui_widget_counts(self):
        for path, data in self.loaded.items():
            if not _is_ui_format(data):
                continue
            for node in _iter_ui_nodes(data):
                node_type = node.get("type")
                if node_type not in ANIMA_WIDGET_COUNTS:
                    continue  # only Anima nodes have a known widget contract
                # Widgets promoted to input sockets appear in "inputs" with a
                # "widget" key and drop out of widgets_values.
                converted = sum(
                    1 for inp in (node.get("inputs") or [])
                    if isinstance(inp, dict) and "widget" in inp
                )
                expected = ANIMA_WIDGET_COUNTS[node_type] - converted
                actual = len(node.get("widgets_values") or [])
                with self.subTest(file=path.name, node=node.get("id"), type=node_type):
                    self.assertEqual(
                        actual, expected,
                        f"{path.name}: {node_type} (id {node.get('id')}) has "
                        f"{actual} widget values but INPUT_TYPES implies {expected}",
                    )

    def test_ui_links_reference_existing_nodes(self):
        for path, data in self.loaded.items():
            if not _is_ui_format(data):
                continue
            node_ids = {n.get("id") for n in _iter_ui_nodes(data)}
            for link in data.get("links") or []:
                if not isinstance(link, (list, tuple)) or len(link) < 5:
                    continue
                link_id, src, _src_slot, dst, _dst_slot = link[:5]
                with self.subTest(file=path.name, link=link_id):
                    self.assertIn(
                        src, node_ids,
                        f"{path.name}: link {link_id} source node {src} does not exist",
                    )
                    self.assertIn(
                        dst, node_ids,
                        f"{path.name}: link {link_id} target node {dst} does not exist",
                    )


if __name__ == "__main__":
    unittest.main()
