from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest

from debug_utils import inspect_link_api, save_memory_to_txt
from plotting_utils import simple_plot
from tensor_utils import freeze_pyg_data
from rp import graph as graph_package
from rp.graph import snapshots
from rp import reporting


class FakeTensor:
    def __init__(self, value, events=None):
        self.value = value
        self.events = list(events or [])

    def detach(self):
        self.events.append("detach")
        return self

    def clone(self):
        return FakeTensor(self.value, self.events + ["clone"])

    def cpu(self):
        self.events.append("cpu")
        return self


class FakeTorch:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)


class FakeData:
    def __init__(self, values):
        self.values = dict(values)

    def clone(self):
        return FakeData(self.values)

    def keys(self):
        return self.values.keys()

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value


class TensorUtilityCharacterizationTests(unittest.TestCase):
    def test_snapshot_public_and_compatibility_apis_share_implementation(self):
        self.assertIn("freeze_pyg_data", graph_package.__all__)
        self.assertIs(freeze_pyg_data, graph_package.freeze_pyg_data)
        self.assertIs(freeze_pyg_data, snapshots.freeze_pyg_data)

    def test_freeze_clones_tensors_and_deep_copies_metadata(self):
        tensor = FakeTensor(7)
        metadata = {"route": ["a", "b"]}
        original = FakeData({"x": tensor, "metadata": metadata})

        frozen = freeze_pyg_data(original, torch_api=FakeTorch())

        self.assertIsNot(frozen, original)
        self.assertIsNot(frozen["x"], tensor)
        self.assertEqual(frozen["x"].value, 7)
        self.assertEqual(frozen["x"].events, ["detach", "clone", "cpu"])
        self.assertIsNot(frozen["metadata"], metadata)
        frozen["metadata"]["route"].append("c")
        self.assertEqual(metadata, {"route": ["a", "b"]})


class DebugUtilityCharacterizationTests(unittest.TestCase):
    def test_reporting_and_compatibility_apis_share_implementation(self):
        self.assertIs(save_memory_to_txt, reporting.save_memory_to_txt)
        self.assertIs(inspect_link_api, reporting.inspect_link_api)
        self.assertIs(simple_plot, reporting.simple_plot)

    def test_memory_dump_uses_indexed_legacy_text_format(self):
        previous_cwd = os.getcwd()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(output):
                    filepath = save_memory_to_txt(["first", {"v": 2}], "memory.txt")
                with open(filepath, "r", encoding="utf-8") as file:
                    contents = file.read()
                self.assertEqual(filepath, os.path.join(directory, "memory.txt"))
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(contents, "0: first\n1: {'v': 2}\n")
        self.assertIn(f"[INFO] buffer 已儲存到 {filepath}", output.getvalue())

    def test_inspection_prints_public_methods_and_attributes(self):
        class Link:
            attribute = 3

            def method(self):
                return None

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            inspect_link_api(Link())

        self.assertEqual(
            output.getvalue(),
            "METHODS: ['method']\nATTRS: ['attribute']\n",
        )


class FakePyplot:
    def __init__(self):
        self.calls = []

    def figure(self):
        self.calls.append(("figure",))

    def plot(self, values):
        self.calls.append(("plot", values))

    def xlabel(self, value):
        self.calls.append(("xlabel", value))

    def ylabel(self, value):
        self.calls.append(("ylabel", value))

    def savefig(self, value):
        self.calls.append(("savefig", value))

    def close(self):
        self.calls.append(("close",))


class PlottingUtilityCharacterizationTests(unittest.TestCase):
    def test_simple_plot_preserves_call_order_and_labels(self):
        pyplot = FakePyplot()
        simple_plot(
            [1, 3],
            xlabel="X",
            ylabel="Y",
            filename="result.png",
            pyplot_api=pyplot,
        )
        self.assertEqual(
            pyplot.calls,
            [
                ("figure",),
                ("plot", [1, 3]),
                ("xlabel", "X"),
                ("ylabel", "Y"),
                ("savefig", "result.png"),
                ("close",),
            ],
        )


if __name__ == "__main__":
    unittest.main()
