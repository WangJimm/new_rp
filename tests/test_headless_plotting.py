"""Exercise PNG rendering without training, SUMO or any GUI imports."""
from __future__ import annotations
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PLOTTING = ROOT / "rp/reporting/plotting.py"


def load_plotter():
    spec = importlib.util.spec_from_file_location("headless_plotting_under_test", PLOTTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HeadlessPlottingTests(unittest.TestCase):
    def test_pngs_and_background_gc_never_import_tk_or_pyplot(self):
        code = r"""
import gc, importlib.abc, importlib.util, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
class RejectGUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tkinter' or fullname.startswith('tkinter.') or fullname == 'matplotlib.pyplot' or fullname.endswith('backend_tkagg'):
            raise AssertionError('Unexpected GUI/pyplot import: ' + fullname)
sys.meta_path.insert(0, RejectGUI())
spec=importlib.util.spec_from_file_location('plotter', sys.argv[1])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
folder=Path(sys.argv[2])
module.simple_plot([1,2,1],filename=str(folder/'main.png'))
def render(index):
    path=folder/f'worker_{index}.png'
    module.simple_plot([index,2,3],xlabel='Episode',ylabel='Reward',filename=str(path))
    gc.collect()
    return path.read_bytes()[:8]
with ThreadPoolExecutor(max_workers=2) as workers:
    signatures=list(workers.map(render, range(6)))
assert all(s==b'\x89PNG\r\n\x1a\n' for s in signatures)
assert (folder/'main.png').read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
assert not any(name=='tkinter' or name.startswith('tkinter.') or name=='matplotlib.pyplot' for name in sys.modules)
print('HEADLESS_PNG_OK')
"""
        with tempfile.TemporaryDirectory() as temp:
            env = os.environ.copy()
            # A GUI-preferring machine must still be safe with the explicit canvas.
            env["MPLBACKEND"] = "TkAgg"
            env["MPLCONFIGDIR"] = str(Path(temp) / "mplconfig")
            result = subprocess.run([sys.executable, "-c", code, str(PLOTTING), temp],
                                    env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("HEADLESS_PNG_OK", result.stdout)

    def test_explicit_adapter_is_closed_when_saving_fails(self):
        class BrokenAdapter:
            closed = False
            def figure(self): pass
            def plot(self, values): pass
            def xlabel(self, value): pass
            def ylabel(self, value): pass
            def savefig(self, path): raise OSError("test save failure")
            def close(self): self.closed = True
        adapter = BrokenAdapter()
        with self.assertRaisesRegex(OSError, "test save failure"):
            load_plotter().simple_plot([1, 2], pyplot_api=adapter)
        self.assertTrue(adapter.closed)

    def test_training_selects_agg_before_importing_heavy_dependencies(self):
        tree = ast.parse((ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8-sig"))
        calls = [node for node in tree.body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                 and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "matplotlib"
                 and node.value.func.attr == "use"]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call.value.args[0].value, "Agg")
        self.assertTrue(next(k.value.value for k in call.value.keywords if k.arg == "force"))
        heavy = [node.lineno for node in tree.body if isinstance(node, ast.Import)
                 and any(alias.name in ("traci", "torch") for alias in node.names)]
        self.assertLess(call.lineno, min(heavy))


if __name__ == "__main__": unittest.main()
