"""Offline notebook checks; no Colab account, CUDA or model downloads required."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_setup.ipynb"
DOCUMENT = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {cell["id"]: "".join(cell["source"]) for cell in DOCUMENT["cells"]}


def execute(cell_id, namespace=None):
    namespace = {} if namespace is None else namespace
    with redirect_stdout(io.StringIO()):
        exec(compile(CELLS[cell_id], f"{NOTEBOOK.name}:{cell_id}", "exec"), namespace)
    return namespace


class ColabNotebookTests(unittest.TestCase):
    def test_notebook_has_clean_valid_python_cells(self):
        self.assertEqual(DOCUMENT["nbformat"], 4)
        for cell in DOCUMENT["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_old_python_reports_requirement(self):
        with patch.object(sys, "version_info", (3, 11)):
            with self.assertRaisesRegex(RuntimeError, "Python >=3.12"):
                execute("config")

    def test_clone_rerun_and_ref_change(self):
        with tempfile.TemporaryDirectory() as folder:
            remote = Path(folder) / "remote"
            checkout = Path(folder) / "checkout"
            remote.mkdir()

            def git(*args):
                return subprocess.run(
                    ["git", *args], cwd=remote, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()

            git("init", "--initial-branch=main")
            git("config", "user.name", "Notebook Test")
            git("config", "user.email", "notebook@example.invalid")
            (remote / "pyproject.toml").write_text('[project]\nname="fixture"\n')
            git("add", "pyproject.toml")
            git("commit", "-m", "Initial fixture")
            first_commit = git("rev-parse", "HEAD")
            namespace = execute("config")
            namespace.update(REPO_URL=str(remote), REPO_REF="main", REPO_DIR=checkout)
            execute("checkout", namespace)
            execute("checkout", namespace)

            (remote / "marker.txt").write_text("new commit")
            git("add", "marker.txt")
            git("commit", "-m", "Second fixture")
            execute("checkout", namespace)
            self.assertTrue((checkout / "marker.txt").is_file())
            namespace["REPO_REF"] = first_commit
            execute("checkout", namespace)
            self.assertFalse((checkout / "marker.txt").exists())

            # Both tracked edits and untracked files must survive a failed rerun.
            (checkout / "pyproject.toml").write_text("local edit")
            with self.assertRaisesRegex(RuntimeError, "локальні зміни"):
                execute("checkout", namespace)
            self.assertEqual((checkout / "pyproject.toml").read_text(), "local edit")
            (checkout / "pyproject.toml").write_text('[project]\nname="fixture"\n')
            (checkout / "personal.txt").write_text("keep me")
            with self.assertRaisesRegex(RuntimeError, "локальні зміни"):
                execute("checkout", namespace)
            self.assertEqual((checkout / "personal.txt").read_text(), "keep me")

    def test_existing_non_checkout_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            namespace = execute("config")
            namespace["REPO_DIR"] = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "не є окремим Git checkout"):
                execute("checkout", namespace)
            self.assertTrue(Path(folder).is_dir())

    def test_missing_torch_reports_runtime_selection(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "PyTorch відсутній"):
                execute("gpu")

    def test_cpu_runtime_fails_with_actionable_message(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "Change runtime type"):
                execute("gpu")
        torch.tensor.assert_not_called()

    def test_cuda_smoke_runs_arithmetic_on_gpu(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.__version__ = "test"
        torch.tensor.return_value.__mul__.return_value.sum.return_value.item.return_value = 14.0
        torch.cuda.get_device_properties.return_value = SimpleNamespace(
            name="Test GPU", total_memory=16 * 1024**3,
        )
        with patch.dict(sys.modules, {"torch": torch}):
            execute("gpu")
        torch.tensor.assert_called_once_with([1.0, 2.0, 3.0], device="cuda")
        torch.cuda.synchronize.assert_called_once_with()

    def test_cuda_operation_failure_is_reported(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.tensor.side_effect = RuntimeError("CUDA driver error")
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA-обчислення не вдалося"):
                execute("gpu")

    def test_install_uses_kernel_python_and_propagates_pip_failure(self):
        namespace = execute("config")
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                execute("install", namespace)
        self.assertEqual(run.call_args.args[0], [
            sys.executable, "-m", "pip", "install", "--editable", str(namespace["REPO_DIR"]),
        ])

    def test_imported_numpy_upgrade_requires_restart(self):
        namespace = execute("config")
        with patch("subprocess.run"), patch("importlib.metadata.version", return_value="new"), \
                patch.dict(sys.modules, {"numpy": SimpleNamespace(__version__="old")}):
            with self.assertRaisesRegex(RuntimeError, "Restart session"):
                execute("install", namespace)


if __name__ == "__main__":
    unittest.main()
