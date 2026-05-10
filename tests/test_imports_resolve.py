"""Import-resolution smoke test.

Catches the failure class where one module references a symbol that doesn't
exist in another module. py_compile catches syntax errors but NOT runtime
ImportError (the import is inside a function body or relative). Runtime
import-resolution check catches both.

Real-world incident this prevents (commit b369289 → 752bb11):
  src/execute.py:225 had `from tempo_octave import clamp_octave`
  src/tempo_octave.py never exported clamp_octave (only normalize_tempo)
  Result: every render failed with ImportError. py_compile passed; tests
  passed (none touched stretch_and_pitch); 60 cohort renders trashed.

This test imports each src module fresh + walks function-body imports.
~50 ms total. Adds a hard guard against the same class of bug.
"""
from __future__ import annotations

import importlib
import sys
import ast
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _src_modules() -> list[str]:
    """All .py modules in src/ that are safe to import without GPU/network."""
    skip = {
        # Modules that require torch/transformers + GPU at import time —
        # too expensive for unit-test smoke. They're caught indirectly when
        # the modules that import them resolve.
        "analyze",       # pulls torchaudio, demucs, librosa
        "execute",       # pulls torchaudio, pyrubberband
        "director",      # heavy
        "main",          # CLI entry — argparse side effects
        "beat_this_wrapper",
        "bs_roformer_wrapper",
        "clap_wrapper",
        "training",      # subdir
    }
    out = []
    for p in sorted(SRC_DIR.glob("*.py")):
        name = p.stem
        if name.startswith("_") or name in skip:
            continue
        out.append(name)
    return out


@pytest.fixture(autouse=True)
def _ensure_src_on_path():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


@pytest.mark.parametrize("module", _src_modules())
def test_module_imports(module: str) -> None:
    """Each src/ module imports cleanly. No ImportError on first-party
    references; missing third-party deps (torch, transformers, etc) are
    skipped to keep the test runnable on laptop CI without heavy installs.
    The function-body test below catches first-party API mismatches even
    when the module can't import at top level here.
    """
    try:
        importlib.import_module(module)
    except ImportError as e:
        msg = str(e)
        third_party = ('torch', 'transformers', 'demucs', 'pyrubberband',
                       'librosa', 'soundfile', 'beat_this', 'madmom',
                       'laion_clap', 'lion_pytorch', 'bitsandbytes',
                       'scipy', 'h5py', 'pyloudnorm', 'audiocraft')
        if any(t in msg for t in third_party):
            pytest.skip(f"third-party dep missing: {e}")
        raise


def _function_body_imports(py_path: Path) -> list[tuple[int, str, list[str]]]:
    """Walk the module AST, return [(lineno, module_name, [imported_names])]
    for `from <mod> import a, b` statements *inside function bodies*.

    These lazy imports are common in this codebase (avoid import-time cost
    of heavy deps). py_compile + module-level import smoke do not catch
    them; they only fail at runtime when the function executes.
    """
    out: list[tuple[int, str, list[str]]] = []
    try:
        tree = ast.parse(py_path.read_text())
    except Exception:
        return out
    class V(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0
        def visit_FunctionDef(self, node):
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_ImportFrom(self, node):
            if self.depth > 0 and node.module:
                names = [a.name for a in node.names if a.name != '*']
                out.append((node.lineno, node.module, names))
    V().visit(tree)
    return out


@pytest.mark.parametrize("py_path", sorted(SRC_DIR.glob("*.py")),
                         ids=lambda p: p.stem)
def test_function_body_imports_resolve(py_path: Path) -> None:
    """Symbols imported inside function bodies actually exist at the source.

    This is the test that would have caught the clamp_octave incident.
    Walks AST → for each `from X import Y, Z` inside a function → imports X
    and asserts each name is an attribute of X.

    Modules that can't be imported in this env (GPU deps) are skipped — we
    log them rather than fail. Tradeoff: we lose coverage for heavy modules,
    keep test runnable on laptop CI.
    """
    skip_modules = {
        "torch", "torchaudio", "transformers", "demucs", "pyrubberband",
        "librosa", "soundfile", "beat_this", "madmom", "laion_clap",
        "lion_pytorch", "sophia_optimizer", "bitsandbytes", "trl", "peft",
        "accelerate", "datasets", "audiocraft", "bs_roformer",
    }
    body_imports = _function_body_imports(py_path)
    failures: list[str] = []
    for lineno, mod_name, names in body_imports:
        # Skip 3rd-party heavy deps; we can't import them on laptop.
        head = mod_name.split('.')[0]
        if head in skip_modules:
            continue
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            # First-party module that can't import — likely transitively
            # depends on a heavy dep. Skip but log.
            print(f"  [skip] {py_path.name}:{lineno} {mod_name} ({e})")
            continue
        for name in names:
            if not hasattr(mod, name):
                failures.append(
                    f"{py_path.name}:{lineno} `from {mod_name} import {name}` "
                    f"— {mod_name} has no attribute '{name}'")
    if failures:
        pytest.fail("\n".join(failures))
