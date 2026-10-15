"""Executor mínimo para ambientes sem pytest (os testes continuam compatíveis com pytest).

  python tests/run_tests.py
"""

import importlib
import inspect
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    import os

    os.chdir(ROOT)
    failed = passed = 0
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        mod = importlib.import_module(f.stem)
        fixtures = {n: getattr(o, "__wrapped__", o) for n, o in vars(mod).items()
                    if callable(o) and hasattr(o, "_pytestfixturefunction") or n in ("cfg",)}
        for name, fn in vars(mod).items():
            if not (name.startswith("test_") and callable(fn)):
                continue
            kwargs = {}
            for p in inspect.signature(fn).parameters:
                fx = fixtures[p]
                kwargs[p] = fx() if callable(fx) else fx
            try:
                fn(**kwargs)
                passed += 1
                print(f"PASS {f.stem}::{name}")
            except Exception:
                failed += 1
                print(f"FAIL {f.stem}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passaram, {failed} falharam")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())