# =============================================================================
# RUN EVERY TEST WITHOUT PYTEST:   python tests/run_all.py
#
# Each test file runs in its OWN Python process. NGSolve and PyTorch load
# different native maths libraries; used together in one process they can
# crash (segmentation fault). The same rule is why run.py imports NGSolve
# only in data stages and PyTorch only in learning stages.
# (With pytest installed: python -m pytest tests/test_geometry.py, one file
#  at a time for the same reason.)
# =============================================================================

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

MODULES = ["test_geometry", "test_solver", "test_protocol", "test_models", "test_metrics_tuning"]

RUN_ONE_MODULE = """
import sys, traceback
sys.path.insert(0, %r)
import tests.%s as module
failures = 0
for name in sorted(dir(module)):
    if name.startswith("test_"):
        try:
            getattr(module, name)()
            print("PASS  %s." + name)
        except Exception:
            failures += 1
            print("FAIL  %s." + name)
            traceback.print_exc()
sys.exit(1 if failures else 0)
"""

failed_files = []
for name in MODULES:
    code = RUN_ONE_MODULE % (str(PROJECT), name, name, name)
    result = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT))
    if result.returncode != 0:
        failed_files.append(name)

if failed_files:
    print("\nFAILED: %s" % ", ".join(failed_files))
    sys.exit(1)
print("\nAll tests passed.")
