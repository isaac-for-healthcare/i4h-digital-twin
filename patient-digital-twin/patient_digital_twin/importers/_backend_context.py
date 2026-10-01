# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope the upstream projects' shared scripts namespace and relative paths."""

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

_BACKEND_LOCK = RLock()


def _script_modules():
    return [name for name in sys.modules if name == "scripts" or name.startswith("scripts.")]


@contextmanager
def backend_context(root):
    """Serialize our adapters and restore import/CLI state, even after failure.

    Upstream uses relative paths and a top-level scripts package. These are
    process-wide settings; unrelated threads must not depend on the cwd during
    inference. Use python_executable for process isolation when needed.
    """
    root = Path(root).resolve()
    with _BACKEND_LOCK:
        cwd, path, argv = Path.cwd(), sys.path[:], sys.argv
        saved = {name: sys.modules.pop(name) for name in _script_modules()}
        try:
            sys.path.insert(0, str(root))
            os.chdir(root)
            importlib.invalidate_caches()
            yield
        finally:
            for name in _script_modules():
                del sys.modules[name]
            sys.modules.update(saved)
            sys.path[:] = path
            sys.argv = argv
            os.chdir(cwd)
            importlib.invalidate_caches()
