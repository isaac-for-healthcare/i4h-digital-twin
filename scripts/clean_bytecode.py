# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove generated Python bytecode below this repository without following symlinks."""

import os
from pathlib import Path


def main():
    """Remove .pyc/.pyo files and empty cache directories, including local environments."""
    root = Path(__file__).resolve().parents[1]
    removed = 0
    for folder, directories, files in os.walk(root, topdown=False, followlinks=False):
        path = Path(folder)
        if '.git' in path.relative_to(root).parts:
            continue
        for name in files:
            file = path / name
            if file.suffix in {'.pyc', '.pyo'} and not file.is_symlink():
                file.unlink()
                removed += 1
        if path.name == '__pycache__' and not any(path.iterdir()):
            path.rmdir()
    print(f'Removed {removed} bytecode files from {root}')


if __name__ == '__main__':
    main()
