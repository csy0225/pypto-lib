# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Card-free contracts for the categorized Step3p5 test tree."""
from __future__ import annotations

from pathlib import Path


def test_step3p5_top_level_contains_only_index_and_compatibility_files() -> None:
    root = Path(__file__).resolve().parents[1]
    files = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != ".DS_Store"
    }
    assert files == {
        "__init__.py",
        "README.md",
    }


def test_step3p5_category_packages_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for category in (
        "common",
        "probes",
        "harnesses",
        "unit",
        "system",
        "precision",
        "ci",
    ):
        package = root / category
        assert package.is_dir()
        assert (package / "__init__.py").is_file()
