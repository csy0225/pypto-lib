# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compatibility entry point for the categorized MTP3 IPC device harness."""
from __future__ import annotations

from tests.step3p5.harnesses import _stage_whole_mtp3_ipc as _impl


def __getattr__(name: str):
    return getattr(_impl, name)


def main() -> int:
    return _impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
