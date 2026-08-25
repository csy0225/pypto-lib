# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Step3p5 logical-rank to physical-device topology checks."""
from __future__ import annotations

from collections.abc import Iterable


def validate_consecutive_device_ids(
    device_ids: Iterable[int],
    *,
    owner: str,
) -> list[int]:
    """Return device IDs after validating the dev_offset + rank contract."""
    resolved = list(device_ids)
    if not resolved or any(
        type(device_id) is not int or device_id < 0
        for device_id in resolved
    ):
        raise ValueError(
            f"{owner} device_ids must contain non-negative integers"
        )
    expected = list(range(resolved[0], resolved[0] + len(resolved)))
    if resolved != expected:
        raise ValueError(
            f"{owner} logical ranks require consecutive ordered device_ids, "
            f"got {resolved}"
        )
    return resolved
