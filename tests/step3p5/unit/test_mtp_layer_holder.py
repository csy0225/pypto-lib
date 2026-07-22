from __future__ import annotations

from pathlib import Path

import pytest

from tools.step3p5.mtp_layer_holder import (
    _mtp_build_output_dirs,
    _prepare_selected_programs,
)


def test_mtp_build_output_dirs_are_unique_and_layer_identified(
    tmp_path: Path,
) -> None:
    output_dirs = _mtp_build_output_dirs(
        str(tmp_path),
        num_layers=3,
        run_tag="fixed-run",
    )

    assert len(output_dirs) == 3
    assert len(set(output_dirs)) == 3
    assert tuple(Path(path).name for path in output_dirs) == (
        "MtpLayerHidden0_fixed-run",
        "MtpLayerHidden1_fixed-run",
        "MtpLayerHidden2_fixed-run",
    )
    assert all(Path(path).parent == tmp_path for path in output_dirs)


@pytest.mark.parametrize("num_layers", [0, -1])
def test_mtp_build_output_dirs_reject_invalid_layer_count(
    tmp_path: Path,
    num_layers: int,
) -> None:
    with pytest.raises(ValueError, match="num_layers must be positive"):
        _mtp_build_output_dirs(
            str(tmp_path),
            num_layers=num_layers,
            run_tag="fixed-run",
        )


@pytest.mark.parametrize("run_tag", ["", "nested/path"])
def test_mtp_build_output_dirs_reject_invalid_run_tag(
    tmp_path: Path,
    run_tag: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="run_tag must be one non-empty path component",
    ):
        _mtp_build_output_dirs(
            str(tmp_path),
            num_layers=3,
            run_tag=run_tag,
        )


def test_selected_mtp_prepares_one_multi_program_worker() -> None:
    calls = []
    runtime_cm = object()

    class Program:
        def __init__(self, name: str) -> None:
            self.name = name

        def prepare(self, *, extra_compiled=()):
            calls.append((self.name, tuple(item.name for item in extra_compiled)))
            return runtime_cm

    programs = [Program("mtp45"), Program("mtp46"), Program("mtp47")]

    assert _prepare_selected_programs(programs) is runtime_cm
    assert calls == [("mtp45", ("mtp46", "mtp47"))]


@pytest.mark.parametrize("num_programs", [0, 1, 2, 4])
def test_selected_mtp_rejects_wrong_program_count(num_programs: int) -> None:
    programs = [object() for _ in range(num_programs)]
    with pytest.raises(ValueError, match="exactly 3 compiled programs"):
        _prepare_selected_programs(programs)
