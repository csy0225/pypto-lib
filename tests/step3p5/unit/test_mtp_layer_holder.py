from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.step3p5.mtp_layer_holder import (
    MtpLayerHolder,
    _mtp_build_output_dirs,
    _prepare_selected_programs,
)

_ROOT = Path(__file__).resolve().parents[3]
_HOLDER = _ROOT / "tools" / "step3p5" / "mtp_layer_holder.py"


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


def test_mtp_weight_reshape_preserves_ipc_provenance() -> None:
    source = _HOLDER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    reshape = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "W_reshape"
    )
    body = ast.get_source_segment(source, reshape)
    assert body is not None

    assert "source.reshape(tuple(shape))" in body
    assert "DeviceTensor(source.data_ptr" not in body


class _CleanupSignal(BaseException):
    pass


class _FakePrepare:
    def __init__(self, *, exit_error=None, close_error=None):
        self.exit_error = exit_error
        self.close_error = close_error
        self.exit_calls = []
        self.close_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_calls.append((exc_type, exc, tb))
        if self.exit_error is not None:
            raise self.exit_error
        return False

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _mtp_holder() -> MtpLayerHolder:
    holder = MtpLayerHolder([0], "/tmp/out", "/tmp/ckpt")
    holder.compiled = [object(), object(), object()]
    return holder


def test_mtp_partial_enter_uses_close_and_clears_state() -> None:
    holder = _mtp_holder()
    cm = _FakePrepare()

    def fail_before_enter():
        holder._prepare_cm = cm
        holder._prepare_entered = False
        raise RuntimeError("partial MTP enter")

    holder._enter_impl = fail_before_enter
    with pytest.raises(RuntimeError, match="partial MTP enter"):
        holder.__enter__()

    assert cm.close_calls == 1
    assert holder._prepare_cm is None
    assert holder._runtime is None
    assert holder._prepare_entered is False


def test_mtp_cleanup_failure_preserves_runtime_for_retry() -> None:
    holder = _mtp_holder()
    runtime = object()
    cm = _FakePrepare(exit_error=_CleanupSignal("MTP close failed"))
    holder._prepare_cm = cm
    holder._prepare_entered = True
    holder._runtime = runtime

    with pytest.raises(_CleanupSignal, match="MTP close failed"):
        holder.__exit__(None, None, None)

    assert cm.exit_calls
    assert holder._prepare_cm is cm
    assert holder._runtime is runtime
    assert holder._prepare_entered is True

    cm.exit_error = None
    holder.__exit__(None, None, None)
    assert holder._prepare_cm is None
    assert holder._runtime is None
    assert holder._prepare_entered is False
