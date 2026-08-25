# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from tools.step3p5.whole_decode_holder import (
    _RESIDENT_GATE_ARGS,
    _RESIDENT_ROPE_ARGS,
    WholeDecodeHolder,
    _resident_const_args,
)


class _CleanupSignal(BaseException):
    pass


class _FakeRuntime:
    def __init__(
        self,
        *,
        fail_at: int | None = None,
        free_fail_at: int | None = None,
    ) -> None:
        self.fail_at = fail_at
        self.free_fail_at = free_fail_at
        self.uploaded = []
        self.free_attempts = []
        self.freed = []

    def alloc_stacked_tensor(self, host_tensor):
        if self.fail_at is not None and len(self.uploaded) == self.fail_at:
            raise RuntimeError("upload failed")
        stacked = object()
        self.uploaded.append((host_tensor, stacked))
        return stacked

    def free_stacked_tensor(self, stacked) -> None:
        attempt = len(self.free_attempts)
        self.free_attempts.append(stacked)
        if self.free_fail_at is not None and attempt == self.free_fail_at:
            raise _CleanupSignal("free failed")
        self.freed.append(stacked)


class _FakePrepare:
    def __init__(
        self,
        *,
        exit_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.exit_error = exit_error
        self.close_error = close_error
        self.exit_calls = []
        self.close_calls = 0

    def __exit__(self, exc_type, exc, tb):
        self.exit_calls.append((exc_type, exc, tb))
        if self.exit_error is not None:
            raise self.exit_error
        return False

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _holder_with_constants(
    *,
    fail_at: int | None = None,
    free_fail_at: int | None = None,
):
    holder = WholeDecodeHolder([0, 1], "/tmp/out", "/tmp/ckpt", kv_ipc=False)
    holder.rt = _FakeRuntime(fail_at=fail_at, free_fail_at=free_fail_at)
    holder._rope_ready = True
    originals = {}
    for name in _RESIDENT_ROPE_ARGS + _RESIDENT_GATE_ARGS:
        tensor = torch.zeros((2, 2), dtype=torch.float32)
        originals[name] = tensor
        setattr(holder, name, tensor)
    return holder, originals


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("none", ()),
        ("rope", _RESIDENT_ROPE_ARGS),
        ("gate", _RESIDENT_GATE_ARGS),
        ("all", _RESIDENT_ROPE_ARGS + _RESIDENT_GATE_ARGS),
    ],
)
def test_resident_mode_selects_expected_constants(monkeypatch, mode, expected) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", mode)
    assert _resident_const_args() == expected


def test_invalid_resident_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "invalid")
    with pytest.raises(ValueError, match="invalid"):
        _resident_const_args()


def test_invalid_resident_mode_is_rejected_before_prepare(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "invalid")
    holder = WholeDecodeHolder([0, 1], "/tmp/out", "/tmp/ckpt", kv_ipc=False)

    class _NeverPrepare:
        def prepare(self, *, persistent):
            del persistent
            raise AssertionError("prepare must not be called")

    holder.compiled = _NeverPrepare()

    with pytest.raises(ValueError, match="invalid"):
        holder.__enter__()

    assert holder._prepare_cm is None
    assert holder.rt is None


@pytest.mark.parametrize(
    "device_ids",
    ([], [0, 2], [1, 0], [0, 0]),
)
def test_holder_rejects_nonconsecutive_logical_rank_mapping(
    device_ids,
) -> None:
    with pytest.raises(ValueError, match="device_ids|logical ranks"):
        WholeDecodeHolder(
            device_ids,
            "/tmp/out",
            "/tmp/ckpt",
            kv_ipc=False,
        )


def test_holder_accepts_consecutive_nonzero_device_range() -> None:
    holder = WholeDecodeHolder(
        list(range(8, 16)),
        "/tmp/out",
        "/tmp/ckpt",
        kv_ipc=False,
    )
    assert holder.device_ids == list(range(8, 16))
    assert holder.dev_offset == 8


def test_set_hidden_rejects_rank_mismatch() -> None:
    holder = WholeDecodeHolder([0, 1], "/tmp/out", "/tmp/ckpt", kv_ipc=False)
    holder._consts = {"BATCH": 2, "HIDDEN": 4}
    holder.current_hidden = torch.zeros(
        (2, 2, 4),
        dtype=torch.bfloat16,
    )
    holder.num_tokens_per_owner = torch.zeros(2, dtype=torch.int32)
    hidden = torch.zeros((2, 2, 4), dtype=torch.bfloat16)
    hidden[1, 0, 0] = 1

    with pytest.raises(ValueError, match="requires identical hidden"):
        holder.set_hidden(hidden)


def test_resident_constants_are_uploaded_once_and_rebound(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "all")
    holder, originals = _holder_with_constants()

    owned = holder._make_resident_consts()

    assert tuple(owned) == _RESIDENT_ROPE_ARGS + _RESIDENT_GATE_ARGS
    assert len(holder.rt.uploaded) == 8
    for name, original in originals.items():
        assert getattr(holder, name) is owned[name]
        assert any(host is original for host, _stacked in holder.rt.uploaded)


def test_partial_upload_failure_rolls_back_owned_constants(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "all")
    holder, originals = _holder_with_constants(fail_at=2)

    with pytest.raises(RuntimeError, match="upload failed"):
        holder._make_resident_consts()

    assert len(holder.rt.freed) == 2
    for name, original in originals.items():
        assert getattr(holder, name) is original


def test_partial_upload_failure_attempts_all_frees_and_preserves_upload_error(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "all")
    holder, originals = _holder_with_constants(fail_at=3, free_fail_at=0)

    with pytest.raises(RuntimeError, match="upload failed"):
        holder._make_resident_consts()

    uploaded = [stacked for _host, stacked in holder.rt.uploaded]
    assert holder.rt.free_attempts == list(reversed(uploaded))
    assert holder._resident_consts == {"rope_cs": uploaded[2]}
    assert holder._resident_const_hosts["rope_cs"] is originals["rope_cs"]
    assert holder.rope_cs is uploaded[2]
    for name in ("rope_cf", "rope_sf"):
        assert getattr(holder, name) is originals[name]


def test_partial_upload_rollback_retries_before_runtime_close(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "all")
    holder, originals = _holder_with_constants(fail_at=3, free_fail_at=0)
    runtime = holder.rt
    holder.rt = None
    prepare_cm = _FakePrepare()

    def enter_impl(_resident_const_names):
        holder._prepare_cm = prepare_cm
        holder._prepare_entered = True
        holder.rt = runtime
        holder._make_resident_consts()

    holder.compiled = object()
    monkeypatch.setattr(holder, "_enter_impl", enter_impl)

    with pytest.raises(RuntimeError, match="upload failed"):
        holder.__enter__()

    # The first rollback free failed, so __enter__ cleanup retried it while the
    # prepared runtime was still live, then closed the runtime.
    uploaded = [stacked for _host, stacked in runtime.uploaded]
    assert runtime.free_attempts == list(reversed(uploaded)) + [uploaded[2]]
    assert prepare_cm.exit_calls and prepare_cm.exit_calls[0][0] is RuntimeError
    assert holder._resident_consts == {}
    assert holder._resident_const_hosts == {}
    assert holder._prepare_cm is None
    assert holder.rt is None
    for name, original in originals.items():
        assert getattr(holder, name) is original


def test_failed_enter_cleans_residents_and_prepared_runtime(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "none")
    holder, _originals = _holder_with_constants(free_fail_at=0)
    runtime = holder.rt
    holder.rt = None
    prepare_cm = _FakePrepare(exit_error=_CleanupSignal("prepare cleanup failed"))
    owned = {"first": object(), "second": object()}

    def fail_after_prepare(_resident_const_names):
        holder.rt = runtime
        holder._prepare_cm = prepare_cm
        holder._prepare_entered = True
        holder._resident_consts = owned
        raise RuntimeError("late enter failed")

    holder.compiled = object()
    monkeypatch.setattr(holder, "_enter_impl", fail_after_prepare)

    with pytest.raises(RuntimeError, match="late enter failed"):
        holder.__enter__()

    assert runtime.free_attempts == [owned["second"], owned["first"]]
    assert prepare_cm.exit_calls == []
    assert holder._resident_consts == {"second": owned["second"]}
    assert holder._prepare_cm is prepare_cm
    assert holder.rt is runtime

    # The first free failed, so the prepared runtime must stay available for a
    # later explicit retry even though __enter__ preserved its original error.
    runtime.free_fail_at = None
    prepare_cm.exit_error = None
    holder.__exit__(None, None, None)
    assert holder._resident_consts == {}
    assert holder._prepare_cm is None
    assert holder.rt is None
    assert holder._rope_ready is False


def test_cleanup_retry_preserves_runtime_then_finishes(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "rope")
    holder, originals = _holder_with_constants(free_fail_at=0)
    runtime = holder.rt
    holder._resident_consts = holder._make_resident_consts()
    prepare_cm = _FakePrepare()
    holder._prepare_cm = prepare_cm
    holder._prepare_entered = True

    with pytest.raises(_CleanupSignal, match="free failed"):
        holder.__exit__(None, None, None)

    # The first free failed, so the worker and its ownership journal remain
    # live for an explicit retry. Successful frees are restored immediately.
    assert holder._prepare_cm is prepare_cm
    assert holder.rt is runtime
    assert len(holder._resident_consts) == 1
    assert prepare_cm.exit_calls == []
    assert any(
        getattr(holder, name) is originals[name]
        for name in _RESIDENT_ROPE_ARGS
    )

    runtime.free_fail_at = None
    holder.__exit__(None, None, None)

    assert holder._resident_consts == {}
    assert holder._resident_const_hosts == {}
    assert holder._prepare_cm is None
    assert holder.rt is None
    assert holder._rope_ready is False
    assert prepare_cm.exit_calls
    for name in _RESIDENT_ROPE_ARGS:
        assert getattr(holder, name) is originals[name]


def test_exit_attempts_all_cleanup_before_raising_first_failure(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "rope")
    holder, _originals = _holder_with_constants(free_fail_at=0)
    runtime = holder.rt
    holder._resident_consts = holder._make_resident_consts()
    expected_frees = list(
        reversed([stacked for _host, stacked in runtime.uploaded])
    )
    prepare_cm = _FakePrepare(exit_error=RuntimeError("prepare cleanup failed"))
    holder._prepare_cm = prepare_cm
    holder._prepare_entered = True

    with pytest.raises(_CleanupSignal, match="free failed"):
        holder.__exit__(None, None, None)

    assert runtime.free_attempts == expected_frees
    assert prepare_cm.exit_calls == []
    assert holder._resident_consts
    assert holder._prepare_cm is prepare_cm
    assert holder.rt is runtime

    runtime.free_fail_at = None
    with pytest.raises(RuntimeError, match="prepare cleanup failed"):
        holder.__exit__(None, None, None)

    assert len(prepare_cm.exit_calls) == 1
    assert holder._prepare_cm is prepare_cm
    assert holder.rt is runtime

    prepare_cm.exit_error = None
    holder.__exit__(None, None, None)
    assert holder._resident_consts == {}
    assert holder._prepare_cm is None
    assert holder.rt is None
    assert holder._rope_ready is False


def test_prepare_cm_partial_enter_uses_close_contract(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "none")
    holder = WholeDecodeHolder([0, 1], "/tmp/out", "/tmp/ckpt", kv_ipc=False)

    class _PartialEnter:
        def __init__(self):
            self.close_calls = 0

        def __enter__(self):
            raise RuntimeError("partial prepare enter failed")

        def __exit__(self, *_exc):
            raise AssertionError("unentered context must not receive __exit__")

        def close(self):
            self.close_calls += 1

    prepare_cm = _PartialEnter()

    class _Compiled:
        def prepare(self, *, persistent):
            assert persistent is True
            return prepare_cm

    holder.compiled = _Compiled()
    monkeypatch.setattr(
        holder,
        "_enter_impl",
        lambda _names: holder._prepare_runtime(),
    )

    with pytest.raises(RuntimeError, match="partial prepare enter failed"):
        holder.__enter__()

    assert prepare_cm.close_calls == 1
    assert holder._prepare_cm is None
    assert holder.rt is None
    assert holder._prepare_entered is False


def test_distributed_worker_prepare_enter_is_identity_contract() -> None:
    from pypto.runtime.distributed_runner import DistributedWorker

    sentinel = object()
    assert DistributedWorker.__enter__(sentinel) is sentinel


def test_terminal_cleanup_invalidates_rope_ready(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "none")
    holder, _originals = _holder_with_constants()
    holder._rope_ready = True
    holder._finalize_closed_state()
    assert holder._rope_ready is False


def test_resident_rope_rejects_host_side_mutation(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_H4_RESIDENT", "rope")
    holder, _originals = _holder_with_constants()
    holder._resident_consts = holder._make_resident_consts()

    with pytest.raises(ValueError, match="worker-resident constants"):
        holder.set_meta(rope_cf=torch.ones((2, 2), dtype=torch.float32))
