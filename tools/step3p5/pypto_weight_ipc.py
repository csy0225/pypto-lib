#!/usr/bin/env python3
"""Step3p5 weight residency via device-IPC address-map (card-free wiring).

GOAL
----
Live serving cannot duplicate the ~47GB/rank model weights into a separate
pypto worker. Instead we reuse vLLM's already-resident sharded weights via a
device-IPC address-map, exactly like the KV one-key-pool
(``_stage_kvpool_pageattn.py`` + ``_stage_attn_worker.py::attn_setup`` one-key
path). The runtime import support already exists:
``CTRL_IMPORT_IPC = 12`` in
``pypto/runtime/src/common/hierarchical/worker_manager.h`` and
``DistributedWorker.import_ipc`` in
``pypto/python/pypto/runtime/distributed_runner.py``. NO runtime rebuild.

This module is the weight analogue of the KV pool. Two roles:

  exporter (vLLM-side, one per TP rank)
      Consolidates THIS rank's PyPTO-layout weight bundle (the exact bundle
      ``weight_loader.load_step3p5_weights_for_rank`` produces — i.e. vLLM's
      resident sharded params already translated through
      ``weight_translate.build_vllm_to_pypto_transform_plan``. In the
      released native-W8A8 path routed-MoE weights remain INT8 with FP32
      scales (no dequant), while MTP matrices remain checkpoint-native BF16.
      It packs those tensors into ONE contiguous device buffer, calls
      ``aclrtIpcMemGetExportKey`` ONCE, writes the 256-byte key +
      ``pypto_weight_map.rank{r}.json`` to a shared dir, and keeps the export
      handle alive for the whole serving life.

  importer (pypto worker-side, mirrors KVPoolMap)
      Reads the key + map, calls ``rt.import_ipc(key)`` ONCE to map the whole
      pool into the worker's address space, then builds a per-bundle-key
      ``DeviceTensor(peer_base + offset, shape, dtype)`` — zero-copy, no
      H2D/D2H, exactly the kernel-arg shape ``Step3p5DecodeFwd`` consumes.

KEY+MAP SCHEMA
--------------
``pypto_weight.key.rank{r}``  : raw 256-byte ``aclrtIpcMemGetExportKey`` blob.

``pypto_weight_map.rank{r}.json`` ::

    {
      "version": 1,
      "rank": <int>,
      "tp_world_size": <int>,
      "pool_bytes": <int>,            # total consolidated buffer size
      "pool_dtype_bytes": 2,          # legacy default; each entry carries its own dtype
      "map": {
        "<bundle_key>": {             # e.g. "wq_full", "moe_w13_r", "lm_head_weight"
          "offset": <int>,            # byte offset within the pool
          "shape": [<int>, ...],      # resident physical shape
          "dtype": "bfloat16"|"float32"|"int8",
          "layout": "ND"|"FRACTAL_NZ",
          "nbytes": <int>
        },
        ...
      }
    }

The ``map`` keys are exactly the ``models.step3p5.weight_loader.KEY_*``
constants so the worker can hand the whole-decode program the same names it
already expects. Router gate matrices (legacy or explicit ``*_NK`` key)
and ``KEY_MOE_ROUTER_BIAS`` are FP32; every other unquantized key is BF16.

PADDING / ALIGNMENT
-------------------
Each weight is laid out at an offset 512-byte aligned (>= any tile/UB row
alignment pypto requires) and the per-key ``nbytes`` is the tensor's true
byte size; the gap to the next 512-byte boundary is dead padding. This keeps
``DeviceTensor(base+off, shape, dtype)`` valid regardless of dtype. We do NOT
pack fp32+bf16 into a single dtype view; the pool is a raw byte buffer and
each entry carries its own dtype in the map.

CARD-FREE STATUS
----------------
Authored as a self-consistent deliverable. The exporter/importer are written
to mirror the PROVEN KV-pool pattern (``_stage_kvpool_pageattn.py`` exporter
+ ``_stage_attn_worker.py`` one-key ``attn_setup`` import) line-for-line in
the IPC surface. What is NOT verifiable without a device is marked
``OPEN_DEVICE_QUESTION`` in the docstrings below; the lead validates on
cards 8-15 later.

OPEN_DEVICE_QUESTION (1): does ``aclrtIpcMemGetExportKey`` accept a
consolidated buffer that WE ``aclrtMalloc``-ed (KV pool does exactly this in
``_stage_kvpool_pageattn.exporter``), vs the KV backend which exports
vLLM's own MemPool-reallocated tensors? For weights we own the buffer
end-to-end (we allocate + fill + export), so this is strictly simpler than
the KV case. Verified pattern: ``_stage_kvpool_pageattn.exporter`` lines
``aclrtMalloc`` -> fill via ``aclrtMemcpy`` H2D -> ``aclrtIpcMemGetExportKey``.

OPEN_DEVICE_QUESTION (2): the worker's ``rt.import_ipc`` returns a peer
device pointer; building ``DeviceTensor(peer_base+off, ...)`` for a sub-span
of an imported region is exactly what ``KVPoolMap.paged_block`` does
(``_stage_attn_worker.py`` lines 933-958 + ``_stage_kvpool_pageattn.KVPoolMap``).
Confirmed by the KV pool PASS; no new runtime surface needed.

OPEN_DEVICE_QUESTION (3): can one IPC key cover a ~47GB/rank buffer? The KV
pool exports ~hundreds of MB per rank with one key. The ACL IPC handle is a
VA-mapping descriptor, not a size-limited token, and the import maps the
whole region. If a single-key 47GB export hits an ACL limit, the exporter's
``max_pool_bytes`` knob splits the bundle into N pools (N keys, N maps) — the
importer loops over them. Default N=1; raise only if the device rejects the
single-key export. (KV pool has never needed splitting; included as a
safety valve.)
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# --- ACL IPC constants (mirror _stage_kvpool_pageattn.py / _stage_attn_backend.py) ---
KEY_BUF = 256                      # aclrtIpcMemGetExportKey handle size
_HUGE_FIRST = 0                    # ACL_MEM_MALLOC_HUGE_FIRST
_H2D = 1                           # ACL_MEMCPY_HOST_TO_DEVICE
_ALIGN = 512                       # per-weight byte alignment inside the pool

# Canonical dtypes in the exported whole-decode pool.  The loader keeps norm
# tables in BF16, then ``_prepare_checkpoint_bundle`` promotes the small tables
# below to the FP32 dtypes consumed by the program ABI.
_PROGRAM_FP32_KEYS = {
    "input_rms_weight",
    "post_attn_rms_weight",
    "q_norm_weight",
    "k_norm_weight",
    "final_norm_weight",
    "mtp_enorm_weight",
    "mtp_hnorm_weight",
    "mtp_input_rms_weight",
    "mtp_post_attn_rms_weight",
    "mtp_q_norm_weight",
    "mtp_k_norm_weight",
    "mtp_shared_head_norm_weight",
}
_FP32_KEYS = _PROGRAM_FP32_KEYS | {
    "moe_gate_w",
    "moe_gate_w_nk",
    "moe_router_bias",
    "moe_w13_r_scale",
    "moe_w_down_r_scale",
}
_INT8_KEYS = {"moe_w13_r", "moe_w_down_r"}
_LAYOUT_ND = "ND"
_LAYOUT_FRACTAL_NZ = "FRACTAL_NZ"
_KNOWN_LAYOUTS = {_LAYOUT_ND, _LAYOUT_FRACTAL_NZ}
_LEGACY_ROUTED_KEYS = {
    "moe_w_gate_r",
    "moe_w_up_r",
    "moe_w_gate_r_scale",
    "moe_w_up_r_scale",
}


def _dtype_for(key: str, *, native_w8a8: bool = True) -> str:
    """Return the canonical logical dtype for one pool key.

    The packed routed ABI has two explicit variants: a BF16 reference bundle
    and the native W8A8 bundle used by the live decode path. The caller must
    select the variant; the default is the production native-W8A8 contract.
    """
    if key in _INT8_KEYS:
        return "int8" if native_w8a8 else "bfloat16"
    return "float32" if key in _FP32_KEYS else "bfloat16"


def _layout_for(key: str, *, native_w8a8: bool = True) -> str:
    """Return the resident physical layout for one pool key."""
    if native_w8a8 and key in _INT8_KEYS:
        return _LAYOUT_FRACTAL_NZ
    return _LAYOUT_ND


def _validate_layout_shape(key: str, shape: Tuple[int, ...], layout: str) -> None:
    """Reject an NZ label whose physical fractal axes are not explicit."""
    if layout not in _KNOWN_LAYOUTS:
        raise ValueError(f"weight {key!r} has unsupported layout {layout!r}")
    if layout == _LAYOUT_FRACTAL_NZ:
        if len(shape) < 4 or tuple(shape[-2:]) != (16, 32):
            raise ValueError(
                f"weight {key!r} FRACTAL_NZ physical shape must end in "
                f"[16, 32], got {shape!r}",
            )


def _torch_dtype(name: str):
    import torch  # noqa: PLC0415
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "float16": torch.float16,
    }.get(name, torch.bfloat16)


def _align_up(n: int, a: int = _ALIGN) -> int:
    return (n + a - 1) // a * a


# =============================================================================
# Fail-closed pool-map validation (stdlib-only, card-free).
# =============================================================================
class WeightMapInvalid(ValueError):
    """Raised when an exported weight pool-map fails fail-closed validation."""


# dtype name -> item size in bytes (mirrors _torch_dtype WITHOUT importing torch,
# so the validator stays stdlib-only and runs card-free).
_DTYPE_ITEMSIZE = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1}


def _tensor_dtype_name(tensor: Any) -> str:
    """Return a supported torch dtype name without importing torch."""
    try:
        name = str(tensor.dtype).removeprefix("torch.")
    except AttributeError as exc:
        raise TypeError("weight entry must expose a dtype") from exc
    if name not in _DTYPE_ITEMSIZE:
        raise TypeError(f"unsupported tensor dtype {name!r}")
    return name


def _infer_native_w8a8(bundle: Dict[str, Any]) -> bool:
    """Infer the routed bundle variant for legacy static callers."""
    if set(_ROUTED_FP32_SCALE_KEYS) & set(bundle):
        return True
    for key in _INT8_KEYS:
        tensor = bundle.get(key)
        if tensor is not None and _tensor_dtype_name(tensor) == "int8":
            return True
    return False

# native-W8A8 routed-expert contract (design rule 3 / hard-constraint 4):
# routed projection weights MUST be INT8; their per-channel scales MUST be FP32.
# A BF16-dequantized routed weight is a forbidden fallback and must be rejected.
_ROUTED_INT8_KEYS = ("moe_w13_r", "moe_w_down_r")
_ROUTED_FP32_SCALE_KEYS = (
    "moe_w13_r_scale",
    "moe_w_down_r_scale",
)
# Router gate matrix + bias are FP32 regardless of routed quantization.
_ROUTER_FP32_KEYS = (
    "moe_gate_w", "moe_gate_w_nk", "moe_router_bias",
)


def _prod(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def validate_weight_map(
    pool_map: Dict[str, Any],
    *,
    expected: Optional[Dict[str, Tuple[Tuple[int, ...], str]]] = None,
    native_w8a8: bool = True,
    allow_extra_keys: bool = True,
    align: int = _ALIGN,
) -> None:
    """Fail-closed validation of one rank's weight pool-map (stdlib-only).

    Raises :class:`WeightMapInvalid` on any violation. Pure dict/int math — no
    torch, no device, no pypto import — so it runs card-free and is the gate the
    model loader gets before declaring a rank *ready* (design §4.2 analogue,
    hard-constraints 3/4).

    This is the reusable *live-serving* core. The whole-network CI has its own
    MTP-aware superset gate in
    ``tests/step3p5/ci/run_whole_network_ci.py::_validate_pool_map`` (adds the
    MTP BF16/FP32 key requirements + KV non-alias); the two share the same
    structural + native-W8A8 rules and must stay in sync.

    Structural checks (always):
      - ``version == 1``; ``pool_bytes`` is a positive int;
      - ``map`` is a non-empty dict; each entry has
        offset/shape/dtype/layout/nbytes;
      - dtype/layout are known names; ``nbytes == prod(shape) * itemsize``;
      - ``offset`` is ``align``-byte aligned; ``nbytes > 0``;
      - ``[offset, offset + nbytes)`` lies inside ``pool_bytes``;
      - no two entries overlap.

    native-W8A8 checks (when ``native_w8a8``):
      - packed routed W13/down keys are INT8 (never a BF16 dequant);
      - packed W13/down scale keys are FP32;
      - router gate/bias, when present, are FP32.

    Expected cross-check (when ``expected`` is a ``{key: (shape, dtype)}`` map):
      - every expected key is present with matching shape and dtype;
      - unexpected keys are rejected unless ``allow_extra_keys``.
    """
    if not isinstance(pool_map, dict):
        raise WeightMapInvalid(
            f"pool_map must be a dict, got {type(pool_map).__name__}"
        )
    version = pool_map.get("version")
    if version != 1:
        raise WeightMapInvalid(f"unsupported map version {version!r} (want 1)")
    pool_bytes = pool_map.get("pool_bytes")
    if not isinstance(pool_bytes, int) or pool_bytes <= 0:
        raise WeightMapInvalid(
            f"pool_bytes must be a positive int, got {pool_bytes!r}"
        )
    entries = pool_map.get("map")
    if not isinstance(entries, dict) or not entries:
        raise WeightMapInvalid("map must be a non-empty dict")

    spans: List[Tuple[int, int, str]] = []
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise WeightMapInvalid(f"entry {key!r} must be a dict")
        for field in ("offset", "shape", "dtype", "layout", "nbytes"):
            if field not in entry:
                raise WeightMapInvalid(f"entry {key!r} missing field {field!r}")
        offset, shape, dtype, layout, nbytes = (
            entry["offset"],
            entry["shape"],
            entry["dtype"],
            entry["layout"],
            entry["nbytes"],
        )
        if not isinstance(offset, int) or offset < 0:
            raise WeightMapInvalid(
                f"{key!r} offset must be a non-negative int, got {offset!r}"
            )
        if not isinstance(nbytes, int) or nbytes <= 0:
            raise WeightMapInvalid(
                f"{key!r} nbytes must be a positive int, got {nbytes!r}"
            )
        if dtype not in _DTYPE_ITEMSIZE:
            raise WeightMapInvalid(f"{key!r} unknown dtype {dtype!r}")
        if layout not in _KNOWN_LAYOUTS:
            raise WeightMapInvalid(f"{key!r} unknown layout {layout!r}")
        if not isinstance(shape, (list, tuple)) or not shape:
            raise WeightMapInvalid(
                f"{key!r} shape must be a non-empty sequence, got {shape!r}"
            )
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise WeightMapInvalid(
                f"{key!r} shape must contain positive ints, got {shape!r}"
            )
        try:
            _validate_layout_shape(key, tuple(shape), layout)
        except ValueError as exc:
            raise WeightMapInvalid(str(exc)) from exc
        want_nbytes = _prod(shape) * _DTYPE_ITEMSIZE[dtype]
        if want_nbytes != nbytes:
            raise WeightMapInvalid(
                f"{key!r} nbytes {nbytes} != prod({list(shape)}) * "
                f"itemsize({dtype})={want_nbytes}"
            )
        if offset % align != 0:
            raise WeightMapInvalid(
                f"{key!r} offset {offset} not {align}-byte aligned"
            )
        end = offset + nbytes
        if end > pool_bytes:
            raise WeightMapInvalid(
                f"{key!r} span [{offset},{end}) exceeds pool_bytes {pool_bytes}"
            )
        spans.append((offset, end, key))

    spans.sort()
    for (o0, e0, k0), (o1, e1, k1) in zip(spans, spans[1:]):
        if o1 < e0:
            raise WeightMapInvalid(
                f"entries {k0!r} [{o0},{e0}) and {k1!r} [{o1},{e1}) overlap"
            )

    legacy = sorted(_LEGACY_ROUTED_KEYS & set(entries))
    if legacy:
        raise WeightMapInvalid(
            f"map contains removed routed keys: {legacy}; use packed W13 ABI"
        )

    # Known ABI keys are fail-closed. Unknown auxiliary keys remain allowed
    # for compatibility (for example standalone KV buffers), but a known key
    # can never be silently reinterpreted by the importer.
    known_keys = _FP32_KEYS | _INT8_KEYS
    for key in sorted(known_keys & set(entries)):
        if not native_w8a8 and key in _ROUTED_FP32_SCALE_KEYS:
            raise WeightMapInvalid(
                f"non-native routed map must not contain scale key {key!r}"
            )
        want = _dtype_for(key, native_w8a8=native_w8a8)
        got = entries[key]["dtype"]
        if got != want:
            detail = " (BF16-dequant routed is forbidden)" if (
                native_w8a8 and key in _INT8_KEYS
            ) else ""
            raise WeightMapInvalid(
                f"canonical dtype for {key!r} is {want!r}, got {got!r}{detail}"
            )
        want_layout = _layout_for(key, native_w8a8=native_w8a8)
        got_layout = entries[key]["layout"]
        if got_layout != want_layout:
            raise WeightMapInvalid(
                f"canonical layout for {key!r} is {want_layout!r}, "
                f"got {got_layout!r}",
            )

    if native_w8a8:
        required_routed = set(_ROUTED_INT8_KEYS) | set(_ROUTED_FP32_SCALE_KEYS)
        missing_routed = sorted(required_routed - set(entries))
        if missing_routed:
            raise WeightMapInvalid(
                f"native-W8A8 map missing packed routed keys: {missing_routed}"
            )

    if expected is not None:
        missing = [k for k in expected if k not in entries]
        if missing:
            raise WeightMapInvalid(f"map missing expected keys: {sorted(missing)}")
        if not allow_extra_keys:
            extra = [k for k in entries if k not in expected]
            if extra:
                raise WeightMapInvalid(f"map has unexpected keys: {sorted(extra)}")
        for k, spec in expected.items():
            exp_shape, exp_dtype = spec
            got_shape = tuple(int(s) for s in entries[k]["shape"])
            if got_shape != tuple(int(s) for s in exp_shape):
                raise WeightMapInvalid(
                    f"{k!r} shape {list(got_shape)} != expected {list(exp_shape)}"
                )
            if entries[k]["dtype"] != exp_dtype:
                raise WeightMapInvalid(
                    f"{k!r} dtype {entries[k]['dtype']!r} != expected {exp_dtype!r}"
                )


# =============================================================================
# Exporter (vLLM-side): consolidate this rank's PyPTO bundle -> ONE pool + key.
# =============================================================================
class WeightIpcExporter:
    """Consolidate a per-rank PyPTO weight bundle into ONE IPC-exportable pool.

    Mirrors ``_stage_kvpool_pageattn.exporter``: aclrtMalloc one buffer, H2D-copy
    each (already-transformed) weight in at its aligned offset, call
    aclrtIpcMemGetExportKey ONCE, write key + map JSON. Keeps the export handle
    alive (call ``teardown`` to aclrtIpcMemClose, mirroring _KvExporter).
    """

    def __init__(self, dev: int) -> None:
        self._dev = dev
        self._acl = ctypes.CDLL("libascendcl.so")
        self._acl.aclrtIpcMemGetExportKey.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64,
        ]
        # aclrtIpcMemClose closes the opaque key, not the device pointer.
        # Keep the exact export key alive until the owner is torn down.
        self._acl.aclrtIpcMemClose.argtypes = [ctypes.c_char_p]
        self._acl.aclrtMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int,
        ]
        self._acl.aclrtMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_int,
        ]
        self._acl.aclrtFree.argtypes = [ctypes.c_void_p]
        self._pool_ptr: Optional[int] = None
        self._pool_bytes: int = 0
        self._initialized = False
        self._ipc_owner = None
        # (dptr, nbytes) of the export handle, for teardown (mirror _KvExporter).
        self._export_key: Optional[ctypes.Array[ctypes.c_char]] = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._acl.aclInit(None)
        self._acl.aclrtSetDevice(self._dev)
        self._initialized = True

    @staticmethod
    def plan_layout(
        bundle: Dict[str, Any],
        *,
        native_w8a8: Optional[bool] = None,
    ) -> List[Tuple[str, int, Tuple[int, ...], str, int]]:
        """Lay out a bundle while enforcing its canonical dtype contract.

        ``native_w8a8`` should be supplied by the checkpoint/live loader. If
        omitted, the variant is inferred from packed routed scales or INT8
        routed tensors. The actual tensor dtype is checked against the ABI; it
        is never allowed to overwrite the dtype recorded in the map.
        """
        if not isinstance(bundle, dict) or not bundle:
            raise ValueError("weight bundle must be a non-empty dict")
        if native_w8a8 is None:
            native_w8a8 = _infer_native_w8a8(bundle)

        legacy = sorted(_LEGACY_ROUTED_KEYS & set(bundle))
        if legacy:
            raise ValueError(
                f"weight bundle contains removed routed keys: {legacy}; "
                "use packed moe_w13_r/moe_w_down_r ABI"
            )
        if native_w8a8:
            required = set(_ROUTED_INT8_KEYS) | set(_ROUTED_FP32_SCALE_KEYS)
            missing = sorted(required - set(bundle))
            if missing:
                raise ValueError(
                    f"native-W8A8 bundle missing packed routed keys: {missing}"
                )
        else:
            scales = sorted(set(_ROUTED_FP32_SCALE_KEYS) & set(bundle))
            if scales:
                raise ValueError(
                    f"non-native routed bundle must not contain scale keys: {scales}"
                )

        layout: List[Tuple[str, int, Tuple[int, ...], str, int]] = []
        offset = 0
        for key in sorted(bundle):
            t = bundle[key]
            shape = tuple(int(s) for s in t.shape)
            if not shape or any(dim <= 0 for dim in shape):
                raise ValueError(f"weight {key!r} has invalid shape {shape!r}")
            actual = _tensor_dtype_name(t)
            expected = _dtype_for(key, native_w8a8=native_w8a8)
            layout_name = _layout_for(key, native_w8a8=native_w8a8)
            _validate_layout_shape(key, shape, layout_name)
            if actual != expected:
                raise ValueError(
                    f"canonical dtype mismatch for {key!r}: got {actual!r}, "
                    f"want {expected!r}"
                )
            nbytes = int(t.numel() * t.element_size())
            want_nbytes = int(t.numel()) * _DTYPE_ITEMSIZE[expected]
            if nbytes != want_nbytes:
                raise ValueError(
                    f"byte-size mismatch for {key!r}: got {nbytes}, "
                    f"want {want_nbytes} for dtype {expected!r}"
                )
            layout.append((key, offset, shape, expected, nbytes))
            offset = _align_up(offset + nbytes)
        return layout

    def export(
        self,
        bundle: Dict[str, Any],
        *,
        out_dir: str,
        rank: int,
        tp_world_size: int = 8,
        max_pool_bytes: Optional[int] = None,
        native_w8a8: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Consolidate ``bundle`` into one pool, emit key + map, return summary.

        ``max_pool_bytes``: if set and the bundle exceeds it, raise (splitting
        into N pools is a TODO behind OPEN_DEVICE_QUESTION 3; for now one pool).
        """
        self._ensure_init()
        if self._ipc_owner is None:
            from tools.step3p5.ipc_session import maybe_start_owner  # noqa: PLC0415

            self._ipc_owner = maybe_start_owner(
                out_dir,
                role="weight",
                rank=rank,
                device_id=(
                    int(os.environ["PYPTO_IPC_DEVICE_OFFSET"]) + rank
                    if "PYPTO_IPC_DEVICE_OFFSET" in os.environ
                    else self._dev
                ),
            )
        if native_w8a8 is None:
            native_w8a8 = _infer_native_w8a8(bundle)
        layout = self.plan_layout(bundle, native_w8a8=native_w8a8)
        if not layout:
            raise RuntimeError("weight bundle is empty; nothing to export")
        pool_bytes = _align_up(layout[-1][1] + layout[-1][4])
        if max_pool_bytes is not None and pool_bytes > max_pool_bytes:
            raise RuntimeError(
                f"weight pool {pool_bytes/1e9:.2f} GB exceeds max_pool_bytes "
                f"{max_pool_bytes/1e9:.2f} GB; multi-pool split not implemented "
                f"(see OPEN_DEVICE_QUESTION 3)",
            )

        # Allocate ONE contiguous device buffer for the whole rank bundle.
        # Blocker-B experiment (env PYPTO_WEIGHT_IPC_VA_SHIFT_GB): pre-allocate a
        # throwaway block first so the exported pool VA lands ABOVE the runtime's
        # comm-window region on peers, avoiding the MoE ep_all_to_all x IPC-pool
        # peer-access VA overlap that stalls the whole-net e2e.  Kept alive (never
        # freed) so the low VA stays occupied.  Default 0 = disabled (no shift).
        import os as _os  # noqa: PLC0415
        _shift_gb = float(_os.environ.get("PYPTO_WEIGHT_IPC_VA_SHIFT_GB", "0"))
        if _shift_gb > 0:
            _shift_bytes = int(_shift_gb * (1 << 30))
            _sp = ctypes.c_void_p()
            _rc = self._acl.aclrtMalloc(ctypes.byref(_sp), _shift_bytes, _HUGE_FIRST)
            if _rc != 0:
                raise RuntimeError(f"VA-shift aclrtMalloc rc={_rc} bytes={_shift_bytes}")
            self._va_shift_ptr = _sp  # keep alive
            print(f"[weight-ipc exporter] VA-shift {_shift_gb} GiB pre-alloc @ "
                  f"0x{int(_sp.value or 0):x}", flush=True)
        dptr = ctypes.c_void_p()
        rc = self._acl.aclrtMalloc(ctypes.byref(dptr), pool_bytes, _HUGE_FIRST)
        if rc != 0:
            raise RuntimeError(f"aclrtMalloc rc={rc} nbytes={pool_bytes}")
        self._pool_ptr = int(dptr.value or 0)
        self._pool_bytes = pool_bytes

        # H2D-copy each weight at its aligned offset. bundle tensors are host
        # tensors (from weight_loader); we use their .contiguous() data_ptr.
        for key, offset, shape, dtype_name, nbytes in layout:
            t = bundle[key]
            if t.numel() * t.element_size() != nbytes:
                raise RuntimeError(
                    f"size mismatch key={key} plan={nbytes} actual={t.numel()*t.element_size()}",
                )
            host = ctypes.c_void_p(int(t.contiguous().data_ptr()))
            dst = ctypes.c_void_p(self._pool_ptr + offset)
            rc = self._acl.aclrtMemcpy(dst, nbytes, host, nbytes, _H2D)
            if rc != 0:
                raise RuntimeError(f"aclrtMemcpy rc={rc} key={key} nbytes={nbytes}")

        # ONE export key for the whole pool (the KV-pool pattern).
        key_buf = ctypes.create_string_buffer(KEY_BUF)
        rc = self._acl.aclrtIpcMemGetExportKey(
            ctypes.c_void_p(self._pool_ptr), ctypes.c_size_t(pool_bytes),
            key_buf, KEY_BUF, ctypes.c_uint64(0x1),
        )
        if rc != 0:
            raise RuntimeError(
                f"aclrtIpcMemGetExportKey rc={rc} pool={hex(self._pool_ptr)} "
                f"nbytes={pool_bytes}",
            )
        self._export_key = key_buf

        os.makedirs(out_dir, exist_ok=True)
        key_path = os.path.join(out_dir, f"pypto_weight.key.rank{rank}")
        map_path = os.path.join(out_dir, f"pypto_weight_map.rank{rank}.json")
        # The key is an opaque ACL capability.  Publish it before the map, and
        # publish the ready manifest only after both files are complete.
        key_tmp = key_path + f".tmp.{os.getpid()}"
        with open(key_tmp, "wb") as f:
            f.write(key_buf.raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(key_tmp, key_path)
        # Sentinel so the importer can poll-and-wait (mirror _stage_kvpool_pageattn).
        map_obj = {
            "version": 1,
            "rank": rank,
            "tp_world_size": tp_world_size,
            "pool_bytes": pool_bytes,
            "pool_dtype_bytes": 2,
            "map": {
                key: {
                    "offset": offset,
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "layout": _layout_for(
                        key, native_w8a8=bool(native_w8a8),
                    ),
                    "nbytes": nbytes,
                }
                for key, offset, shape, dtype_name, nbytes in layout
            },
        }
        from tools.step3p5.ipc_session import (  # noqa: PLC0415
            atomic_write_json,
            attach_session,
            write_ready_manifest,
        )

        map_obj = attach_session(map_obj, self._ipc_owner)
        atomic_write_json(map_path, map_obj)
        write_ready_manifest(
            map_path + ".done",
            map_path=map_path,
            key_path=key_path,
            owner=self._ipc_owner,
        )
        print(
            f"[weight-ipc exporter] rank={rank} pool_GiB={pool_bytes/2**30:.2f} "
            f"keys={len(layout)} ONE_KEY pool_base={hex(self._pool_ptr)} "
            f"-> {key_path}",
            flush=True,
        )
        return {
            "ok": True,
            "rank": rank,
            "pool_bytes": pool_bytes,
            "num_keys": len(layout),
            "key_path": key_path,
            "map_path": map_path,
            "pool_base_debug": self._pool_ptr,
        }

    def teardown(self) -> int:
        """Close the IPC export handle + free the pool (mirror _KvExporter.teardown)."""
        closed = 0
        if self._export_key is not None:
            try:
                if self._acl.aclrtIpcMemClose(self._export_key) == 0:
                    closed += 1
            except Exception:  # noqa: BLE001
                pass
            self._export_key = None
        if self._pool_ptr is not None:
            try:
                self._acl.aclrtFree(ctypes.c_void_p(self._pool_ptr))
            except Exception:  # noqa: BLE001
                pass
            self._pool_ptr = None
        if self._ipc_owner is not None:
            try:
                self._ipc_owner.close()
            finally:
                self._ipc_owner = None
        return closed


def _prepare_checkpoint_bundle(
    ckpt_dir: str,
    *,
    rank: int,
    tp_world_size: int = 8,
    int8_routed: bool = False,
    kv_ipc: bool = False,
    production_hidden_only: bool = False,
) -> Dict[str, Any]:
    """Load and normalize one rank's checkpoint bundle for the PyPTO ABI."""
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        load_step3p5_weights_for_rank,
        verify_bundle_shapes,
    )
    import torch  # noqa: PLC0415

    bundle = load_step3p5_weights_for_rank(
        ckpt_dir,
        rank,
        tp_world_size,
        int8_routed=int8_routed,
        decode_native_moe=production_hidden_only,
    )
    verify_bundle_shapes(
        bundle,
        tp_world_size,
        decode_native_moe=production_hidden_only,
        int8_routed=int8_routed,
    )
    # The whole_decode host_orch expects FP32 for the norm weights + final_norm
    # (matching the dummy device harness), but weight_loader stores norms as bf16.
    # Zero-copy IPC cannot cast at read time, so materialize FP32 bytes here so the
    # exported pool + map dtype are FP32 (router weights are already FP32).
    for key in _PROGRAM_FP32_KEYS:
        if key in bundle and str(bundle[key].dtype) != "torch.float32":
            bundle[key] = bundle[key].to(torch.float32)
    if production_hidden_only:
        # Production ownership:
        # - vLLM owns target final norm/LM head;
        # - vLLM owns every MTP shared-head norm/LM head;
        # - PyPTO still owns MTP token embedding in the selected-layer body,
        #   so KEY_EMBED must remain in this combined Main+MTP body pool.
        from models.step3p5 import weight_loader as keys  # noqa: PLC0415

        forbidden_tail = (
            keys.KEY_FINAL_NORM,
            keys.KEY_LM_HEAD,
            keys.KEY_MTP_SH_NORM,
            keys.KEY_MTP_SH_OUT,
        )
        for key in forbidden_tail:
            bundle.pop(key, None)
    fp32_keys = _FP32_KEYS & set(bundle)
    routed_keys = _INT8_KEYS & set(bundle)
    for key, tensor in bundle.items():
        if key in fp32_keys:
            expected_dtype = "float32"
        elif key in routed_keys and int8_routed:
            expected_dtype = "int8"
        else:
            expected_dtype = "bfloat16"
        actual_dtype = str(tensor.dtype).removeprefix("torch.")
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"bundle dtype mismatch for {key!r}: got {actual_dtype!r}, "
                f"want {expected_dtype!r}"
            )
    if kv_ipc:
        # Standalone validation-only KV. Live serving exports vLLM's allocator
        # through vllm_kvpool_backend instead and must keep this disabled.
        import models.step3p5.config as cfg  # noqa: PLC0415

        cache_rows, head_dim = int(cfg.KV_CACHE_ROWS_DYN), int(cfg.HEAD_DIM)
        bundle["k_cache"] = torch.zeros(
            [cache_rows, head_dim], dtype=torch.bfloat16
        )
        bundle["v_cache"] = torch.zeros(
            [cache_rows, head_dim], dtype=torch.bfloat16
        )
        # MTP layers 45/46/47 own distinct KV slices. Keep them in the same
        # one-key IPC pool but never alias them with the main-network cache.
        num_mtp = int(cfg.NUM_NEXTN_PREDICT_LAYERS)
        bundle["mtp_k_cache"] = torch.zeros(
            [num_mtp * cache_rows, head_dim], dtype=torch.bfloat16
        )
        bundle["mtp_v_cache"] = torch.zeros(
            [num_mtp * cache_rows, head_dim], dtype=torch.bfloat16
        )
    return bundle


def export_from_checkpoint_resident(
    ckpt_dir: str,
    *,
    rank: int,
    tp_world_size: int = 8,
    out_dir: str,
    dev: int = 0,
    int8_routed: bool = False,
    kv_ipc: bool = False,
    production_hidden_only: bool = False,
) -> Tuple[WeightIpcExporter, Dict[str, Any], Dict[str, Any]]:
    """Export one rank and return the owner object plus the host bundle.

    The caller must retain the returned ``WeightIpcExporter`` for the whole
    serving lifetime.  This is the model-loader-owned path: the vLLM worker
    process owns the native-W8A8 PyPTO allocation and its IPC export handle,
    eliminating the separate checkpoint-exporter process.
    """
    bundle = _prepare_checkpoint_bundle(
        ckpt_dir,
        rank=rank,
        tp_world_size=tp_world_size,
        int8_routed=int8_routed,
        kv_ipc=kv_ipc,
        production_hidden_only=production_hidden_only,
    )
    exporter = WeightIpcExporter(dev)
    summary = exporter.export(
        bundle,
        out_dir=out_dir,
        rank=rank,
        tp_world_size=tp_world_size,
        native_w8a8=int8_routed,
    )
    return exporter, summary, bundle


def export_mtp_hidden_weights_from_checkpoint(
    ckpt_dir: str,
    *,
    rank: int,
    tp_world_size: int = 8,
    out_dir: str,
    dev: int = 0,
) -> Tuple[WeightIpcExporter, Dict[str, Any], Dict[str, Any]]:
    """Export only the selected MTP hidden-body bundle.

    This is the standalone selected-layer bring-up path.  The production
    Main+MTP service may use one model-owned pool, but an offline MTP device
    run must not accidentally depend on Main KV, Main tail, or MTP shared-head
    buffers.  The returned pool therefore contains exactly the embedding and
    selected MTP transformer-body weights consumed by ``MtpLayerHolder``.

    Only the checkpoint embedding and MTP45/46/47 tensors are read.  Main
    decoder/MoE weights are never loaded, so this path cannot dequantize or
    otherwise alter the native-W8A8 Main ownership domain.
    """
    import torch  # noqa: PLC0415
    from models.step3p5 import weight_loader as keys  # noqa: PLC0415
    from models.step3p5.config import (  # noqa: PLC0415
        HIDDEN,
        INTERMEDIATE,
        NUM_HEADS_SWA,
        NUM_HEADS_SWA_LOCAL_PAD,
        NUM_HIDDEN_LAYERS,
        NUM_KV_HEADS,
        NUM_NEXTN_PREDICT_LAYERS,
        VOCAB,
    )
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        _ShardCache,
        _hf_mtp_keys,
        _read_index,
        _slice_eh_proj,
        _slice_g_proj,
        _slice_kv_proj,
        _slice_mlp_col,
        _slice_mlp_row,
        _slice_o_proj,
        _slice_q_proj,
        _to_bf16,
    )

    if not 0 <= int(rank) < int(tp_world_size):
        raise ValueError(
            f"rank {rank} is outside tp_world_size={tp_world_size}"
        )
    if int(tp_world_size) != 8:
        raise ValueError("selected Step3p5 MTP exporter requires TP=8")
    num_heads_local = int(NUM_HEADS_SWA) // int(tp_world_size)
    kv_heads_local = int(NUM_KV_HEADS) // int(tp_world_size)
    intermediate_local = int(INTERMEDIATE) // int(tp_world_size)
    hidden_local = int(HIDDEN) // int(tp_world_size)

    selected: Dict[str, Any] = {}
    weight_map = _read_index(ckpt_dir)
    with _ShardCache(ckpt_dir, weight_map) as cache:
        selected[keys.KEY_EMBED] = _to_bf16(
            cache.get("model.embed_tokens.weight").contiguous()
        )
        rows: dict[str, list[Any]] = {
            keys.KEY_MTP_ENORM: [],
            keys.KEY_MTP_HNORM: [],
            keys.KEY_MTP_EH_PROJ: [],
            keys.KEY_MTP_INPUT_RMS: [],
            keys.KEY_MTP_POST_ATTN_RMS: [],
            keys.KEY_MTP_Q_NORM: [],
            keys.KEY_MTP_K_NORM: [],
            keys.KEY_MTP_WQ: [],
            keys.KEY_MTP_WK: [],
            keys.KEY_MTP_WV: [],
            keys.KEY_MTP_WO: [],
            keys.KEY_MTP_WG: [],
            keys.KEY_MTP_DENSE_GATE: [],
            keys.KEY_MTP_DENSE_UP: [],
            keys.KEY_MTP_DENSE_DOWN: [],
        }
        for local_idx in range(int(NUM_NEXTN_PREDICT_LAYERS)):
            mtp = _hf_mtp_keys(int(NUM_HIDDEN_LAYERS) + local_idx)
            rows[keys.KEY_MTP_ENORM].append(
                cache.get(mtp["enorm"]).to(torch.float32)
            )
            rows[keys.KEY_MTP_HNORM].append(
                cache.get(mtp["hnorm"]).to(torch.float32)
            )
            rows[keys.KEY_MTP_EH_PROJ].append(
                _slice_eh_proj(
                    cache.get(mtp["eh_proj"]),
                    int(rank),
                    hidden_local,
                )
            )
            rows[keys.KEY_MTP_INPUT_RMS].append(
                cache.get(mtp["input_rms"]).to(torch.float32)
            )
            rows[keys.KEY_MTP_POST_ATTN_RMS].append(
                cache.get(mtp["post_attn_rms"]).to(torch.float32)
            )
            rows[keys.KEY_MTP_Q_NORM].append(
                cache.get(mtp["q_norm"]).to(torch.float32)
            )
            rows[keys.KEY_MTP_K_NORM].append(
                cache.get(mtp["k_norm"]).to(torch.float32)
            )
            rows[keys.KEY_MTP_WQ].append(
                _slice_q_proj(
                    cache.get(mtp["q_proj"]),
                    int(rank),
                    num_heads_local,
                )
            )
            rows[keys.KEY_MTP_WK].append(
                _slice_kv_proj(
                    cache.get(mtp["k_proj"]),
                    int(rank),
                    kv_heads_local,
                )
            )
            rows[keys.KEY_MTP_WV].append(
                _slice_kv_proj(
                    cache.get(mtp["v_proj"]),
                    int(rank),
                    kv_heads_local,
                )
            )
            rows[keys.KEY_MTP_WO].append(
                _slice_o_proj(
                    cache.get(mtp["o_proj"]),
                    int(rank),
                    num_heads_local,
                )
            )
            rows[keys.KEY_MTP_WG].append(
                _slice_g_proj(
                    cache.get(mtp["g_proj"]),
                    int(rank),
                    num_heads_local,
                    pad_to=int(NUM_HEADS_SWA_LOCAL_PAD),
                )
            )
            rows[keys.KEY_MTP_DENSE_GATE].append(
                _slice_mlp_col(
                    cache.get(mtp["gate_proj"]),
                    int(rank),
                    intermediate_local,
                )
            )
            rows[keys.KEY_MTP_DENSE_UP].append(
                _slice_mlp_col(
                    cache.get(mtp["up_proj"]),
                    int(rank),
                    intermediate_local,
                )
            )
            rows[keys.KEY_MTP_DENSE_DOWN].append(
                _slice_mlp_row(
                    cache.get(mtp["down_proj"]),
                    int(rank),
                    intermediate_local,
                )
            )
        for key, values in rows.items():
            selected[key] = torch.stack(values, dim=0).contiguous()

    if tuple(selected[keys.KEY_EMBED].shape) != (int(VOCAB), int(HIDDEN)):
        raise RuntimeError("selected MTP embedding shape is invalid")
    forbidden_tail = {
        keys.KEY_FINAL_NORM,
        keys.KEY_LM_HEAD,
        keys.KEY_MTP_SH_NORM,
        keys.KEY_MTP_SH_OUT,
    }
    if forbidden_tail & set(selected):
        raise RuntimeError("selected MTP weight pool contains a vLLM-owned tail")
    exporter = WeightIpcExporter(dev)
    summary = exporter.export(
        selected,
        out_dir=out_dir,
        rank=rank,
        tp_world_size=tp_world_size,
        native_w8a8=False,
    )
    return exporter, summary, selected


def export_from_checkpoint(
    ckpt_dir: str,
    *,
    rank: int,
    tp_world_size: int = 8,
    out_dir: str,
    dev: int = 0,
    int8_routed: bool = False,
    kv_ipc: bool = False,
    production_hidden_only: bool = False,
) -> Dict[str, Any]:
    """Convenience: load a rank bundle from a checkpoint + export it.

    Uses ``weight_loader.load_step3p5_weights_for_rank`` which already applies
    every vLLM->PyPTO layout transform (qkv split/transpose, head-pad, fp32
    promotion for gate/router_bias). With ``int8_routed=True`` the routed MoE
    tensors remain native INT8 plus FP32 scales; this function does not
    dequantize them. The result is the exact per-rank bundle
    ``Step3p5DecodeFwd`` consumes, so the worker can bind each key straight to
    a kernel arg.

    For LIVE serving the bundle would instead be assembled directly from vLLM's
    resident ``nn.Parameter`` tensors using the same transform plan
    (``weight_translate.build_vllm_to_pypto_transform_plan``); that path is
    OPEN_DEVICE_QUESTION (4) below — it must not copy weights to host but
    transform device-side into the pool. The checkpoint path here is the
    numerically-identical reference used by the card-free validation harness.
    """
    exporter, summary, _bundle = export_from_checkpoint_resident(
        ckpt_dir,
        rank=rank,
        tp_world_size=tp_world_size,
        out_dir=out_dir,
        dev=dev,
        int8_routed=int8_routed,
        kv_ipc=kv_ipc,
        production_hidden_only=production_hidden_only,
    )
    # Historical standalone callers intentionally keep the raw allocation
    # alive until process exit.  No __del__ tears it down.
    del exporter, _bundle
    return summary


# =============================================================================
# Importer (pypto worker-side): ONE import_ipc -> per-key DeviceTensor.
# =============================================================================
class WeightIpcMap:
    """Worker-side weight pool: ONE imported peer_base + per-key DeviceTensors.

    Mirrors ``_stage_kvpool_pageattn.KVPoolMap`` and the one-key path in
    ``_stage_attn_worker.py::attn_setup`` (lines 933-958): import the pool
    ONCE via ``rt.import_ipc(key)``, then address each weight by its byte
    offset from the map JSON as a zero-copy ``DeviceTensor``.
    """

    def __init__(
        self,
        peer_base: int,
        pool_map: Dict[str, Any],
        *,
        runtime: Any = None,
        worker_id: int = 0,
    ) -> None:
        self.peer_base = int(peer_base)
        self.pool_map = pool_map
        self._map = pool_map["map"]
        self._runtime = runtime
        self._worker_id = int(worker_id)

    def _imported_tensor(self, offset: int, shape, dtype):
        """Wrap one pool slice as a DeviceTensor that public dispatch accepts.

        Dispatch derives its wire descriptor from an owner ``Buffer`` whose base is
        the argument's own address, so an interior view of the imported pool needs a
        Buffer of its own; ``imported_tensor`` mints and registers it. Without a
        runtime (offline map inspection) this falls back to a raw-pointer handle,
        which is rejected at dispatch rather than silently mis-dispatched.
        """
        from pypto.runtime.device_tensor import DeviceTensor  # noqa: PLC0415

        ptr = self.peer_base + int(offset)
        if self._runtime is None:
            return DeviceTensor(ptr, shape, dtype)
        return self._runtime.imported_tensor(ptr, shape, dtype, worker_id=self._worker_id)

    @classmethod
    def from_files(
        cls, key_path: str, map_path: str, *, rt, worker_id: int = 0,
        native_w8a8: bool = True,
    ) -> "WeightIpcMap":
        """Read key + map, import the pool into the worker's address space.

        ``rt`` is a ``DistributedWorker`` (or anything exposing
        ``import_ipc(key, worker_id=0) -> int``). Returns a WeightIpcMap bound
        to the imported peer_base.
        """
        with open(map_path) as f:
            pool_map = json.load(f)
        validate_weight_map(pool_map, native_w8a8=native_w8a8)
        with open(key_path, "rb") as f:
            key = f.read()
        peer_base = int(rt.import_ipc(key, worker_id=worker_id))
        print(
            f"[weight-ipc importer] ONE_KEY pool import peer_base={hex(peer_base)} "
            f"keys={len(pool_map['map'])} pool_GiB={pool_map['pool_bytes']/2**30:.2f}",
            flush=True,
        )
        return cls(peer_base, pool_map, runtime=rt, worker_id=worker_id)

    def device_tensor(self, key: str):
        """Build the zero-copy DeviceTensor for one bundle key.

        Returns a ``pypto.runtime.device_tensor.DeviceTensor`` backed by
        ``peer_base + offset`` with the key's shape/dtype — pass it directly
        as the kernel arg in place of a torch.Tensor (no H2D/D2H).
        """
        if key not in self._map:
            raise KeyError(
                f"weight-ipc map missing key={key!r} "
                f"(have {sorted(self._map)[:8]}...)",
            )
        entry = self._map[key]
        offset = int(entry["offset"])
        shape = tuple(int(s) for s in entry["shape"])
        dtype = _torch_dtype(entry["dtype"])
        return self._imported_tensor(offset, shape, dtype)

    def device_tensor_slice(self, key: str, start: int, stop: int):
        """Leading-dim sub-view ``[start:stop]`` of one bundle key, zero-copy.

        ``DeviceTensor.__getitem__`` gives the right address and shape but no owner
        Buffer, and a sub-view's address differs from the pool base, so it cannot
        borrow the parent's. Re-mint provenance here rather than at the call site,
        so callers never have to know the dispatch guard exists.
        """
        sub = self.device_tensor(key)[int(start):int(stop)]
        return self._imported_tensor(sub.data_ptr - self.peer_base, sub.shape, sub.dtype)

    def bundle(self) -> Dict[str, Any]:
        """Build the full {key: DeviceTensor} dict for the whole-decode program.

        Convenience for ``Step3p5DecodeFwd``: pass this as the weight args.
        Keys not needed by a given program variant are simply ignored by the
        caller; building the full dict is cheap (just DeviceTensor handles).
        """
        return {key: self.device_tensor(key) for key in self._map}

    def offset(self, key: str) -> int:
        return int(self._map[key]["offset"])


# =============================================================================
# G5b import_ipc_all path (N=1 whole-decode weight residency, mirrors
# pypto_kv_ipc.build_stacked_kv). The orphaned WeightIpcMap.from_files uses
# rt.import_ipc (missing C++ facade); the WORKING path is the pure-Python batch
# import_ipc_all (distributed_runner.py:1086) + direct WeightIpcMap(peer_base=va).
# =============================================================================
def import_weights_all(
    rt, out_dir: str, *, tp: int, dev_offset: int = 0,
    native_w8a8: bool = True,
) -> List["WeightIpcMap"]:
    """Batch-import all ``tp`` per-rank weight pools via ``DistributedWorker.import_ipc_all``.

    Reads ``pypto_weight.key.rank{r}`` + ``pypto_weight_map.rank{r}.json`` from
    ``out_dir`` (written by ``WeightIpcExporter.export``), imports every rank's pool
    ONCE into the resident worker's chip children (device ``dev_offset + r``), and
    returns per-rank ``WeightIpcMap`` (peer_base = imported VA). Mirrors the KV path
    in ``_stage_whole_decode_run.py`` (device_key_map -> import_ipc_all -> per-rank Map).
    """
    device_key_map: Dict[int, bytes] = {}
    maps_json: List[Dict[str, Any]] = []
    from tools.step3p5.ipc_session import (  # noqa: PLC0415
        validate_key_file,
        validate_live_session,
    )

    for r in range(tp):
        key_path = os.path.join(out_dir, f"pypto_weight.key.rank{r}")
        map_path = os.path.join(out_dir, f"pypto_weight_map.rank{r}.json")
        ready_path = map_path + ".done"
        key = validate_key_file(key_path)
        with open(map_path) as f:
            pool_map = json.load(f)
        validate_weight_map(pool_map, native_w8a8=native_w8a8)
        validate_live_session(
            pool_map,
            expected_rank=r,
            expected_tp=tp,
            expected_device_id=dev_offset + r,
            expected_role="weight",
            ready_path=ready_path,
            map_path=map_path,
            key_path=key_path,
        )
        device_key_map[dev_offset + r] = key
        maps_json.append(pool_map)
    # Pass each rank's consolidated pool size so simpler registers the imported
    # region as a live child allocation; without it the dispatch provenance guard
    # rejects the interior DeviceTensor(peer_base + offset) weight pointers.
    region_bytes: Dict[int, int] = {
        dev_offset + r: int(maps_json[r]["pool_bytes"]) for r in range(tp)
    }
    vas = rt.import_ipc_all(device_key_map, region_bytes=region_bytes)  # {device_id: peer VA}
    print(
        "[weight-ipc importer] import_ipc_all peer_bases="
        + str([hex(vas[dev_offset + r]) for r in range(tp)]),
        flush=True,
    )
    return [
        WeightIpcMap(vas[dev_offset + r], maps_json[r], runtime=rt, worker_id=r)
        for r in range(tp)
    ]


def build_stacked_weight(weight_maps: List["WeightIpcMap"], key: str):
    """Build a ``StackedDeviceTensor`` for one host_orch weight param across ranks.

    ``weight_maps[r]`` imported for chip ``r`` (its DeviceTensors resident on chip r).
    Returns a StackedDeviceTensor whose leading dim == tp, matching the whole-decode
    host_orch ``[tp, ...]`` weight signature (host_orch slices ``[r]`` per rank),
    zero-copy. Mirrors ``pypto_kv_ipc.build_stacked_kv``.
    """
    from pypto.runtime.device_tensor import StackedDeviceTensor  # noqa: PLC0415
    tp = len(weight_maps)
    shards = [weight_maps[r].device_tensor(key) for r in range(tp)]
    full = (tp, *tuple(shards[0].shape))
    return StackedDeviceTensor(shards, full, list(range(tp)))


# =============================================================================
# OPEN_DEVICE_QUESTION (4): LIVE vLLM-resident weight path (design only).
# =============================================================================
# The checkpoint convenience path above H2D-copies host tensors into the pool.
# For LIVE serving we must NOT round-trip vLLM's resident device weights
# through host. The live exporter should:
#   1. iterate vLLM's sharded nn.Parameters (per the transform plan in
#      weight_translate.build_vllm_to_pypto_transform_plan);
#   2. for each weight, allocate its slot in the pool (DeviceTensor view) and
#      run the transform (transpose / qkv-split / native-W8A8 scale wiring /
#      head-pad)
#      DEVICE-SIDE into that slot (a tiny pypto kernel or aclrtMemcpy D2D for
#      the pure-transpose cases + native INT8/FP32-scale mapping for routed
#      experts);
#   3. then aclrtIpcMemGetExportKey the whole pool ONCE.
# This avoids both the host round-trip AND duplicating weights in a second
# process. The transform kernels follow the same layout rules as the offline
# weight_loader; routed experts must remain INT8 + FP32 scale and MTP matrices
# remain BF16. NOT implemented here — the map schema + importer are identical
# regardless of how the pool was filled.


# =============================================================================
# Smoke self-check (card-free): validate layout planning + map round-trip
# against the canonical bundle contract WITHOUT a device.
# =============================================================================
def _smoke_layout() -> int:
    """Plan a layout from expected_shapes + verify offsets/alignment/dtypes.

    No device, no ACL, no checkpoint load — pure dict math. Confirms:
      - every bundle key gets an offset
      - offsets are 512-aligned and non-overlapping
      - dtypes match the FP32/BF16/INT8 rule
      - total pool_bytes is sane
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        build_compact_shape_table,
        build_synthetic_bundle,
    )

    shapes = build_compact_shape_table(8, int8_routed=True)
    bundle = build_synthetic_bundle(
        rank=0,
        tp_world_size=8,
        shape_overrides=shapes,
        int8_routed=True,
    )
    import torch  # noqa: PLC0415

    for key in _PROGRAM_FP32_KEYS:
        if key in bundle:
            bundle[key] = bundle[key].to(torch.float32)
    layout = WeightIpcExporter.plan_layout(bundle, native_w8a8=True)
    keys = [k for k, _, _, _, _ in layout]
    assert len(keys) == len(shapes), f"key count mismatch {len(keys)} vs {len(shapes)}"
    # Non-overlapping + aligned.
    end = 0
    for key, offset, shape, dtype_name, nbytes in layout:
        assert offset % _ALIGN == 0, f"{key} offset {offset} not aligned"
        assert offset >= end, f"{key} offset {offset} overlaps prev end {end}"
        assert dtype_name == _dtype_for(key), f"{key} dtype {dtype_name} != {_dtype_for(key)}"
        end = offset + nbytes
    pool_bytes = _align_up(end)
    print(
        f"[weight-ipc smoke] keys={len(keys)} pool_GiB={pool_bytes/2**30:.2f} "
        f"fp32_keys={[k for k in keys if k in _FP32_KEYS]} "
        f"int8_keys={[k for k in keys if k in _INT8_KEYS]}",
        flush=True,
    )
    # Round-trip the map JSON the importer would read.
    map_obj = {
        "version": 1, "rank": 0, "tp_world_size": 8,
        "pool_bytes": pool_bytes, "pool_dtype_bytes": 2,
        "map": {
            k: {
                "offset": o,
                "shape": list(s),
                "dtype": d,
                "layout": _layout_for(k, native_w8a8=True),
                "nbytes": n,
            }
            for k, o, s, d, n in layout
        },
    }
    json.loads(json.dumps(map_obj))  # serializable
    validate_weight_map(map_obj, native_w8a8=True)
    print("[weight-ipc smoke] layout + map round-trip OK", flush=True)
    return 0


def _main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "smoke":
        return _smoke_layout()
    role = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if role == "export-checkpoint":
        # export_from_checkpoint <ckpt> <rank> <out_dir> [dev]
        ckpt = sys.argv[2]
        rank = int(sys.argv[3])
        out_dir = sys.argv[4]
        dev = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        summary = export_from_checkpoint(
            ckpt, rank=rank, out_dir=out_dir, dev=dev)
        print(json.dumps(summary, indent=2))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
