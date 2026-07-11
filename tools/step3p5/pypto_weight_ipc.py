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
      ``weight_translate.build_vllm_to_pypto_transform_plan`` incl. W8A8
      dequant) into ONE contiguous device buffer, calls
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
      "pool_dtype_bytes": 2,          # bf16 (all weights are bf16/fp32; see note)
      "map": {
        "<bundle_key>": {             # e.g. "wq_full", "moe_w_gate_r", "lm_head_weight"
          "offset": <int>,            # byte offset within the pool
          "shape": [<int>, ...],      # PyPTO bundle shape (per-rank, post-transform)
          "dtype": "bfloat16"|"float32",
          "nbytes": <int>
        },
        ...
      }
    }

The ``map`` keys are exactly the ``models.step3p5.weight_loader.KEY_*``
constants so the worker can hand the whole-decode program the same names it
already expects. ``KEY_MOE_GATE_W`` and ``KEY_MOE_ROUTER_BIAS`` are fp32;
every other key is bf16 (see ``weight_loader._fp32_keys``).

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

# Bundle keys that are FP32 (everything else is BF16).
# Mirrors weight_loader.py fp32_keys = {KEY_MOE_GATE_W, KEY_MOE_ROUTER_BIAS}.
_FP32_KEYS = {"moe_gate_w", "moe_router_bias"}


def _dtype_for(key: str) -> str:
    return "float32" if key in _FP32_KEYS else "bfloat16"


def _torch_dtype(name: str):
    import torch  # noqa: PLC0415
    return torch.float32 if name == "float32" else torch.bfloat16


def _align_up(n: int, a: int = _ALIGN) -> int:
    return (n + a - 1) // a * a


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
        self._acl.aclrtIpcMemClose.argtypes = [ctypes.c_void_p]
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
        # (dptr, nbytes) of the export handle, for teardown (mirror _KvExporter).
        self._export_handle: Optional[Tuple[int, int]] = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._acl.aclInit(None)
        self._acl.aclrtSetDevice(self._dev)
        self._initialized = True

    @staticmethod
    def plan_layout(bundle: Dict[str, Any]) -> List[Tuple[str, int, Tuple[int, ...], str, int]]:
        """Lay out the bundle into the pool: aligned (offset, nbytes) per key.

        Returns a list of (key, offset, shape, dtype_name, nbytes) in pool order.
        Shape/dtype come straight from each bundle tensor (post-transform PyPTO
        layout). Order is deterministic (sorted by key) so exporter/importer
        agree without communicating it out-of-band beyond the map JSON.
        """
        layout: List[Tuple[str, int, Tuple[int, ...], str, int]] = []
        offset = 0
        for key in sorted(bundle):
            t = bundle[key]
            shape = tuple(int(s) for s in t.shape)
            dtype_name = _dtype_for(key)
            # Defensive: honor the tensor's own dtype if it disagrees with the
            # canonical rule (e.g. a checkpoint that ships gate_w in bf16).
            td = str(t.dtype).removeprefix("torch.")
            if td in ("float32", "bfloat16"):
                dtype_name = td
            nbytes = int(t.numel() * t.element_size())
            layout.append((key, offset, shape, dtype_name, nbytes))
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
    ) -> Dict[str, Any]:
        """Consolidate ``bundle`` into one pool, emit key + map, return summary.

        ``max_pool_bytes``: if set and the bundle exceeds it, raise (splitting
        into N pools is a TODO behind OPEN_DEVICE_QUESTION 3; for now one pool).
        """
        self._ensure_init()
        layout = self.plan_layout(bundle)
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
        self._export_handle = (self._pool_ptr, pool_bytes)

        os.makedirs(out_dir, exist_ok=True)
        key_path = os.path.join(out_dir, f"pypto_weight.key.rank{rank}")
        map_path = os.path.join(out_dir, f"pypto_weight_map.rank{rank}.json")
        with open(key_path, "wb") as f:
            f.write(key_buf.raw)
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
                    "nbytes": nbytes,
                }
                for key, offset, shape, dtype_name, nbytes in layout
            },
        }
        with open(map_path, "w") as f:
            json.dump(map_obj, f, indent=2)
        open(map_path + ".done", "w").write("1")
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
        }

    def teardown(self) -> int:
        """Close the IPC export handle + free the pool (mirror _KvExporter.teardown)."""
        closed = 0
        if self._export_handle is not None:
            dptr, _ = self._export_handle
            try:
                if self._acl.aclrtIpcMemClose(ctypes.c_void_p(dptr)) == 0:
                    closed += 1
            except Exception:  # noqa: BLE001
                pass
            self._export_handle = None
        if self._pool_ptr is not None:
            try:
                self._acl.aclrtFree(ctypes.c_void_p(self._pool_ptr))
            except Exception:  # noqa: BLE001
                pass
            self._pool_ptr = None
        return closed


def export_from_checkpoint(
    ckpt_dir: str,
    *,
    rank: int,
    tp_world_size: int = 8,
    out_dir: str,
    dev: int = 0,
    int8_routed: bool = False,
) -> Dict[str, Any]:
    """Convenience: load a rank bundle from a checkpoint + export it.

    Uses ``weight_loader.load_step3p5_weights_for_rank`` which already applies
    every vLLM->PyPTO transform (qkv split/transpose, w8a8 dequant, head-pad,
    fp32 promotion for gate/router_bias). The result is the exact per-rank
    bundle ``Step3p5DecodeFwd`` consumes, so the worker can bind each key
    straight to a kernel arg.

    For LIVE serving the bundle would instead be assembled directly from vLLM's
    resident ``nn.Parameter`` tensors using the same transform plan
    (``weight_translate.build_vllm_to_pypto_transform_plan``); that path is
    OPEN_DEVICE_QUESTION (4) below — it must not copy weights to host but
    transform device-side into the pool. The checkpoint path here is the
    numerically-identical reference used by the card-free validation harness.
    """
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        load_step3p5_weights_for_rank,
        verify_bundle_shapes,
    )
    import torch  # noqa: PLC0415
    bundle = load_step3p5_weights_for_rank(
        ckpt_dir, rank, tp_world_size, int8_routed=int8_routed,
    )
    verify_bundle_shapes(bundle, tp_world_size)
    # The whole_decode host_orch expects FP32 for the norm weights + final_norm
    # (matching the dummy device harness), but weight_loader stores norms as bf16.
    # Zero-copy IPC cannot cast at read time, so materialize FP32 bytes here so the
    # exported pool + map dtype are FP32 (moe_gate_w/moe_router_bias already FP32).
    _PROG_FP32 = ("input_rms_weight", "post_attn_rms_weight", "q_norm_weight",
                  "k_norm_weight", "final_norm_weight")
    for _k in _PROG_FP32:
        if _k in bundle and str(bundle[_k].dtype) != "torch.float32":
            bundle[_k] = bundle[_k].to(torch.float32)
    exp = WeightIpcExporter(dev)
    return exp.export(
        bundle, out_dir=out_dir, rank=rank, tp_world_size=tp_world_size,
    )


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

    def __init__(self, peer_base: int, pool_map: Dict[str, Any]) -> None:
        self.peer_base = int(peer_base)
        self.pool_map = pool_map
        self._map = pool_map["map"]

    @classmethod
    def from_files(cls, key_path: str, map_path: str, *, rt, worker_id: int = 0) -> "WeightIpcMap":
        """Read key + map, import the pool into the worker's address space.

        ``rt`` is a ``DistributedWorker`` (or anything exposing
        ``import_ipc(key, worker_id=0) -> int``). Returns a WeightIpcMap bound
        to the imported peer_base.
        """
        with open(key_path, "rb") as f:
            key = f.read()
        peer_base = int(rt.import_ipc(key, worker_id=worker_id))
        with open(map_path) as f:
            pool_map = json.load(f)
        print(
            f"[weight-ipc importer] ONE_KEY pool import peer_base={hex(peer_base)} "
            f"keys={len(pool_map['map'])} pool_GiB={pool_map['pool_bytes']/2**30:.2f}",
            flush=True,
        )
        return cls(peer_base, pool_map)

    def device_tensor(self, key: str):
        """Build the zero-copy DeviceTensor for one bundle key.

        Returns a ``pypto.runtime.device_tensor.DeviceTensor`` backed by
        ``peer_base + offset`` with the key's shape/dtype — pass it directly
        as the kernel arg in place of a torch.Tensor (no H2D/D2H).
        """
        from pypto.runtime.device_tensor import DeviceTensor  # noqa: PLC0415
        if key not in self._map:
            raise KeyError(
                f"weight-ipc map missing key={key!r} "
                f"(have {sorted(self._map)[:8]}...)",
            )
        entry = self._map[key]
        offset = int(entry["offset"])
        shape = tuple(int(s) for s in entry["shape"])
        dtype = _torch_dtype(entry["dtype"])
        return DeviceTensor(self.peer_base + offset, shape, dtype)

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
def import_weights_all(rt, out_dir: str, *, tp: int, dev_offset: int = 0) -> List["WeightIpcMap"]:
    """Batch-import all ``tp`` per-rank weight pools via ``DistributedWorker.import_ipc_all``.

    Reads ``pypto_weight.key.rank{r}`` + ``pypto_weight_map.rank{r}.json`` from
    ``out_dir`` (written by ``WeightIpcExporter.export``), imports every rank's pool
    ONCE into the resident worker's chip children (device ``dev_offset + r``), and
    returns per-rank ``WeightIpcMap`` (peer_base = imported VA). Mirrors the KV path
    in ``_stage_whole_decode_run.py`` (device_key_map -> import_ipc_all -> per-rank Map).
    """
    device_key_map: Dict[int, bytes] = {}
    maps_json: List[Dict[str, Any]] = []
    for r in range(tp):
        with open(os.path.join(out_dir, f"pypto_weight.key.rank{r}"), "rb") as f:
            device_key_map[dev_offset + r] = f.read()
        with open(os.path.join(out_dir, f"pypto_weight_map.rank{r}.json")) as f:
            maps_json.append(json.load(f))
    vas = rt.import_ipc_all(device_key_map)  # {device_id: peer VA}
    print(
        "[weight-ipc importer] import_ipc_all peer_bases="
        + str([hex(vas[dev_offset + r]) for r in range(tp)]),
        flush=True,
    )
    return [WeightIpcMap(vas[dev_offset + r], maps_json[r]) for r in range(tp)]


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
#      run the transform (transpose / qkv-split / w8a8-dequant / head-pad)
#      DEVICE-SIDE into that slot (a tiny pypto kernel or aclrtMemcpy D2D for
#      the pure-transpose cases + a dequant kernel for W8A8 routed experts);
#   3. then aclrtIpcMemGetExportKey the whole pool ONCE.
# This avoids both the host round-trip AND duplicating weights in a second
# process. The transform kernels are the same ones the offline
# weight_loader uses (see _dequant_w8a8 / _to_bf16 / _to_fp32), just emitted
# as device ops. NOT implemented here — card-free; the lead wires device-side
# transforms when validating live serving. The map schema + importer are
# identical regardless of how the pool was filled.


# =============================================================================
# Smoke self-check (card-free): validate layout planning + map round-trip
# against the canonical bundle contract WITHOUT a device.
# =============================================================================
def _smoke_layout() -> int:
    """Plan a layout from expected_shapes + verify offsets/alignment/dtypes.

    No device, no ACL, no checkpoint load — pure dict math. Confirms:
      - every bundle key gets an offset
      - offsets are 512-aligned and non-overlapping
      - dtypes match the fp32/bf16 rule
      - total pool_bytes is sane
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from models.step3p5.weight_loader import expected_shapes  # noqa: PLC0415

    shapes = expected_shapes(8)
    # Synthesize a host bundle of the right shape/dtype per key.
    import torch  # noqa: PLC0415
    bundle: Dict[str, Any] = {}
    for key, shape in shapes.items():
        dtype = torch.float32 if key in _FP32_KEYS else torch.bfloat16
        bundle[key] = torch.empty(shape, dtype=dtype)
    layout = WeightIpcExporter.plan_layout(bundle)
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
        f"fp32_keys={[k for k in keys if k in _FP32_KEYS]}",
        flush=True,
    )
    # Round-trip the map JSON the importer would read.
    map_obj = {
        "version": 1, "rank": 0, "tp_world_size": 8,
        "pool_bytes": pool_bytes, "pool_dtype_bytes": 2,
        "map": {k: {"offset": o, "shape": list(s), "dtype": d, "nbytes": n}
                for k, o, s, d, n in layout},
    }
    json.loads(json.dumps(map_obj))  # serializable
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
