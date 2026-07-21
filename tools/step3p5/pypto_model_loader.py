#!/usr/bin/env python3
"""``load_format=pypto`` model loader for the Step3p5 tail-only decode instance.

Design (progress ledger C3 / vllm-live-backend §2, §5.3):

The tail-only decode ``Step3p5Model`` holds target ``embed_tokens``, final
``norm`` and ``lm_head`` parameters, while the MTP draft shell holds only the
three shared heads plus zero-parameter Attention registration objects.  The
model ``load_weights`` seams filter the checkpoint to those vLLM-owned
prefixes before strict loading.

Beyond the tail weights, the PyPTO loader owns the *decoder* side:

  1. build this TP rank's native-W8A8 PyPTO weight bundle and consolidate it
     into one IPC-exportable device pool (``export_from_checkpoint_resident``);
  2. fail closed unless the exported pool-map passes ``validate_weight_map``
     with the native-W8A8 contract (routed INT8, scales FP32, no BF16 dequant,
     512-byte aligned, non-overlapping);
  3. retain the ``WeightIpcExporter`` on the model instance so the pool + IPC
     key stay valid for the whole serving lifetime;
  4. drop the CPU-side bundle so only the device pool remains resident.

The combined PyPTO body pool keeps ``KEY_EMBED`` because the selected MTP
program owns MTP token lookup.  Main decode still receives vLLM-embedded
hidden and never reads that table.  Target final norm/LM head and all MTP
shared-head tensors are excluded from the PyPTO pool.

Registration happens on import via ``@register_model_loader("pypto")``; the
sitecustomize shim imports this module before vLLM builds the model.

Environment inputs (the loader only receives a ``LoadConfig``):
  PYPTO_STEP3P5_EXPORT_WEIGHTS  "1" to build the device exporter (needs a card).
                                Off by default so tail-weight-only bring-up on a
                                card-free host still works.
  PYPTO_WEIGHT_IPC_DIR          shared dir for ``pypto_weight.key.rank{r}`` +
                                ``pypto_weight_map.rank{r}.json`` (required when
                                exporting; missing => fail closed).
"""
from __future__ import annotations

import json
import os

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

logger = init_logger(__name__)


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


@register_model_loader("pypto")
class PyPtoStep3p5ModelLoader(DefaultModelLoader):
    """Tail-only loader: vLLM keeps embed/norm/lm_head; PyPTO owns W8A8 decoder."""

    def load_weights(self, model, model_config: ModelConfig) -> None:
        # DefaultModelLoader materializes only the parameters that exist on the
        # tail-only model: embed_tokens, final norm, lm_head.  Decoder checkpoint
        # tensors have no target parameter and are skipped by name matching.
        super().load_weights(model, model_config)
        self._maybe_export_pypto_decoder_weights(model, model_config)

    # -- native-W8A8 decoder residency (device-gated) -----------------------
    def _maybe_export_pypto_decoder_weights(
        self, model, model_config: ModelConfig
    ) -> None:
        if not _truthy("PYPTO_STEP3P5_EXPORT_WEIGHTS"):
            logger.warning(
                "PYPTO loader: decoder weight export disabled "
                "(PYPTO_STEP3P5_EXPORT_WEIGHTS!=1); tail weights only. The live "
                "PyPTO decode path is unavailable until an exporter is built."
            )
            return

        out_dir = os.environ.get("PYPTO_WEIGHT_IPC_DIR")
        if not out_dir:
            # Fail closed: an export was requested but there is nowhere to write
            # the IPC key/map, so no rank could import the pool.
            raise RuntimeError(
                "PYPTO_STEP3P5_EXPORT_WEIGHTS=1 requires PYPTO_WEIGHT_IPC_DIR"
            )

        from vllm.distributed import (  # noqa: PLC0415
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
            export_from_checkpoint_resident,
            validate_weight_map,
        )

        rank = get_tensor_model_parallel_rank()
        tp = get_tensor_model_parallel_world_size()
        dev = self._current_device_index()
        ckpt = model_config.model

        # Native W8A8: routed experts stay INT8 with FP32 scales (int8_routed).
        exporter, summary, bundle = export_from_checkpoint_resident(
            ckpt,
            rank=rank,
            tp_world_size=tp,
            out_dir=out_dir,
            dev=dev,
            int8_routed=True,
            production_hidden_only=True,
        )

        # Fail closed BEFORE the model is declared ready: the map this rank just
        # wrote must satisfy the native-W8A8 + structural contract.
        with open(summary["map_path"], encoding="utf-8") as f:
            pool_map = json.load(f)
        validate_weight_map(pool_map, native_w8a8=True)

        # Own the exporter for the serving lifetime; drop the CPU bundle so only
        # the device pool remains resident.
        model._pypto_weight_exporter = exporter
        del bundle
        logger.info(
            "PYPTO loader: rank=%d native-W8A8 pool exported (%.2f GiB, %d keys) "
            "and validated; owner attached to model",
            rank,
            summary["pool_bytes"] / 2**30,
            summary["num_keys"],
        )

    @staticmethod
    def _current_device_index() -> int:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001
            pass
        try:
            from vllm.distributed import (  # noqa: PLC0415
                get_tensor_model_parallel_rank,
            )

            return int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            return 0
