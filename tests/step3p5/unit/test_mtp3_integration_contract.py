from __future__ import annotations

import torch

from models.step3p5 import weight_loader as keys
from models.step3p5.config import (
    BATCH,
    HEAD_DIM,
    HIDDEN,
    HIDDEN_Q_SWA_LOCAL,
    INTERMEDIATE_LOCAL,
    KV_CACHE_ROWS_DYN,
    KV_HIDDEN_LOCAL,
    NUM_HEADS_SWA_LOCAL_PAD,
    NUM_NEXTN_PREDICT_LAYERS,
    TP_WORLD_SIZE,
    VOCAB,
    VOCAB_LOCAL,
)
from models.step3p5.mtp_fwd import (
    COMM_CONTROL_SIGNAL_BYTES,
    COMM_SCALAR_DATA_BYTES,
    MTP_CACHE_ROWS,
    MTP_EH_ROWS,
    MTP_HIDDEN_ROWS,
    MTP_INTER_ROWS,
    MTP_QHIDDEN_ROWS,
    MTP_VOCAB_ROWS,
)
from tests.step3p5.harnesses._stage_whole_mtp3_ipc import (
    _load_previous_hidden,
)
from tools.step3p5.pypto_weight_ipc import WeightIpcExporter


def test_mtp3_static_shapes_and_window_alignment() -> None:
    mtp = NUM_NEXTN_PREDICT_LAYERS
    assert MTP_EH_ROWS == mtp * (HIDDEN // TP_WORLD_SIZE)
    assert MTP_HIDDEN_ROWS == mtp * HIDDEN
    assert MTP_QHIDDEN_ROWS == mtp * HIDDEN_Q_SWA_LOCAL
    assert MTP_INTER_ROWS == mtp * INTERMEDIATE_LOCAL
    assert MTP_VOCAB_ROWS == mtp * VOCAB_LOCAL
    assert MTP_CACHE_ROWS == mtp * KV_CACHE_ROWS_DYN
    assert COMM_CONTROL_SIGNAL_BYTES == 512
    assert COMM_SCALAR_DATA_BYTES == 512


def test_mtp3_ipc_layout_dtype_and_alignment_contract() -> None:
    mtp = NUM_NEXTN_PREDICT_LAYERS
    bundle = {
        keys.KEY_MOE_W_GATE_R: torch.zeros(
            1, 1, 1, 1, dtype=torch.int8
        ),
        keys.KEY_MOE_W_GATE_R_SCALE: torch.zeros(
            1, 1, 1, dtype=torch.float32
        ),
        keys.KEY_MTP_ENORM: torch.zeros(
            mtp, HIDDEN, dtype=torch.float32
        ),
        keys.KEY_MTP_EH_PROJ: torch.zeros(
            mtp, 1, 1, dtype=torch.bfloat16
        ),
        keys.KEY_MTP_WG: torch.zeros(
            mtp, 1, NUM_HEADS_SWA_LOCAL_PAD, dtype=torch.bfloat16
        ),
        "mtp_k_cache": torch.zeros(
            mtp, 1, HEAD_DIM, dtype=torch.bfloat16
        ),
    }

    layout = WeightIpcExporter.plan_layout(bundle)
    entries = {key: (offset, dtype) for key, offset, _, dtype, _ in layout}
    assert all(offset % 512 == 0 for offset, _ in entries.values())
    assert entries[keys.KEY_MOE_W_GATE_R][1] == "int8"
    assert entries[keys.KEY_MOE_W_GATE_R_SCALE][1] == "float32"
    assert entries[keys.KEY_MTP_ENORM][1] == "float32"
    assert entries[keys.KEY_MTP_EH_PROJ][1] == "bfloat16"
    assert entries["mtp_k_cache"][1] == "bfloat16"


def test_previous_hidden_single_and_multi_batch_padding(tmp_path) -> None:
    hidden_path = tmp_path / "P42_nh_row0.pt"
    source_row = torch.arange(HIDDEN, dtype=torch.float32)
    source = source_row.unsqueeze(0).repeat(TP_WORLD_SIZE, 1)
    torch.save(source, hidden_path)

    previous = _load_previous_hidden(
        str(hidden_path),
        tp=TP_WORLD_SIZE,
        batch=BATCH,
        hidden=HIDDEN,
        active_batch=3,
    )
    assert previous.dtype == torch.bfloat16
    assert tuple(previous.shape) == (
        TP_WORLD_SIZE,
        BATCH,
        HIDDEN,
    )
    assert torch.equal(previous[:, 0], source.to(torch.bfloat16))
    assert torch.equal(previous[:, 1], source.to(torch.bfloat16))
    assert torch.equal(previous[:, 2], source.to(torch.bfloat16))
    assert torch.count_nonzero(previous[:, 3:]) == 0


def test_mtp3_vocab_and_projection_partition_invariants() -> None:
    assert VOCAB == TP_WORLD_SIZE * VOCAB_LOCAL
    assert HIDDEN % TP_WORLD_SIZE == 0
    assert KV_HIDDEN_LOCAL == HEAD_DIM
