from __future__ import annotations

import argparse

from tools.step3p5.decode_acceptance import run_acceptance


def test_decode_acceptance_synthetic_covers_main_and_mtp() -> None:
    args = argparse.Namespace(
        ckpt_dir=None,
        rank=0,
        tp_world_size=8,
        batch=1,
        seed=0,
        pass_rate=1.0,
        json=True,
    )

    report = run_acceptance(args)

    assert report["ok"]
    assert report["dispatcher"]["expected_layers"] == 48
    assert report["dispatcher"]["kinds"]["mtp_swa_dense"] == 3
    assert report["bundle"]["mode"] == "synthetic"
    assert report["bundle"]["has_mtp"]
    assert report["precision"]["main_logits_pass_rate"] == 1.0
    assert report["precision"]["mtp_logits_pass_rates"] == [1.0, 1.0, 1.0]
