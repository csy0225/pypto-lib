#!/usr/bin/env python3
"""Compare per-layer hidden dumps between opt (loop-form) and instrumented baseline.

Locates the first-divergence layer between two per-layer dump trees. Each tree has
the layout written by the accuracy-bisect harness::

    {root}/per_layer_step{step:02d}/layer{li:02d}.pt

Each ``.pt`` file is rank-0 row0 hidden (bf16 on device, loaded to cpu here).
Layer files are optional per step (truncated runs skip later layers); a missing
file on one side is reported as a structural divergence.

Metric discipline (project convention, see ``_cmp_vec.py``): compare the VALID
row0 vector via ``ratio_allclose(atol=0.04, rtol=0.04)`` + cosine, NOT bare
max-abs (padding rows confound max). NaN/Inf in opt side is a hard fail.

Usage::

    python tools/step3p5/compare_per_layer.py <opt_root> <baseline_root> [step] \\
        [--atol A] [--rtol R] [--max-error-ratio M] [--cos-min C] [--layers L0-L44]

``step`` defaults to 0 (first decode step — enough to localize first divergence).
Example::

    python tools/step3p5/compare_per_layer.py opt_out baseline_out 00
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _load(path: Path) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        # fall back to first tensor value if harness wrapped it
        for v in t.values():
            if isinstance(v, torch.Tensor):
                t = v
                break
    return t.float().flatten()


def _ratio_allclose_pass_rate(
    actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float
) -> float:
    diff = (actual - expected).abs()
    tol = atol + rtol * expected.abs()
    return (diff <= tol).float().mean().item()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm().item()
    nb = b.norm().item()
    if na == 0.0 or nb == 0.0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _layer_path(root: Path, step: int, li: int) -> Path:
    return root / f"per_layer_step{step:02d}" / f"layer{li:02d}.pt"


def _parse_layers(spec: str | None, default_max: int = 45) -> list[int]:
    if not spec:
        return list(range(default_max))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("opt_root", type=Path, help="opt per-layer dump root")
    ap.add_argument("baseline_root", type=Path, help="baseline per-layer dump root")
    ap.add_argument("step", nargs="?", default="0", help="decode step index (default 0)")
    ap.add_argument("--atol", type=float, default=0.04)
    ap.add_argument("--rtol", type=float, default=0.04)
    ap.add_argument(
        "--max-error-ratio",
        type=float,
        default=0.1,
        help="ratio_allclose threshold for pass (default 0.1, project bisect convention)",
    )
    ap.add_argument("--cos-min", type=float, default=0.999)
    ap.add_argument("--layers", type=str, default=None, help="e.g. 0-44 (default 0..44)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    step = int(args.step)
    layers = _parse_layers(args.layers, default_max=45)

    print(
        f"opt_root={args.opt_root}\nbaseline_root={args.baseline_root}\n"
        f"step={step} layers={layers[0]}..{layers[-1]} "
        f"atol={args.atol} rtol={args.rtol} max_error_ratio={args.max_error_ratio} "
        f"cos_min={args.cos_min}"
    )

    rows: list[dict] = []
    first_diverge: int | None = None
    first_struct_diverge: int | None = None

    for li in layers:
        op = _layer_path(args.opt_root, step, li)
        bp = _layer_path(args.baseline_root, step, li)
        opt_ok = op.exists()
        base_ok = bp.exists()

        if not (opt_ok and base_ok):
            tag = []
            if not opt_ok:
                tag.append("opt MISSING")
            if not base_ok:
                tag.append("baseline MISSING")
            row = {
                "layer": li,
                "status": "STRUCT_MISSING",
                "note": ",".join(tag),
            }
            rows.append(row)
            print(
                f"L{li:02d}: STRUCT_MISSING ({row['note']})  "
                f"opt={op.name} base={bp.name}"
            )
            if first_struct_diverge is None:
                first_struct_diverge = li
            continue

        o = _load(op)
        b = _load(bp)
        n = min(o.numel(), b.numel())
        o, b = o[:n], b[:n]

        nan_cnt = int(torch.isnan(o).sum().item())
        inf_cnt = int(torch.isinf(o).sum().item())
        cos = _cosine(o, b)
        pr = _ratio_allclose_pass_rate(o, b, args.atol, args.rtol)
        max_ad = (o - b).abs().max().item()
        o_abs_max = o.abs().max().item()
        b_abs_max = b.abs().max().item()
        mag = (o.norm() / b.norm()).item() if b.norm().item() != 0 else float("inf")

        if nan_cnt or inf_cnt:
            status = "OPT_NANINF"
        elif pr >= (1.0 - args.max_error_ratio) and cos >= args.cos_min:
            status = "PASS"
        else:
            status = "DIVERGE"

        row = {
            "layer": li,
            "status": status,
            "cos": cos,
            "ratio_allclose_pass": pr,
            "max_abs_diff": max_ad,
            "opt_abs_max": o_abs_max,
            "base_abs_max": b_abs_max,
            "mag_ratio_opt_over_base": mag,
            "nan": nan_cnt,
            "inf": inf_cnt,
            "n": n,
        }
        rows.append(row)
        print(
            f"L{li:02d}: {status:9s} cos={cos:.6f} ratio={pr:.6f} "
            f"max|d|={max_ad:.5g} |opt|max={o_abs_max:.5g} |base|max={b_abs_max:.5g} "
            f"mag(o/b)={mag:.4f} nan={nan_cnt} inf={inf_cnt} n={n}"
        )

        if status == "DIVERGE" and first_diverge is None:
            first_diverge = li
        if status == "OPT_NANINF" and first_diverge is None:
            first_diverge = li

    print("\n=== SUMMARY ===")
    if first_struct_diverge is not None:
        print(
            f"first STRUCTURAL divergence (missing layer file): L{first_struct_diverge:02d}"
        )
    if first_diverge is not None:
        print(f"first NUMERICAL divergence: L{first_diverge:02d}")
    else:
        print("no numerical divergence found across compared layers")

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_div = sum(1 for r in rows if r["status"] in ("DIVERGE", "OPT_NANINF"))
    n_miss = sum(1 for r in rows if r["status"] == "STRUCT_MISSING")
    print(f"PASS={n_pass} DIVERGE={n_div} MISSING={n_miss} total={len(rows)}")

    if args.json:
        import json

        print("\n=== JSON ===")
        print(
            json.dumps(
                {
                    "opt_root": str(args.opt_root),
                    "baseline_root": str(args.baseline_root),
                    "step": step,
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "max_error_ratio": args.max_error_ratio,
                    "cos_min": args.cos_min,
                    "first_numerical_divergence": first_diverge,
                    "first_structural_divergence": first_struct_diverge,
                    "rows": rows,
                },
                indent=2,
            )
        )

    return 0 if (first_diverge is None and first_struct_diverge is None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
