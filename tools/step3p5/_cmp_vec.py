#!/usr/bin/env python3
"""Compare two saved 1-D tensors (row0 hidden vectors): cos + ratio_allclose + norms.

Usage: python tools/step3p5/_cmp_vec.py <golden.pt> <device.pt> [atol] [rtol]
Metric discipline (per project memory): compare the VALID row0 vector via cos and
ratio_allclose(atol=0.04), NOT max-abs (max is confounded by padding rows).
"""
from __future__ import annotations

import sys

import torch


def main() -> int:
    ga, da = sys.argv[1], sys.argv[2]
    atol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.04
    rtol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.04
    g = torch.load(ga, map_location="cpu").float().flatten()
    d = torch.load(da, map_location="cpu").float().flatten()
    n = min(g.numel(), d.numel())
    g, d = g[:n], d[:n]
    cos = torch.nn.functional.cosine_similarity(g, d, dim=0).item()
    diff = (g - d).abs()
    tol = atol + rtol * g.abs()
    pass_rate = (diff <= tol).float().mean().item()
    max_ad = diff.max().item()
    worst = int(diff.argmax())
    print(f"golden={ga}")
    print(f"device={da}")
    print(f"  n={n} cos={cos:.6f} ratio_allclose(atol={atol},rtol={rtol})={pass_rate:.6f}")
    print(f"  |g|norm={g.norm():.4f} |d|norm={d.norm():.4f}  magratio(d/g)={(d.norm()/g.norm()).item():.4f}")
    print(f"  max|g-d|={max_ad:.5f} @ch{worst}  g[{worst}]={g[worst]:.5f} d[{worst}]={d[worst]:.5f}")
    print(f"  VERDICT={'MATCH' if (cos >= 0.999 and pass_rate >= 0.98) else 'DIVERGE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
