#!/usr/bin/env python3
"""Run Step3p5 W8A8 prefill precision gates and package artifacts.

The suite mirrors the decode precision closure flow:

1. consume vLLM eager detail dumps as the golden oracle;
2. replay PyPTO host/reference math from the W8A8 checkpoint bundle;
3. compare per-layer non-attention-core tensors;
4. compare final RMSNorm + LM-head logits;
5. emit JSON + Markdown reports and optionally a tar package.

Long-prefill cases (32k/64k/128k) can be checked with an evenly-spaced
per-layer token sample via ``--max-detail-tokens`` so the CPU MoE replay
finishes in bounded time while still covering prompt head/middle/tail.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SEQ_LENGTHS = (1024, 4096, 8192, 32768, 65536, 131072)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _case_seq_len(case_name: str) -> int | None:
    matches = re.findall(r"(\d+)\s*k", case_name.lower())
    if matches:
        return int(matches[-1]) * 1024
    matches = re.findall(r"(\d+)", case_name)
    if matches:
        value = int(matches[-1])
        if value in {1, 4, 8, 32, 64, 128}:
            return value * 1024
        return value
    return None


def _load_manifest(golden_root: Path) -> dict[str, Any]:
    manifest_path = golden_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing golden manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _case_dump_root(golden_root: Path, case: dict[str, Any]) -> Path:
    if "dump_root" in case:
        return Path(case["dump_root"])
    if "dump_dir" in case:
        return Path(case["dump_dir"])
    candidate = golden_root / case["name"] / "dump"
    if candidate.exists():
        return candidate
    # Decode golden manifests store absolute paths in dump_files.  Fall back to
    # their common parent so the same manifest layout also works for prefill.
    dump_files = case.get("dump_files") or []
    if dump_files:
        return Path(dump_files[0]["file"]).parent
    raise FileNotFoundError(f"cannot infer dump root for case {case.get('name')!r}")


def _select_cases(
    manifest: dict[str, Any],
    required_lengths: tuple[int, ...],
    selected: list[str] | None,
) -> list[dict[str, Any]]:
    cases = [c for c in manifest.get("cases", []) if c.get("dump_files")]
    if selected:
        wanted = set(selected)
        cases = [c for c in cases if c.get("name") in wanted]
    if not required_lengths:
        return cases

    by_len: dict[int, dict[str, Any]] = {}
    for case in cases:
        seq = case.get("seq_len") or case.get("prompt_len") or _case_seq_len(case["name"])
        if seq is not None:
            by_len[int(seq)] = case
    missing = [seq for seq in required_lengths if seq not in by_len]
    if missing:
        raise FileNotFoundError(
            "golden cases do not cover required prefill lengths: "
            + ", ".join(str(x) for x in missing)
        )
    return [by_len[seq] for seq in required_lengths]


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Step3p5 W8A8 prefill precision report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Checkpoint: `{report['ckpt_dir']}`",
        f"- Golden root: `{report['golden_root']}`",
        f"- Output root: `{report['output_root']}`",
        f"- Overall: **{'PASS' if report['ok'] else 'FAIL'}**",
        "",
        "| Case | Seq len | Detail | Final logits | Worst pass rate | Notes |",
        "|---|---:|---|---|---:|---|",
    ]
    for case in report["cases"]:
        detail = case.get("detail", {})
        logits = case.get("final_logits", {})
        lines.append(
            "| {name} | {seq} | {detail_status} | {logits_status} | {worst:.6f} | {notes} |".format(
                name=case["name"],
                seq=case.get("seq_len", "n/a"),
                detail_status="PASS" if detail.get("ok") else "FAIL",
                logits_status="PASS" if logits.get("ok") else "FAIL",
                worst=float(detail.get("worst_pass_rate", 0.0)),
                notes=case.get("notes", ""),
            )
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python tools/step3p5/prefill_precision_suite.py \\",
        f"  --golden-root {report['golden_root']} \\",
        f"  --ckpt-dir {report['ckpt_dir']} \\",
        f"  --output-root {report['output_root']} \\",
        "  --make-tar",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _package(output_root: Path, tar_path: Path, include_golden: bool) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if tar_path.suffixes[-2:] == [".tar", ".gz"] else "w"
    with tarfile.open(tar_path, mode) as tar:
        for path in sorted(output_root.rglob("*")):
            if path == tar_path or path.is_dir():
                continue
            if not include_golden and path.name.endswith(".pt") and "golden" in path.parts:
                continue
            tar.add(path, arcname=path.relative_to(output_root.parent))


def run(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(_repo_root()))

    from tools.step3p5.final_logits_from_vllm import generate as generate_logits
    from tools.step3p5.pypto_all_layers_detail_compare import compare as compare_detail

    golden_root = Path(args.golden_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(golden_root)
    required_lengths = tuple(args.seq_len or DEFAULT_SEQ_LENGTHS)
    cases = _select_cases(manifest, required_lengths, args.case)

    case_reports: list[dict[str, Any]] = []
    for case in cases:
        case_name = case["name"]
        case_out = output_root / case_name
        case_out.mkdir(parents=True, exist_ok=True)
        dump_root = _case_dump_root(golden_root, case)

        seq_len_hint = case.get("seq_len") or case.get("prompt_len") or _case_seq_len(case_name)
        detail_args = argparse.Namespace(
            dump_root=str(dump_root),
            ckpt_dir=args.ckpt_dir,
            tp_world_size=args.tp_world_size,
            rtol=args.rtol,
            atol=args.atol,
            mlp_rtol=args.mlp_rtol,
            mlp_atol=args.mlp_atol,
            pass_rate=args.pass_rate,
            max_tokens=args.max_detail_tokens,
            out=None,
        )
        detail_report = compare_detail(detail_args)
        detail_path = case_out / "prefill_detail_report.json"
        detail_path.write_text(
            json.dumps(detail_report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        logits_out = case_out / "final_logits"
        logits_report = generate_logits(argparse.Namespace(
            golden_root=str(golden_root),
            output_root=str(logits_out),
            ckpt_dir=args.ckpt_dir,
            case=[case_name],
            tp_world_size=args.tp_world_size,
            max_steps=1,
            chunk_size=args.logits_chunk_size,
            rtol=args.rtol,
            atol=args.atol,
            pass_rate=args.pass_rate,
        ))

        case_reports.append({
            "name": case_name,
            "seq_len": seq_len_hint or detail_report.get("seq_len"),
            "dump_root": str(dump_root),
            "detail_report": str(detail_path),
            "detail": {
                "ok": detail_report["ok"],
                "num_checks": detail_report["num_checks"],
                "worst_pass_rate": detail_report["worst_pass_rate"],
                "token_sample_count": detail_report.get("token_sample_count"),
            },
            "final_logits": {
                "ok": logits_report["ok"],
                "report": str(logits_out / "final_logits_report.json"),
            },
            "ok": detail_report["ok"] and logits_report["ok"],
            "notes": (
                "sampled detail tokens"
                if detail_report.get("token_sample_count") != detail_report.get("seq_len")
                else "full detail tokens"
            ),
        })

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "golden_root": str(golden_root),
        "ckpt_dir": args.ckpt_dir,
        "output_root": str(output_root),
        "tp_world_size": args.tp_world_size,
        "required_seq_lengths": list(required_lengths),
        "cases": case_reports,
        "ok": bool(case_reports) and all(c["ok"] for c in case_reports),
    }
    report_path = output_root / "STEP3P5_W8A8_PREFILL_REPORT.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md_path = output_root / "STEP3P5_W8A8_PREFILL_REPORT.md"
    _write_markdown(report, md_path)

    if args.copy_golden_manifest:
        shutil.copy2(golden_root / "manifest.json", output_root / "golden_manifest.json")
    if args.make_tar:
        tar_path = Path(args.tar_path) if args.tar_path else output_root.with_suffix(".tar")
        report["tar_path"] = str(tar_path)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        _package(output_root, tar_path, include_golden=args.include_golden_in_tar)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-root", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--seq-len", type=int, action="append", default=None,
                        help="Required prompt length. Repeatable. Default: 1k,4k,8k,32k,64k,128k.")
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--mlp-rtol", type=float, default=8e-2)
    parser.add_argument("--mlp-atol", type=float, default=2e-1)
    parser.add_argument("--pass-rate", type=float, default=0.997)
    parser.add_argument("--max-detail-tokens", type=int, default=1024)
    parser.add_argument("--logits-chunk-size", type=int, default=4096)
    parser.add_argument("--copy-golden-manifest", action="store_true", default=True)
    parser.add_argument("--make-tar", action="store_true")
    parser.add_argument("--tar-path", default=None)
    parser.add_argument("--include-golden-in-tar", action="store_true")
    args = parser.parse_args()

    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
