#!/usr/bin/env python3
"""Upload the recent 6 experiments x 5 epoch checkpoints to one HF Hub repo.

By default this uploads model/adapter artifacts and skips optimizer/scheduler/RNG
state files. Pass --include-training-state if you explicitly want those large
training-resume files as well.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


CHECKPOINTS: list[tuple[str, str, str]] = [
    ("ceonly-nll-bs64-5ep", "epoch1", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-308"),
    ("ceonly-nll-bs64-5ep", "epoch2", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-616"),
    ("ceonly-nll-bs64-5ep", "epoch3", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-924"),
    ("ceonly-nll-bs64-5ep", "epoch4", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-1232"),
    ("ceonly-nll-bs64-5ep", "epoch5", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-1540"),
    ("ceonly-nll-bs8-5ep", "epoch1", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-2463"),
    ("ceonly-nll-bs8-5ep", "epoch2", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-4926"),
    ("ceonly-nll-bs8-5ep", "epoch3", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-7389"),
    ("ceonly-nll-bs8-5ep", "epoch4", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-9852"),
    ("ceonly-nll-bs8-5ep", "epoch5", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-12315"),
    ("ceonly-dft-bs64-5ep", "epoch1", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-308"),
    ("ceonly-dft-bs64-5ep", "epoch2", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-616"),
    ("ceonly-dft-bs64-5ep", "epoch3", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-924"),
    ("ceonly-dft-bs64-5ep", "epoch4", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-1232"),
    ("ceonly-dft-bs64-5ep", "epoch5", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-1540"),
    ("ceonly-eaft-bs64-5ep", "epoch1", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-308"),
    ("ceonly-eaft-bs64-5ep", "epoch2", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-616"),
    ("ceonly-eaft-bs64-5ep", "epoch3", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-924"),
    ("ceonly-eaft-bs64-5ep", "epoch4", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-1232"),
    ("ceonly-eaft-bs64-5ep", "epoch5", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-1540"),
    ("offpolicy-kdonly-fkl-clip005-bs64-5ep", "epoch1", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-308"),
    ("offpolicy-kdonly-fkl-clip005-bs64-5ep", "epoch2", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-616"),
    ("offpolicy-kdonly-fkl-clip005-bs64-5ep", "epoch3", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-924"),
    ("offpolicy-kdonly-fkl-clip005-bs64-5ep", "epoch4", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-1232"),
    ("offpolicy-kdonly-fkl-clip005-bs64-5ep", "epoch5", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-1540"),
    ("offpolicy-kdonly-fkl-noclip-bs64-5ep", "epoch1", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-308"),
    ("offpolicy-kdonly-fkl-noclip-bs64-5ep", "epoch2", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-616"),
    ("offpolicy-kdonly-fkl-noclip-bs64-5ep", "epoch3", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-924"),
    ("offpolicy-kdonly-fkl-noclip-bs64-5ep", "epoch4", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-1232"),
    ("offpolicy-kdonly-fkl-noclip-bs64-5ep", "epoch5", "outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-1540"),
]


TRAINING_STATE_ALLOW_PATTERNS = ["optimizer.pt", "scheduler.pt", "rng_state.pth"]
DEFAULT_IGNORE_PATTERNS = [*TRAINING_STATE_ALLOW_PATTERNS]


def make_manifest(base_dir: Path) -> dict:
    entries = []
    total_bytes = 0
    for experiment, epoch, rel_path in CHECKPOINTS:
        path = base_dir / rel_path
        adapter = path / "adapter_model.safetensors"
        if not adapter.exists():
            raise FileNotFoundError(f"Missing adapter: {adapter}")
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        total_bytes += size
        entries.append(
            {
                "experiment": experiment,
                "epoch": epoch,
                "local_path": rel_path,
                "hub_path": f"{experiment}/{epoch}",
                "checkpoint": path.name,
                "bytes_in_local_checkpoint": size,
            }
        )
    return {
        "base_model": "Qwen/Qwen3-1.7B",
        "num_checkpoints": len(entries),
        "layout": "<experiment>/<epoch>/",
        "entries": entries,
        "local_bytes_including_training_state": total_bytes,
    }


def write_model_card(repo_dir: Path, repo_id: str, manifest: dict) -> None:
    rows = "\n".join(
        f"| `{e['experiment']}` | `{e['epoch']}` | `{e['hub_path']}` | `{e['checkpoint']}` |"
        for e in manifest["entries"]
    )
    readme = f"""---
base_model: Qwen/Qwen3-1.7B
library_name: peft
tags:
- peft
- lora
- qwen3
- medical
---

# Recent 6 Medical OPSD Checkpoints

This repository stores LoRA adapter checkpoints from 6 recent experiments,
with 5 epoch checkpoints each, for 30 adapters total.

Base model: `Qwen/Qwen3-1.7B`

## Layout

Each adapter lives under:

`<experiment>/<epoch>/`

Example:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(base, "{repo_id}", subfolder="ceonly-nll-bs64-5ep/epoch1")
```

## Checkpoints

| Experiment | Epoch | Hub Path | Original Checkpoint |
|---|---:|---|---|
{rows}
"""
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "README.md").write_text(readme, encoding="utf-8")
    (repo_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hub repo id, e.g. username/repo-name")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--base-dir", default=".", help="Repository root containing outputs/")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-training-state",
        action="store_true",
        help="Also upload optimizer.pt, scheduler.pt, and rng_state.pth files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    api = HfApi()
    manifest = make_manifest(base_dir)

    print(f"Found {manifest['num_checkpoints']} checkpoints.")
    if args.dry_run:
        for entry in manifest["entries"]:
            print(f"[dry-run] {entry['local_path']} -> {entry['hub_path']}")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token is None:
        print("No HF_TOKEN/HUGGINGFACE_HUB_TOKEN env var found; using cached login if available.")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
        token=token,
    )

    metadata_dir = base_dir / ".hf_upload_recent6_metadata"
    write_model_card(metadata_dir, args.repo_id, manifest)
    for file_name in ["README.md", "manifest.json"]:
        print(f"[upload] {file_name}")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            path_or_fileobj=str(metadata_dir / file_name),
            path_in_repo=file_name,
            token=token,
        )

    ignore_patterns = [] if args.include_training_state else DEFAULT_IGNORE_PATTERNS
    for idx, (experiment, epoch, rel_path) in enumerate(CHECKPOINTS, start=1):
        local_path = base_dir / rel_path
        path_in_repo = f"{experiment}/{epoch}"
        print(f"[{idx:02d}/{len(CHECKPOINTS)}] {local_path} -> {args.repo_id}/{path_in_repo}")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            folder_path=str(local_path),
            path_in_repo=path_in_repo,
            ignore_patterns=ignore_patterns,
            commit_message=f"Upload {experiment} {epoch}",
            token=token,
        )

    print("Done.")


if __name__ == "__main__":
    main()
