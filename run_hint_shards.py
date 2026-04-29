import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_OUTPUT_DIR = "hint_datasets/shards"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run generate_hint_sample.py over many index shards in parallel. "
            "Each shard writes to its own JSONL file."
        )
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dataset index to generate.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        required=True,
        help="Last dataset index to generate, inclusive.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100,
        help="Number of samples per shard.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of shard processes to run at the same time.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4.1-mini",
        help="OpenRouter model id.",
    )
    parser.add_argument(
        "--hint_type",
        choices=["concise", "structured", "full_rewrite", "all"],
        default="structured",
        help="Hint type passed to generate_hint_sample.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for shard JSONL files. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--prefix",
        default="medical_hints",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing shard files and regenerate them.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not pass --debug to generate_hint_sample.py.",
    )
    return parser.parse_args()


def count_jsonl_rows(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def build_shards(start_index, end_index, shard_size, output_dir, prefix):
    shards = []
    for shard_start in range(start_index, end_index + 1, shard_size):
        shard_end = min(shard_start + shard_size - 1, end_index)
        output_file = (
            output_dir / f"{prefix}_{shard_start:05d}_{shard_end:05d}.jsonl"
        )
        shards.append((shard_start, shard_end, output_file))
    return shards


def sum_shard_costs(shards):
    total_cost = 0.0
    total_rows = 0
    cost_entries = 0
    for _, _, output_file in shards:
        if not output_file.exists():
            continue
        with output_file.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                total_rows += 1
                row = json.loads(line)
                for details in row.get("cost_details", {}).values():
                    cost = details.get("estimated_cost", {}).get("usd")
                    if cost is not None:
                        total_cost += float(cost)
                        cost_entries += 1
    return total_cost, total_rows, cost_entries


def run_shard(args, shard):
    shard_start, shard_end, output_file = shard
    expected_rows = shard_end - shard_start + 1
    existing_rows = count_jsonl_rows(output_file)

    if output_file.exists() and args.force:
        output_file.unlink()
        existing_rows = 0

    if existing_rows >= expected_rows:
        return {
            "status": "skipped",
            "range": (shard_start, shard_end),
            "output_file": output_file,
            "message": f"already has {existing_rows} rows",
        }

    if output_file.exists() and existing_rows > 0:
        return {
            "status": "partial",
            "range": (shard_start, shard_end),
            "output_file": output_file,
            "message": (
                f"has only {existing_rows}/{expected_rows} rows; "
                "rerun with --force to regenerate this shard"
            ),
        }

    script_path = Path(__file__).with_name("generate_hint_sample.py")
    command = [
        sys.executable,
        str(script_path),
        "--model",
        args.model,
        "--start-index",
        str(shard_start),
        "--end-index",
        str(shard_end),
        "--hint_type",
        args.hint_type,
        "--output-file",
        str(output_file),
    ]
    if not args.no_debug:
        command.append("--debug")

    start_time = time.monotonic()
    completed = subprocess.run(command)
    elapsed = time.monotonic() - start_time

    if completed.returncode == 0:
        return {
            "status": "done",
            "range": (shard_start, shard_end),
            "output_file": output_file,
            "seconds": elapsed,
            "message": f"finished in {elapsed:.1f}s",
        }

    return {
        "status": "failed",
        "range": (shard_start, shard_end),
        "output_file": output_file,
        "seconds": elapsed,
        "returncode": completed.returncode,
        "message": f"failed with exit code {completed.returncode}",
    }


def main():
    args = parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.end_index < args.start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index.")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if args.parallel <= 0:
        raise ValueError("--parallel must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = build_shards(
        args.start_index,
        args.end_index,
        args.shard_size,
        output_dir,
        args.prefix,
    )

    print("HINT SHARD GENERATION")
    print("=" * 80)
    print(f"range={args.start_index}-{args.end_index}")
    print(f"shards={len(shards)} shard_size={args.shard_size}")
    print(f"parallel={args.parallel}")
    print(f"model={args.model}")
    print(f"hint_type={args.hint_type}")
    print(f"output_dir={output_dir}")

    counts = {"done": 0, "skipped": 0, "partial": 0, "failed": 0}
    start_time = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_shard, args, shard): shard for shard in shards}
        for future in as_completed(futures):
            result = future.result()
            counts[result["status"]] += 1
            shard_start, shard_end = result["range"]
            print(
                f"[{result['status']}] {shard_start:05d}-{shard_end:05d} "
                f"{result['output_file']} - {result['message']}"
            )

    elapsed = time.monotonic() - start_time
    total_cost, total_rows, cost_entries = sum_shard_costs(shards)
    print("\nSUMMARY")
    print("=" * 80)
    print(f"done={counts['done']}")
    print(f"skipped={counts['skipped']}")
    print(f"partial={counts['partial']}")
    print(f"failed={counts['failed']}")
    print(f"jsonl_rows={total_rows}")
    print(f"cost_entries={cost_entries}")
    print(f"total_cost_usd=${total_cost:.6f}")
    print(f"elapsed={elapsed:.1f}s")

    if counts["partial"] or counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
