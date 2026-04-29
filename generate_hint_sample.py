import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from datasets import load_dataset


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "z-ai/glm-4.7-flash"
DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"
DATASET_CONFIG = "en"
DATASET_SPLIT = "train"
MAX_TOKENS_BY_HINT_TYPE = {
    "concise": 512,
    "structured": 768,
    "full_rewrite": 2048,
}
DEFAULT_OUTPUT_DIR = "hint_samples"
MODEL_PRICING_PER_MILLION = {
    "z-ai/glm-4.7-flash": {"input": 0.06, "output": 0.40},
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "openai/gpt-5-mini": {"input": 0.25, "output": 2.00},
    "openai/gpt-5.1": {"input": 1.25, "output": 10.00},
    "openai/gpt-5.1-chat": {"input": 1.25, "output": 10.00},
    "openai/gpt-5.1-codex-mini": {"input": 0.25, "output": 2.00},
    "openai/gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "google/gemini-2.5-flash-lite:nitro": {"input": 0.10, "output": 0.40},
    "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "qwen/qwen3-30b-a3b-instruct-2507": {"input": 0.09, "output": 0.30},
    "qwen/qwen3-235b-a22b-2507": {"input": 0.071, "output": 0.10},
}
SYSTEM_PROMPT = (
    "You write medical teacher-context hints from solved examples. "
    "Return only the requested hint text. "
    "Do not explain the task, show hidden reasoning, add unsupported facts, or include extra commentary."
)


PROMPTS = {
    "concise": """Rewrite the solved example into one concise medical hint.

Requirements:
- Write one short paragraph.
- Do not copy sentences from the original reasoning.
- Use only information supported by the input.
- Keep the key clinical evidence, mechanism, and conclusion.
- Do not use bullet points or step numbers.
- Keep it concise but medically specific.
- Return only the hint.

Question:
{question}

Original reasoning:
{complex_cot}

Final answer:
{response}

Concise hint:""",
    "structured": """Rewrite the solved example into a compact structured medical hint.

Requirements:
- Do not copy sentences from the original reasoning.
- Use only information supported by the input.
- Keep the hint concise but informative.
- Include three sections: Key evidence, Mechanism, and Conclusion.
- Under Key evidence, list the important clinical clues.
- Under Mechanism, explain how the clues lead to the answer.
- Under Conclusion, state the final answer.
- Replace the placeholders below with actual content.
- Return only the hint.

Use this format:
Key evidence:
- ...

Mechanism:
- ...

Conclusion:
- ...

Question:
{question}

Original reasoning:
{complex_cot}

Final answer:
{response}

Structured hint:""",
    "full_rewrite": """Rewrite the solved example into a complete medical explanation.

Requirements:
- Preserve the full clinical logic from the original reasoning.
- Do not copy sentences from the original reasoning.
- Use only information supported by the input.
- Explain how the evidence leads to the final answer.
- Keep the explanation clear and medically specific.
- Do not use bullet points or step numbers.
- Return only the rewritten explanation.

Question:
{question}

Original reasoning:
{complex_cot}

Final answer:
{response}

Full rewritten explanation:""",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate teacher-context hint variants for one sample from "
            "FreedomIntelligence/medical-o1-reasoning-SFT using OpenRouter."
        )
    )
    parser.add_argument(
        "--start-index",
        dest="start_index",
        type=int,
        default=0,
        help="First sample index to process.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=0,
        help="Last sample index to process, inclusive.",
    )
    parser.add_argument(
        "--hint_type",
        choices=["concise", "structured", "full_rewrite", "all"],
        default="all",
        help="Hint variant to generate. Use 'all' to generate all three.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the generated JSON output. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print results only and do not save a JSON file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save raw OpenRouter responses and continue if one hint generation fails.",
    )
    return parser.parse_args()


def get_usage(data):
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion_tokens_details", {}).get("accepted_prediction_tokens")
        or 0
    )
    total_tokens = usage.get("total_tokens") or prompt_tokens + completion_tokens
    reasoning_tokens = 0
    completion_details = usage.get("completion_tokens_details") or {}
    if isinstance(completion_details, dict):
        reasoning_tokens = completion_details.get("reasoning_tokens") or 0
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "openrouter_cost_usd": usage.get("cost"),
        "openrouter_cost_details": usage.get("cost_details"),
    }


def estimate_cost(model, usage):
    if usage.get("openrouter_cost_usd") is not None:
        return {
            "available": True,
            "source": "openrouter_usage",
            "usd": float(usage["openrouter_cost_usd"]),
            "input_usd_per_million": None,
            "output_usd_per_million": None,
        }

    pricing = MODEL_PRICING_PER_MILLION.get(model)
    if not pricing:
        return {
            "available": False,
            "source": None,
            "usd": None,
            "input_usd_per_million": None,
            "output_usd_per_million": None,
        }
    input_cost = usage["prompt_tokens"] * pricing["input"] / 1_000_000
    output_cost = usage["completion_tokens"] * pricing["output"] / 1_000_000
    return {
        "available": True,
        "source": "local_price_table",
        "usd": input_cost + output_cost,
        "input_usd_per_million": pricing["input"],
        "output_usd_per_million": pricing["output"],
    }


def extract_text_from_response(data):
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(
            "OpenRouter returned an error object:\n"
            f"{json.dumps(data['error'], indent=2, ensure_ascii=False)}"
        )

    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "OpenRouter returned an unexpected response shape:\n"
            f"{json.dumps(data, indent=2, ensure_ascii=False)}"
        ) from error

    # Most chat-completion models return text here.
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if isinstance(content, str) and content.strip():
        return content.strip()

    # Some reasoning-capable providers expose text in nonstandard fields.
    for key in ["reasoning", "reasoning_content", "text"]:
        value = message.get(key) or choice.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Last resort for structured reasoning traces.
    reasoning_details = message.get("reasoning_details")
    if isinstance(reasoning_details, list):
        parts = []
        for item in reasoning_details:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    finish_reason = choice.get("finish_reason")
    native_finish_reason = choice.get("native_finish_reason")
    raise RuntimeError(
        "OpenRouter response did not contain text content. "
        f"finish_reason={finish_reason!r}, native_finish_reason={native_finish_reason!r}. "
        "Use --debug and inspect the saved raw_response field."
    )


def call_openrouter(api_key, model, prompt, temperature, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jungseongryong/AdaSD",
            "X-Title": "AdaSD hint generation sample",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"OpenRouter request failed: {error}") from error

    return extract_text_from_response(data), data


def load_dataset_split():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    return dataset


def format_sample(dataset, sample_index):
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={sample_index} is outside dataset size {len(dataset)}"
        )
    sample = dataset[sample_index]
    return {
        "question": str(sample["Question"]).strip(),
        "complex_cot": str(sample["Complex_CoT"]).strip(),
        "response": str(sample["Response"]).strip(),
    }


def main():
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Run: export OPENROUTER_API_KEY='your_key'"
        )

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.end_index < args.start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index.")

    dataset = load_dataset_split()
    hint_types = (
        ["concise", "structured", "full_rewrite"]
        if args.hint_type == "all"
        else [args.hint_type]
    )

    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    total_requests = 0
    total_seconds = 0.0
    for sample_index in range(args.start_index, args.end_index + 1):
        sample_start = time.monotonic()
        sample = format_sample(dataset, sample_index)
        outputs = {}
        errors = {}
        raw_responses = {}
        cost_details = {}

        print("\n" + "#" * 80)
        print(f"SAMPLE {sample_index}")
        print("#" * 80)
        print("\nQUESTION\n" + "=" * 80)
        print(sample["question"])
        print("\nFINAL ANSWER\n" + "=" * 80)
        print(sample["response"])

        for hint_type in hint_types:
            prompt = PROMPTS[hint_type].format(**sample)
            request_start = time.monotonic()
            try:
                text, raw_response = call_openrouter(
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=MAX_TOKENS_BY_HINT_TYPE[hint_type],
                )
                outputs[hint_type] = text
                usage = get_usage(raw_response)
                cost = estimate_cost(args.model, usage)
                cost_details[hint_type] = {
                    "usage": usage,
                    "estimated_cost": cost,
                }
                elapsed = time.monotonic() - request_start
                total_requests += 1
                total_seconds += elapsed
                if cost["available"]:
                    print(
                        f"\n[{hint_type}] generated in {elapsed:.2f}s "
                        f"cost=${cost['usd']:.6f} "
                        f"tokens={usage['prompt_tokens']}+{usage['completion_tokens']}"
                    )
                else:
                    print(
                        f"\n[{hint_type}] generated in {elapsed:.2f}s "
                        f"cost=N/A tokens={usage['prompt_tokens']}+{usage['completion_tokens']}"
                    )
                if args.debug:
                    raw_responses[hint_type] = raw_response
            except Exception as error:
                elapsed = time.monotonic() - request_start
                errors[hint_type] = str(error)
                if not args.debug:
                    raise
                print(
                    f"\n[{hint_type}] generation failed after {elapsed:.2f}s, "
                    "continuing because --debug is set."
                )
                print(errors[hint_type])
            time.sleep(0.25)

        result = {
            "model": args.model,
            "dataset_name": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "sample_index": sample_index,
            "sample": sample,
            "generated_hints": outputs,
            "cost_details": cost_details,
        }
        if errors:
            result["errors"] = errors
        if raw_responses:
            result["raw_responses"] = raw_responses

        for hint_type, text in outputs.items():
            print(f"\n{hint_type.upper()}\n" + "=" * 80)
            print(text)
        for hint_type, error in errors.items():
            print(f"\n{hint_type.upper()} ERROR\n" + "=" * 80)
            print(error)

        if not args.no_save:
            hint_label = args.hint_type
            output_path = output_dir / f"sample_{sample_index}_{hint_label}.json"
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(result, file, indent=2, ensure_ascii=False)
            print(f"\nSaved JSON output to: {output_path}")

        sample_elapsed = time.monotonic() - sample_start
        print(f"\nSample {sample_index} finished in {sample_elapsed:.2f}s")
        sample_cost = sum(
            details["estimated_cost"]["usd"] or 0.0
            for details in cost_details.values()
        )
        if cost_details:
            print(f"Sample {sample_index} estimated cost: ${sample_cost:.6f}")

    if total_requests:
        avg_seconds = total_seconds / total_requests
        print("\nGENERATION SPEED SUMMARY\n" + "=" * 80)
        print(f"successful_requests={total_requests}")
        print(f"total_request_time={total_seconds:.2f}s")
        print(f"avg_request_time={avg_seconds:.2f}s")


if __name__ == "__main__":
    main()
