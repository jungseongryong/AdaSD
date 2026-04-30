import os
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class MedicalSFTScriptArguments(ScriptArguments):
    dataset_name: str = field(
        default="FreedomIntelligence/medical-o1-reasoning-SFT",
        metadata={"help": "Hugging Face dataset name for medical reasoning SFT."},
    )
    dataset_config: str | None = field(
        default="en",
        metadata={"help": "Dataset subset/config name. Use 'en' or another available dataset config."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to train on."},
    )
    question_column: str = field(
        default="Question",
        metadata={"help": "Column containing the user question."},
    )
    reasoning_column: str = field(
        default="Complex_CoT",
        metadata={"help": "Column containing the reasoning chain."},
    )
    response_column: str = field(
        default="Response",
        metadata={"help": "Column containing the final response."},
    )
    eval_size: float = field(
        default=0.01,
        metadata={"help": "Fraction of the train split held out for eval."},
    )
    run_config: str | None = field(
        default=None,
        metadata={"help": "Optional run name suffix for WandB and output directory."},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": "WandB entity. Leave unset to use the default logged-in entity."},
    )
    wandb_project: str = field(
        default="medical-sft",
        metadata={"help": "WandB project name."},
    )
    assistant_format: str = field(
        default="plain",
        metadata={
            "help": "Assistant target format: 'plain' uses reasoning plus response; "
            "'qwen_think' wraps reasoning in <think> tags; 'response_only' trains only on Response."
        },
    )
    loss_variant: str = field(
        default="trl",
        metadata={
            "help": "SFT loss implementation to use. 'trl' delegates to TRL SFTTrainer/SFTConfig "
            "including --loss_type nll or --loss_type dft. 'eaft' uses entropy-adaptive token weighting."
        },
    )


class EAFTSFTTrainer(SFTTrainer):
    """SFTTrainer with Entropy-Adaptive Fine-Tuning loss.

    EAFT scales each supervised token's cross-entropy by the model's normalized
    next-token entropy. The entropy weight is detached, i.e. used as a stop-gradient
    gate, so optimization still updates the model through the supervised CE term.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("EAFT loss requires labels in the trainer batch.")

        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        # Causal LM convention: logits at position t predict label at position t+1.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        valid_mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)

        # Use fp32 for stable entropy/log-prob calculations, especially under bf16 training.
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_ce = -torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_weight = (entropy / math.log(vocab_size)).clamp(min=0.0, max=1.0).detach()

        weighted_loss = token_ce * entropy_weight * valid_mask
        loss = weighted_loss.sum() / valid_mask.sum().clamp_min(1)

        if return_outputs:
            return loss, outputs
        return loss


def _build_assistant_content(example, args):
    reasoning = str(example.get(args.reasoning_column, "") or "").strip()
    response = str(example.get(args.response_column, "") or "").strip()

    if args.assistant_format == "plain":
        if reasoning and response:
            return f"{reasoning}\n\n{response}"
        return response or reasoning
    if args.assistant_format == "qwen_think":
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n\n{response}"
        return response
    if args.assistant_format == "response_only":
        return response

    raise ValueError(
        f"Unknown assistant_format={args.assistant_format!r}. "
        "Choose one of: plain, qwen_think, response_only."
    )


def make_format_fn(tokenizer, script_args):
    def format_example(example):
        question = str(example[script_args.question_column]).strip()
        assistant_content = _build_assistant_content(example, script_args)
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    return format_example


if __name__ == "__main__":
    parser = TrlParser((MedicalSFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    script_args.loss_variant = script_args.loss_variant.lower()
    if script_args.loss_variant not in {"trl", "eaft"}:
        raise ValueError("loss_variant must be one of: trl, eaft.")
    loss_name = "eaft" if script_args.loss_variant == "eaft" else getattr(training_args, "loss_type", "trl")

    model_name = model_args.model_name_or_path.rstrip("/").split("/")[-1]
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        full_wandb_run_name = (
            f"medical_sft_{model_name}_"
            f"{script_args.dataset_config or 'default'}_"
            f"{loss_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"ep{training_args.num_train_epochs}"
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "dataset_name": script_args.dataset_name,
                "dataset_config": script_args.dataset_config,
                "dataset_split": script_args.dataset_split,
                "question_column": script_args.question_column,
                "reasoning_column": script_args.reasoning_column,
                "response_column": script_args.response_column,
                "assistant_format": script_args.assistant_format,
                "loss_variant": script_args.loss_variant,
                "trl_loss_type": getattr(training_args, "loss_type", None),
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_seq_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
            },
        )

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.dataset_config:
        dataset = load_dataset(script_args.dataset_name, script_args.dataset_config, split=script_args.dataset_split)
    else:
        dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_split)

    required_columns = [
        script_args.question_column,
        script_args.reasoning_column,
        script_args.response_column,
    ]
    missing_columns = [column for column in required_columns if column not in dataset.column_names]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )

    train_dataset = dataset.map(make_format_fn(tokenizer, script_args))
    split_dataset = train_dataset.train_test_split(test_size=script_args.eval_size, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    trainer_cls = EAFTSFTTrainer if script_args.loss_variant == "eaft" else SFTTrainer
    if script_args.loss_variant == "eaft" and os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n" + "=" * 80)
        print("EAFT LOSS ENABLED")
        print("Loss = normalized_token_entropy.detach() * token_cross_entropy")
        print("Use --loss_variant trl with --loss_type nll/dft for TRL's built-in losses.")
        print("=" * 80 + "\n")

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
