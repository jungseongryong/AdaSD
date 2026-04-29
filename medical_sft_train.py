import os
from dataclasses import dataclass, field

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

    import torch

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

    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
