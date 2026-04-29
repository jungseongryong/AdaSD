import os
from dataclasses import dataclass, field

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from data_collator import SelfDistillationDataCollator
from opsd_trainer import OPSDTrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class MedicalOPSDScriptArguments(ScriptArguments):
    dataset_name: str = field(
        default="FreedomIntelligence/medical-o1-reasoning-SFT",
        metadata={"help": "Hugging Face dataset name for medical reasoning OPSD."},
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
        metadata={"help": "Column containing the medical question."},
    )
    reasoning_column: str = field(
        default="Complex_CoT",
        metadata={"help": "Column containing the reasoning chain."},
    )
    response_column: str = field(
        default="Response",
        metadata={"help": "Column containing the final response."},
    )
    context_column: str | None = field(
        default=None,
        metadata={
            "help": "Optional dataset column used as privileged teacher context. "
            "If unset, context is built from reasoning_column + response_column."
        },
    )
    trajectory_column_source: str | None = field(
        default=None,
        metadata={
            "help": "Optional dataset column used as the off-policy trajectory source. "
            "If unset, trajectory is built from reasoning_column + response_column."
        },
    )
    assistant_format: str = field(
        default="plain",
        metadata={
            "help": "Format for built context/trajectory: 'plain' uses reasoning plus response; "
            "'qwen_think' wraps reasoning in <think> tags; 'response_only' uses only Response."
        },
    )
    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use sampled-token policy-gradient-style loss. Not allowed with --off_policy."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy as a fixed teacher. Requires --use_peft."
        },
    )
    run_config: str | None = field(
        default=None,
        metadata={"help": "Run name suffix for output directory and WandB."},
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty used by vLLM generation in on-policy mode."},
    )
    reason_first: bool = field(
        default=False,
        metadata={"help": "Generate a teacher analysis of the privileged context before distillation."},
    )
    top_k_loss: int = field(
        default=0,
        metadata={"help": "Restrict JSD loss to teacher top-k tokens. 0 means full vocabulary."},
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={"help": "Clip each token's JSD loss to this value. 0 disables clipping."},
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={"help": "Use an EMA copy of trainable weights as the teacher."},
    )
    ema_decay: float = field(
        default=0.999,
        metadata={"help": "EMA teacher decay. Only used with --use_ema_teacher."},
    )
    off_policy: bool = field(
        default=False,
        metadata={
            "help": "Use dataset trajectories for self-distillation instead of sampling from the student."
        },
    )
    final_instruction: str = field(
        default="Please reason step by step and provide the final medical answer clearly.",
        metadata={"help": "Instruction appended to the medical question in student and teacher prompts."},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": "WandB entity. Leave unset to use the default logged-in entity."},
    )
    wandb_project: str = field(
        default="medical-opsd",
        metadata={"help": "WandB project name."},
    )


def _build_reasoning_response(example, args):
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


def make_medical_opsd_format_fn(script_args):
    def format_example(example):
        built_solution = _build_reasoning_response(example, script_args)
        teacher_context = (
            str(example[script_args.context_column]).strip()
            if script_args.context_column
            else built_solution
        )
        trajectory = (
            str(example[script_args.trajectory_column_source]).strip()
            if script_args.trajectory_column_source
            else built_solution
        )

        return {
            "problem": str(example[script_args.question_column]).strip(),
            "solution": built_solution,
            "teacher_context": teacher_context,
            "trajectory": trajectory,
        }

    return format_example


def resolve_model_dtype(model_args):
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
            return dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        return model_args.torch_dtype
    if hasattr(model_args, "dtype") and model_args.dtype is not None:
        return model_args.dtype
    return torch.bfloat16


if __name__ == "__main__":
    parser = TrlParser((MedicalOPSDScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires use_peft=True.")
    if script_args.off_policy and script_args.use_tinker_loss:
        raise ValueError(
            "off_policy=True currently supports the full-vocabulary JSD/KL objective only. "
            "Disable --use_tinker_loss for medical off-policy self-distillation."
        )

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )
    model_name = model_args.model_name_or_path.rstrip("/").split("/")[-1]

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        mode = "offpolicy" if script_args.off_policy else "opsd"
        full_wandb_run_name = (
            f"medical_{mode}_{model_name}_"
            f"{script_args.dataset_config or 'default'}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )
        if script_args.fixed_teacher:
            full_wandb_run_name += "_fixteach"

    print(f"\n{'='*80}")
    print("MEDICAL OPSD RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Dataset: {script_args.dataset_name} ({script_args.dataset_config})")
    print(f"Mode: {'off-policy' if script_args.off_policy else 'on-policy'}")
    print(f"{'='*80}\n")

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
                "context_column": script_args.context_column,
                "trajectory_column_source": script_args.trajectory_column_source,
                "assistant_format": script_args.assistant_format,
                "off_policy": script_args.off_policy,
                "learning_rate": training_args.learning_rate,
                "effective_batch_size": effective_batch_size,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "jsd_token_clip": script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
            },
        )

    model_dtype = resolve_model_dtype(model_args)
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
    training_args.presence_penalty = script_args.presence_penalty

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
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
    for optional_column in [script_args.context_column, script_args.trajectory_column_source]:
        if optional_column:
            required_columns.append(optional_column)
    missing_columns = [column for column in required_columns if column not in dataset.column_names]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )

    train_dataset = dataset.map(
        make_medical_opsd_format_fn(script_args),
        remove_columns=dataset.column_names,
    )

    data_collator = SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        reason_first=script_args.reason_first,
        off_policy=script_args.off_policy,
        teacher_context_column="teacher_context",
        trajectory_column="trajectory",
        final_instruction=script_args.final_instruction,
        reason_first_prompt=(
            "\n\nThe reference medical reasoning above leads to the target answer. "
            "Analyze the clinical reasoning steps and key medical evidence. "
            "Do NOT use <think> tags. Do NOT produce a new final answer yet.\n"
        ),
        transition_prompt=(
            "\n\nAfter studying the reference medical reasoning above, use your own clinical reasoning "
            "to answer the original medical question. Do not copy the reference verbatim:\n"
        ),
    )

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        off_policy=script_args.off_policy,
        teacher_context_column="teacher_context",
        trajectory_column="trajectory",
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)
