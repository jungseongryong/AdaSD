# Implementation Summary

This document summarizes all changes and checks made in this repository during the OPSD/off-policy/medical SFT work.

## Scope

The work covered four related goals:

1. Add an off-policy self-distillation option to the original OPSD code.
2. Keep the original math OPSD/SFT behavior intact.
3. Add medical-o1 SFT support.
4. Add medical-o1 OPSD and off-policy self-distillation support.

The implementation was designed so that the original code path remains available and unchanged by default.

## Repository State

The original repository was cloned from:

```text
https://github.com/siyan-zhao/OPSD
```

It was then pushed to:

```text
https://github.com/jungseongryong/AdaSD
```

The main off-policy commit pushed earlier was:

```text
ecf6c60 Add off-policy self-distillation option
```

Additional local files and modifications were created after that commit and may still need to be committed/pushed.

## Files Added

### `medical_sft_train.py`

Adds SFT support for:

```text
FreedomIntelligence/medical-o1-reasoning-SFT
```

Default dataset config:

```text
en
```

Default input columns:

```text
Question
Complex_CoT
Response
```

Default SFT mapping:

```text
user      = Question
assistant = Complex_CoT + "\n\n" + Response
```

The script also supports:

```bash
--assistant_format plain
--assistant_format qwen_think
--assistant_format response_only
```

Meaning:

```text
plain:
Complex_CoT

Response

qwen_think:
<think>
Complex_CoT
</think>

Response

response_only:
Response
```

### `medical_opsd_train.py`

Adds medical OPSD and medical off-policy self-distillation support.

Default dataset:

```text
FreedomIntelligence/medical-o1-reasoning-SFT
```

Default config:

```text
en
```

Default raw dataset columns:

```text
Question
Complex_CoT
Response
```

The script converts those columns into the format expected by `SelfDistillationDataCollator`:

```text
problem         = Question
solution        = Complex_CoT + "\n\n" + Response
teacher_context = Complex_CoT + "\n\n" + Response
trajectory      = Complex_CoT + "\n\n" + Response
```

For medical off-policy distillation:

```text
student prompt = Question + medical instruction
teacher prompt = Question + teacher_context + medical transition
trajectory     = trajectory
loss           = distribution matching on trajectory tokens
```

The default medical instruction is:

```text
Please reason step by step and provide the final medical answer clearly.
```

This avoids the original math-specific `\boxed{}` instruction.

### `OFF_POLICY_CHANGES.md`

Detailed explanation of the off-policy self-distillation changes.

It covers:

- why off-policy was added as an option
- how `teacher_context_column` and `trajectory_column` work
- how `data_collator.py`, `opsd_train.py`, and `opsd_trainer.py` changed
- how off-policy differs from SFT
- caveats around teacher context leakage
- validation performed

### `FORMAT_SAMPLE_CHECK.md`

Generated inspection file showing actual rendered Qwen3 chat-template text for one sample from each dataset.

Datasets used:

```text
siyanzhao/Openthoughts_math_30k_opsd
FreedomIntelligence/medical-o1-reasoning-SFT
```

Tokenizer used:

```text
Qwen/Qwen3-1.7B
```

The file shows examples for:

```text
Math SFT
Math OPSD on-policy
Math off-policy self-distillation
Medical SFT
Medical OPSD on-policy
Medical off-policy self-distillation
```

Important note:

The `[TRUNCATED]` markers in `FORMAT_SAMPLE_CHECK.md` are document display truncation only. They are not training-time tokenizer truncation.

## Files Modified

### `opsd_train.py`

Added CLI arguments for off-policy self-distillation:

```python
off_policy: bool = False
teacher_context_column: str = "solution"
trajectory_column: str = "solution"
```

These are passed into `OPSDTrainer`.

Also added validation:

```python
if script_args.off_policy and script_args.use_tinker_loss:
    raise ValueError(...)
```

Reason:

`use_tinker_loss` is a sampled-token policy-gradient-style loss. It is naturally tied to on-policy sampled trajectories. For off-policy dataset trajectories, the implementation currently supports only the full-vocabulary JSD/KL objective.

### `opsd_trainer.py`

Added trainer arguments:

```python
off_policy: bool = False
teacher_context_column: str = "solution"
trajectory_column: str = "solution"
```

When `off_policy=True`, `training_step()` skips student generation and uses dataset trajectories from the collator.

On-policy path:

```text
student generates trajectory
student_input = student_prompt + generated_trajectory
teacher_input = teacher_prompt + generated_trajectory
```

Off-policy path:

```text
trajectory = reference_trajectory_ids from dataset
student_input = student_prompt + trajectory
teacher_input = teacher_prompt + trajectory
```

The same `compute_loss()` is reused.

This works because `compute_loss()` already slices:

```python
student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]
```

Those slices correspond to the trajectory tokens in both on-policy and off-policy modes.

### `data_collator.py`

Extended the existing collator without removing the original math behavior.

New optional constructor arguments:

```python
final_instruction=None
reason_first_prompt=None
transition_prompt=None
```

Default behavior remains math-compatible:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

Medical code passes a different instruction:

```text
Please reason step by step and provide the final medical answer clearly.
```

Also added support for configurable teacher context and trajectory columns:

```python
teacher_context_column="solution"
trajectory_column="solution"
```

In off-policy mode, the collator tokenizes:

```text
reference_trajectory_ids
reference_trajectory_attention_mask
reference_trajectory_lengths_per_example
```

These are appended to both student and teacher inputs by `OPSDTrainer.training_step()`.

## Original Code Preservation

The original math entrypoints remain:

```text
opsd_train.py
sft_train.py
```

The new medical entrypoints are separate:

```text
medical_opsd_train.py
medical_sft_train.py
```

The original math dataset remains hardcoded in `opsd_train.py`:

```python
load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
```

The original math SFT dataset remains hardcoded in `sft_train.py`.

The shared collator was extended with default-preserving options. If no medical options are passed, it still uses the original math-style prompt.

## Data Formatting Summary

### Math SFT

Input:

```text
problem
solution
```

Rendered concept:

```text
<user>
problem

Please reason step by step, and put your final answer within \boxed{}.

<assistant>
solution
```

### Math OPSD On-Policy

Student prompt:

```text
Problem: problem

Please reason step by step, and put your final answer within \boxed{}.
```

Teacher prompt:

```text
Problem: problem

Here is a reference solution to this problem:
=== Reference Solution Begin ===
solution
=== Reference Solution End ===

transition prompt

Please reason step by step, and put your final answer within \boxed{}.
```

Trajectory:

```text
student-generated completion
```

Loss is computed on:

```text
student-generated completion tokens
```

### Math Off-Policy Self-Distillation

Student input:

```text
student_prompt + solution
```

Teacher input:

```text
teacher_prompt + solution
```

Loss is computed on:

```text
solution tokens
```

### Medical SFT

Input:

```text
Question
Complex_CoT
Response
```

Default rendered concept:

```text
<user>
Question

<assistant>
Complex_CoT

Response
```

With:

```bash
--assistant_format qwen_think
```

the assistant target becomes:

```text
<think>
Complex_CoT
</think>

Response
```

### Medical OPSD On-Policy

Student prompt:

```text
Problem: Question

Please reason step by step and provide the final medical answer clearly.
```

Teacher prompt:

```text
Problem: Question

Here is a reference solution to this problem:
=== Reference Solution Begin ===
Complex_CoT

Response
=== Reference Solution End ===

After studying the reference medical reasoning above, use your own clinical reasoning
to answer the original medical question. Do not copy the reference verbatim:

Please reason step by step and provide the final medical answer clearly.
```

Trajectory:

```text
student-generated completion
```

Loss is computed on:

```text
student-generated completion tokens
```

### Medical Off-Policy Self-Distillation

Student input:

```text
student_prompt + Complex_CoT + Response
```

Teacher input:

```text
teacher_prompt + Complex_CoT + Response
```

Loss is computed on:

```text
Complex_CoT + Response tokens
```

## Qwen3 Chat Template Observation

Actual rendered Qwen3 text showed that Qwen3 inserts an empty think block by default in assistant messages:

```text
<|im_start|>assistant
<think>

</think>

...
```

This means that with `assistant_format=plain`, the reasoning text is not inside the `<think>` block. It appears after the empty think block as normal assistant text.

If the desired behavior is to train Qwen3-style thinking traces, use:

```bash
--assistant_format qwen_think
```

This makes the target:

```text
<think>
Complex_CoT
</think>

Response
```

## Truncation Analysis

There are two kinds of truncation:

1. Display truncation in `FORMAT_SAMPLE_CHECK.md`.
2. Actual tokenizer/model truncation during training.

Display truncation is only for readability and does not affect training.

### First-Sample Token Lengths

Using `Qwen/Qwen3-1.7B` tokenizer:

```text
math_sft_full_text: 1622
math_opsd_student_prompt: 164
math_opsd_teacher_prompt: 1710
math_offpolicy_trajectory: 1460
math_offpolicy_student_full: 1623
math_offpolicy_teacher_full: 3169

medical_sft_plain_full_text: 553
medical_sft_qwen_think_full_text: 553
medical_opsd_student_prompt: 76
medical_opsd_teacher_prompt_plain: 611
medical_offpolicy_trajectory_plain: 491
medical_offpolicy_student_full_plain: 567
medical_offpolicy_teacher_full_plain: 1102
```

The first samples fit comfortably under both `max_length=16000` and `max_length=20000`.

### Why Off-Policy Needed Extra Care

On-policy OPSD uses generated trajectories:

```text
trajectory length <= max_completion_length
```

For example:

```text
max_completion_length = 1024
```

Off-policy uses dataset trajectories:

```text
trajectory = solution
```

or:

```text
trajectory = Complex_CoT + Response
```

These can be much longer and vary by sample.

Teacher input can also contain the reference solution/context once in the prompt and again as the matched trajectory:

```text
teacher_input = problem + reference context + trajectory
```

Therefore off-policy needs stricter length budgeting.

### Length-Budget Fix

Originally, off-policy trajectory length was budgeted against the student prompt:

```python
max_trajectory_len = max_length - max_student_prompt_len
```

This was not sufficient because teacher prompts are usually longer.

It was changed to:

```python
max_prompt_len_for_trajectory = max(
    result["student_prompt_length"],
    result["teacher_prompt_length"],
)
max_trajectory_len = max(1, self.max_length - max_prompt_len_for_trajectory)
```

This ensures:

```text
student_prompt + trajectory <= max_length
teacher_prompt + trajectory <= max_length
```

## Collator Checks Performed

A no-training check was run using:

```text
Qwen/Qwen3-1.7B tokenizer
max_length = 2048
off_policy = True
first 8 math samples
first 8 medical samples
```

### Math Off-Policy Length Check

Output:

```text
batch_size=8 max_length=2048
student_prompt_len=164
teacher_prompt_len=1710
trajectory_padded_len=338
student_full_len=502 <= max_length: True
teacher_full_len=2048 <= max_length: True
trajectory_lengths_per_example=[338, 338, 338, 338, 338, 333, 338, 338]
```

Interpretation:

The math teacher prompt was long, so trajectory was truncated to fit the teacher full input exactly into `2048` tokens.

### Medical Off-Policy Length Check

Output:

```text
batch_size=8 max_length=2048
student_prompt_len=132
teacher_prompt_len=871
trajectory_padded_len=741
student_full_len=873 <= max_length: True
teacher_full_len=1612 <= max_length: True
trajectory_lengths_per_example=[491, 530, 441, 741, 555, 466, 513, 626]
```

Interpretation:

Medical first samples fit comfortably under the test budget.

## Loss Alignment Check

The most important check was whether loss is computed on the same tokens for student and teacher.

The check reconstructed the tensors used by `training_step()` and `compute_loss()`:

```text
sampled_token_ids
shifted_labels
teacher_trajectory_ids
reference_trajectory_ids
```

It verified:

```text
sampled_token_ids == reference_trajectory_ids
teacher_trajectory_ids == reference_trajectory_ids
shifted_labels active tokens == reference_trajectory_ids non-padding tokens
padding labels == -100
```

### Math Loss Alignment Check

Output:

```text
sampled_token_ids shape: (8, 338)
shifted_labels shape: (8, 338)
teacher_trajectory_ids shape: (8, 338)
reference_trajectory_ids shape: (8, 338)
sample 0: loss_tokens=338, trajectory_len=338, first_loss_token='\n'
sample 1: loss_tokens=338, trajectory_len=338, first_loss_token='Given'
sample 2: loss_tokens=338, trajectory_len=338, first_loss_token='Given'
sample 3: loss_tokens=338, trajectory_len=338, first_loss_token='\n'
sample 4: loss_tokens=338, trajectory_len=338, first_loss_token='\n'
sample 5: loss_tokens=333, trajectory_len=333, first_loss_token='\n'
sample 6: loss_tokens=338, trajectory_len=338, first_loss_token='\n'
sample 7: loss_tokens=338, trajectory_len=338, first_loss_token='To'
Loss alignment checks passed.
```

### Medical Loss Alignment Check

Output:

```text
sampled_token_ids shape: (8, 741)
shifted_labels shape: (8, 741)
teacher_trajectory_ids shape: (8, 741)
reference_trajectory_ids shape: (8, 741)
sample 0: loss_tokens=491, trajectory_len=491, first_loss_token='Okay'
sample 1: loss_tokens=530, trajectory_len=530, first_loss_token='Okay'
sample 2: loss_tokens=441, trajectory_len=441, first_loss_token='Okay'
sample 3: loss_tokens=741, trajectory_len=741, first_loss_token='Alright'
sample 4: loss_tokens=555, trajectory_len=555, first_loss_token='Okay'
sample 5: loss_tokens=466, trajectory_len=466, first_loss_token='I'
sample 6: loss_tokens=513, trajectory_len=513, first_loss_token='Okay'
sample 7: loss_tokens=626, trajectory_len=626, first_loss_token='Okay'
Loss alignment checks passed.
```

Conclusion:

For the checked samples, loss is computed on the same trajectory tokens for:

```text
student
teacher
labels
reference trajectory
```

## Example Commands

### Medical SFT

```bash
accelerate launch \
  --config_file accelerate.yaml \
  --num_processes 4 \
  medical_sft_train.py \
  --model_name_or_path /data0/shared/Qwen3-1.7B \
  --dataset_name FreedomIntelligence/medical-o1-reasoning-SFT \
  --dataset_config en \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --output_dir /data0/siyanz/medical-sft \
  --run_config qwen31b_medical_o1_sft \
  --num_train_epochs 3 \
  --gradient_checkpointing \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_length 16000 \
  --logging_steps 5 \
  --save_steps 100 \
  --torch_dtype bfloat16 \
  --assistant_format qwen_think \
  --wandb_project medical-sft
```

### Medical Off-Policy Self-Distillation

```bash
accelerate launch \
  --config_file accelerate.yaml \
  --num_processes 4 \
  medical_opsd_train.py \
  --model_name_or_path /data0/shared/Qwen3-1.7B \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing \
  --output_dir /data0/siyanz/medical-opsd \
  --run_config qwen31b_medical_offpolicy \
  --num_train_epochs 3 \
  --max_completion_length 1024 \
  --max_length 20000 \
  --beta 0 \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --fixed_teacher \
  --jsd_token_clip 0.05 \
  --off_policy \
  --assistant_format qwen_think \
  --wandb_project medical-opsd
```

### Medical On-Policy OPSD

For on-policy medical OPSD, remove:

```bash
--off_policy
```

and add generation infrastructure, for example:

```bash
--use_vllm \
--vllm_mode colocate \
--vllm_gpu_memory_utilization 0.6 \
--vllm_tensor_parallel_size 1
```

## Temporary Inspection Environment

A local lightweight environment was created for inspection only:

```text
.venv-inspect/
```

It installed:

```text
datasets
transformers
torch
accelerate
sentencepiece
tiktoken
```

This environment was used only to:

- load sample dataset rows
- load the Qwen3 tokenizer
- render chat templates
- run collator alignment checks

It was not used for model training.

This directory should not be committed.

## Current Caution Items

1. `.venv-inspect/` should be ignored or removed before committing.
2. `FORMAT_SAMPLE_CHECK.md` is useful for inspection but may be too verbose for a clean repository.
3. `assistant_format=qwen_think` is likely preferable for Qwen3 medical reasoning if the goal is to train explicit thinking traces.
4. Off-policy teacher context currently defaults to the same text as trajectory. This is intentional for now, but future datasets can separate:

```text
teacher_context
trajectory
```

by using different source columns.
