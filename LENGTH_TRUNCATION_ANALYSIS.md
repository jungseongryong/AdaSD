# Length And Truncation Analysis

This document summarizes how sequence length is handled in SFT, on-policy OPSD, and the newly added off-policy self-distillation path.

## Actual Training Length Settings

The repository scripts use these length settings:

```text
scripts/run_sft.sh:
  --max_length 16000

scripts/run_opsd_1b.sh / run_opsd_4b.sh / run_opsd_8b.sh:
  --max_length 20000
  --max_completion_length 1024
```

For the medical examples added later, the suggested starting points are:

```text
medical SFT:
  --max_length 16000

medical off-policy OPSD:
  --max_length 4096 is enough for the current medical dataset
```

## What Gets Truncated

Off-policy self-distillation constructs two full inputs:

```text
student_input = student_prompt + reference_trajectory
teacher_input = teacher_prompt + reference_trajectory
```

The two parts have different roles:

```text
student_prompt:
  problem + student instruction

teacher_prompt:
  problem + privileged teacher context + transition instruction

reference_trajectory:
  dataset trajectory used as the loss target
```

Therefore there are two different truncation types.

### Prompt Truncation

Prompt truncation happens first.

`data_collator.py` tokenizes prompts with:

```python
truncation=True
max_length=self.max_length
```

If the prompt alone is longer than `max_length`, the prompt is cut from the end.

For the teacher, the prompt has this conceptual structure:

```text
Problem
+ teacher_context / reference solution
+ transition instruction
+ final instruction
+ assistant start tokens
```

If this is too long, the later part of the teacher prompt can be removed. This can damage the teacher condition because the teacher may lose part of the privileged context or instruction.

### Trajectory Truncation

After prompt lengths are known, the collator computes the remaining trajectory budget:

```text
trajectory_budget = max_length - max(student_prompt_length, teacher_prompt_length)
```

The trajectory is then tokenized with:

```python
truncation=True
max_length=trajectory_budget
add_special_tokens=False
```

If the trajectory is longer than this budget, the trajectory is cut from the end.

This matters because the OPSD loss is computed only on the kept trajectory tokens. If the trajectory is:

```text
Complex_CoT + Response
```

then truncation can remove the later part, often including the final answer/response.

## Why Prompt Truncation And Trajectory Truncation Are Counted Separately

They affect different parts of training:

```text
teacher prompt truncation:
  the teacher may lose context or instructions

trajectory truncation:
  some target/loss tokens disappear
```

For off-policy self-distillation, trajectory truncation is especially important because the whole method matches student and teacher distributions on the dataset trajectory.

## Previous Overflow Bug

The earlier implementation used:

```python
max_trajectory_len = max(1, self.max_length - max_prompt_len_for_trajectory)
```

This caused a bug when the prompt already filled the whole budget.

Example:

```text
max_length = 2048
teacher_prompt_length = 2048
remaining trajectory budget = 0
```

The old code forced:

```text
max_trajectory_len = 1
```

so the final teacher input became:

```text
teacher_input length = 2048 + 1 = 2049
```

This violates the intended `max_length` bound.

## Current Safety Change

`data_collator.py` now computes:

```python
max_trajectory_len = self.max_length - max_prompt_len_for_trajectory

if max_trajectory_len <= 0:
    raise ValueError(...)
```

So if no room is left for the off-policy reference trajectory, training stops with a clear error instead of silently creating an oversized sequence.

The collator also warns once per collator instance when:

```text
student prompt truncation occurs
teacher prompt truncation occurs
reason-first teacher prompt truncation occurs
off-policy reference trajectory truncation occurs
```

For trajectory truncation, the batch also includes:

```text
reference_trajectory_full_lengths_per_example
reference_trajectory_truncated_per_example
```

These make it possible to inspect which examples were shortened.

## Full Dataset Analysis

The following analysis used the `Qwen/Qwen3-1.7B` tokenizer and the current off-policy formatting.

Datasets:

```text
Math:
  siyanzhao/Openthoughts_math_30k_opsd
  total samples = 29,434

Medical:
  FreedomIntelligence/medical-o1-reasoning-SFT
  config = en
  total samples = 19,704
```

### Math Dataset

Overall lengths:

```text
student prompt:
  p50=93, p90=157, p95=184, p99=283, max=1771

teacher prompt:
  p50=823, p90=1232, p95=1366, p99=1692, max=3304

trajectory:
  p50=631, p90=1023, p95=1148, p99=1450, max=2888

teacher prompt + trajectory:
  p50=1454, p90=2251, p95=2511, p99=3134, max=6102
```

Truncation by `max_length`:

| max_length | teacher prompt truncated | trajectory truncated | no room for trajectory | overflow in old code |
|---:|---:|---:|---:|---:|
| 2048 | 43 / 29,434, 0.15% | 4,872 / 29,434, 16.55% | 43 / 29,434, 0.15% | 43 / 29,434, 0.15% |
| 4096 | 0 | 21 / 29,434, 0.07% | 0 | 0 |
| 8192 | 0 | 0 | 0 | 0 |
| 16000 | 0 | 0 | 0 | 0 |
| 20000 | 0 | 0 | 0 | 0 |

Longest observed math off-policy teacher input:

```text
teacher_prompt = 3304
trajectory = 2798
teacher_full = 6102
```

Conclusion:

```text
For current math off-policy training, max_length=8192 avoids all truncation.
max_length=4096 is almost enough but truncates 21 trajectories.
max_length=2048 is not recommended.
```

### Medical Dataset

Overall lengths:

```text
student prompt:
  p50=82, p90=143, p95=200, p99=322, max=827

teacher prompt:
  p50=665, p90=869, p95=942, p99=1096, max=1513

trajectory:
  p50=528, p90=703, p95=768, p99=917, max=1362

teacher prompt + trajectory:
  p50=1196, p90=1569, p95=1697, p99=1984, max=2839
```

Truncation by `max_length`:

| max_length | teacher prompt truncated | trajectory truncated | no room for trajectory | overflow in old code |
|---:|---:|---:|---:|---:|
| 2048 | 0 | 127 / 19,704, 0.64% | 0 | 0 |
| 4096 | 0 | 0 | 0 | 0 |
| 8192 | 0 | 0 | 0 | 0 |
| 16000 | 0 | 0 | 0 | 0 |
| 20000 | 0 | 0 | 0 | 0 |

Longest observed medical off-policy teacher input:

```text
teacher_prompt = 1485
trajectory = 1354
teacher_full = 2839
```

Conclusion:

```text
For current medical off-policy training, max_length=4096 avoids all truncation.
max_length=2048 is usable but truncates 127 trajectories.
```

## Recommended Settings

For the current datasets:

```text
medical off-policy:
  --max_length 4096

math off-policy:
  --max_length 8192

mixed or extra-safe:
  --max_length 8192

very conservative:
  --max_length 20000
```

`max_length=20000` is safe but can be unnecessarily large. The current collator pads to the maximum needed length inside the batch, not blindly to 20000 for every batch, but allowing very long examples can still increase memory and time when such examples appear.

## Future Long Teacher Contexts

The current datasets fit comfortably. However, future off-policy datasets may provide a separate, much longer `teacher_context`.

In that case, the correct priority should be:

```text
1. preserve enough trajectory tokens for loss
2. truncate or summarize teacher_context first
3. filter examples with no trajectory budget
4. only truncate trajectory when acceptable
```

A future robust design could add:

```text
--max_teacher_context_length
--min_trajectory_length
--filter_overlong_examples
```

For now, the current warning/error behavior prevents silent overflow and makes truncation visible during training.
