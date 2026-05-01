#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LM_EVAL_BIN="${LM_EVAL_BIN:-/tmp/adasd-venv/bin/lm_eval}"
if [[ ! -x "$LM_EVAL_BIN" ]]; then
  LM_EVAL_BIN="lm_eval"
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/hf_5tasks_recent6_by_epoch}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a TASKS <<< "$TASKS_OVERRIDE"
else
  TASKS=(
    medmcqa
    medqa_4options
    pubmedqa
    mmlu
    leaderboard_instruction_following
  )
fi

MODEL_ARGS_BASE="pretrained=${BASE_MODEL},dtype=bfloat16,attn_implementation=eager,trust_remote_code=True"

declare -a CHECKPOINTS=(
  "ceonly-nll-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-308"
  "ceonly-nll-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-616"
  "ceonly-nll-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-924"
  "ceonly-nll-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-1232"
  "ceonly-nll-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs64_5ep/checkpoint-1540"
  "ceonly-nll-bs8-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-2463"
  "ceonly-nll-bs8-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-4926"
  "ceonly-nll-bs8-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-7389"
  "ceonly-nll-bs8-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-9852"
  "ceonly-nll-bs8-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-nll-bs8-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_nll_bs8_5ep/checkpoint-12315"
  "ceonly-dft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-308"
  "ceonly-dft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-616"
  "ceonly-dft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-924"
  "ceonly-dft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-1232"
  "ceonly-dft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-dft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_dft_bs64_5ep/checkpoint-1540"
  "ceonly-eaft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-308"
  "ceonly-eaft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-616"
  "ceonly-eaft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-924"
  "ceonly-eaft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-1232"
  "ceonly-eaft-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-ceonly-eaft-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_ceonly_eaft_bs64_5ep/checkpoint-1540"
  "offpolicy-kdonly-fkl-clip005-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-308"
  "offpolicy-kdonly-fkl-clip005-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-616"
  "offpolicy-kdonly-fkl-clip005-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-924"
  "offpolicy-kdonly-fkl-clip005-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-1232"
  "offpolicy-kdonly-fkl-clip005-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-clip005-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_clip005_bs64_5ep/checkpoint-1540"
  "offpolicy-kdonly-fkl-noclip-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-308"
  "offpolicy-kdonly-fkl-noclip-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-616"
  "offpolicy-kdonly-fkl-noclip-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-924"
  "offpolicy-kdonly-fkl-noclip-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-1232"
  "offpolicy-kdonly-fkl-noclip-bs64-5ep|outputs/medical-opsd/qwen3-1.7b-rewrite-hints-response-only-offpolicy-kdonly-fkl-noclip-bs64-5ep/qwen3_1_7b_rewrite_hints_response_only_offpolicy_kdonly_fkl_noclip_bs64_5ep/checkpoint-1540"
)

mkdir -p "$OUTPUT_ROOT"

echo "Evaluating ${#CHECKPOINTS[@]} checkpoints on tasks: ${TASKS[*]}"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for item in "${CHECKPOINTS[@]}"; do
  exp="${item%%|*}"
  peft="${item#*|}"
  ckpt="${peft##*/}"
  out_dir="${OUTPUT_ROOT}/${exp}/${ckpt}"
  done_file="${out_dir}/.done"
  failed_file="${out_dir}/.failed"

  if [[ ! -f "${peft}/adapter_model.safetensors" ]]; then
    echo "[missing] ${peft}/adapter_model.safetensors"
    exit 1
  fi

  if [[ -f "$done_file" ]]; then
    echo "[skip] ${exp}/${ckpt}"
    continue
  fi

  mkdir -p "$out_dir"
  rm -f "$failed_file"
  echo "[run] ${exp}/${ckpt}"
  if "$LM_EVAL_BIN" run \
    --model hf \
    --model_args "${MODEL_ARGS_BASE},peft=${peft}" \
    --tasks "${TASKS[@]}" \
    --batch_size auto \
    --device cuda:0 \
    --trust_remote_code \
    --output_path "$out_dir" \
    2>&1 | tee "${out_dir}/eval.log"; then
    touch "$done_file"
  else
    echo "[failed] ${exp}/${ckpt}" | tee "$failed_file"
  fi
done

echo "All evaluations finished."
