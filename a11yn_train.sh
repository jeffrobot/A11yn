#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pick the GPU list from the environment when provided, otherwise use the default set.
DEFAULT_GPU_IDS="0,1,2,3"
if [[ -n "${GPU_IDS:-}" ]]; then
    SELECTED_GPU_IDS="$GPU_IDS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    SELECTED_GPU_IDS="$CUDA_VISIBLE_DEVICES"
else
    SELECTED_GPU_IDS="$DEFAULT_GPU_IDS"
fi

GPU_IDS="$SELECTED_GPU_IDS"
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
DATASET_NAME="${DATASET_NAME:-data/UIReq6.8K/uireq6800.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/a11yn}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-12}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"

# Validate the distributed launch settings before starting training.
if (( NUM_PROCESSES != ${#GPU_ID_ARRAY[@]} )); then
    echo "NUM_PROCESSES (${NUM_PROCESSES}) must match the number of GPU ids in GPU_IDS (${GPU_IDS})." >&2
    exit 1
fi

EFFECTIVE_TRAIN_BATCH=$((NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( EFFECTIVE_TRAIN_BATCH % NUM_GENERATIONS != 0 )); then
    echo "Effective train batch size (${EFFECTIVE_TRAIN_BATCH}) must be divisible by num_generations (${NUM_GENERATIONS})." >&2
    exit 1
fi

# Export runtime settings shared by the training script and accessibility reward code.
if [[ -n "${GPU_IDS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200000}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export A11YN_BLOCK_EXTERNAL_REQUESTS="${A11YN_BLOCK_EXTERNAL_REQUESTS:-1}"
export A11YN_ALLOWED_EXTERNAL_HOSTS="${A11YN_ALLOWED_EXTERNAL_HOSTS:-}"
export A11YN_TAILWIND_CSS_URL="${A11YN_TAILWIND_CSS_URL:-https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css}"
export A11YN_LOCAL_TAILWIND_CSS_PATH="${A11YN_LOCAL_TAILWIND_CSS_PATH:-$SCRIPT_DIR/vendor/tailwind-2.2.19.min.css}"

echo "Launching A11yn with ${NUM_PROCESSES} GPU(s) on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Effective train batch: ${EFFECTIVE_TRAIN_BATCH}, num_generations: ${NUM_GENERATIONS}, vLLM gpu memory util: ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "Accessibility reward external requests blocked: ${A11YN_BLOCK_EXTERNAL_REQUESTS}"
echo "Accessibility reward Tailwind URL: ${A11YN_TAILWIND_CSS_URL}"
echo "Accessibility reward local Tailwind path: ${A11YN_LOCAL_TAILWIND_CSS_PATH}"
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    echo "Resuming from checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi

# Build the accelerate command explicitly so it is easy to print and override.
LAUNCH_CMD=(
    accelerate launch
    --config_file "$SCRIPT_DIR/deepspeed_zero3.yaml"
    --num_processes "$NUM_PROCESSES"
    --num_machines 1
    --main_process_port "$MAIN_PROCESS_PORT"
    --mixed_precision bf16
    --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS"
    "$SCRIPT_DIR/A11yn_train.py"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --dataset_name "$DATASET_NAME"
    --learning_rate 5e-5
    --weight_decay 0.1
    --adam_beta1 0.9
    --adam_beta2 0.99
    --lr_scheduler_type cosine
    --adam_epsilon 1e-08
    --warmup_ratio 0.01
    --output_dir "$OUTPUT_DIR"
    --beta 0.001
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --num_generations "$NUM_GENERATIONS"
    --max_prompt_length 1024
    --max_completion_length 3072
    --gradient_checkpointing
    --logging_strategy steps
    --bf16 True
    --bf16_full_eval True
    --eval_strategy steps
    --save_strategy steps
    --logging_steps 1
    --eval_steps 25
    --save_steps 25
    --load_best_model_at_end True
    --save_total_limit 15
    --report_to wandb
    --use_vllm
    --vllm_mode colocate
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --log_completions True
    --use_peft
    --loss_type grpo
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    LAUNCH_CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

# Print the exact command for reproducibility, then execute it.
printf 'Launch command:'
printf ' %q' "${LAUNCH_CMD[@]}"
printf '\n'

"${LAUNCH_CMD[@]}"
