# A11yn: Aligning LLMs for Web Accessibility-Aware UI Generation (COLM 2026)

This is the official repository for our COLM 2026 paper **"A11yn: Aligning LLMs for Web Accessibility-Aware UI Generation."** In this work, we introduce A11yn (pronounced *align*), a post-training framework that aligns code-generating LLMs to produce web UIs with fewer WCAG violations in a single generation pass.

We release our training pipeline, Web Accessibility reward, and the UIReq-6.8K and RealUIReq-300 datasets in this repository.

## News

- **2026:** Our paper was accepted to the Conference on Language Modeling (COLM 2026).

## Overview

We turn automated WCAG audit results into a verifiable reinforcement-learning signal. For each generated UI, we use Axe-core to count affected DOM nodes at four severity levels and normalize the severity-weighted penalty by DOM size. This discourages inaccessible outputs without rewarding trivially small pages:

```text
reward = 1 - (0.1 * minor + 0.2 * moderate + 0.3 * serious + 0.4 * critical) / DOM size
```

We use this reward with Group Relative Policy Optimization (GRPO) to align Qwen2.5-Coder-7B-Instruct for accessibility-aware HTML generation.

## Highlights

- **Single-pass generation:** We align the model during post-training rather than relying on iterative repair.
- **Auditor-guided reward:** We convert Axe-core violations into an automatic, severity-aware training signal.
- **Size-aware optimization:** We normalize by DOM size to reduce reward hacking through overly simple interfaces.
- **Two released datasets:** We provide 6,800 training instructions and 300 realistic evaluation requests.

## Data

| Dataset | Split | Description | File |
| --- | ---: | --- | --- |
| UIReq-6.8K | 6,800 | Diverse, instruction-only web UI requests spanning 68 application categories | [`data/UIReq6.8K/uireq6800.json`](data/UIReq6.8K/uireq6800.json) |
| RealUIReq-300 | 300 | Human-refined, real-page-grounded requests for accessibility and UI-quality evaluation | [`data/RealUIReq300/realuirequest300.json`](data/RealUIReq300/realuirequest300.json) |

We use UIReq-6.8K for training and RealUIReq-300 as the primary evaluation benchmark in our paper.

## Setup

Our training stack requires Linux, CUDA-capable NVIDIA GPUs, and a recent Python 3 environment. We trained the model on four NVIDIA RTX PRO 6000 Blackwell GPUs with 96 GB VRAM each.

```bash
git clone https://github.com/jeffrobot/A11yn.git
cd A11yn

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
playwright install chromium
```

Weights & Biases logging is enabled by default. Authenticate with `wandb login`, or set `WANDB_MODE=offline`. Set `HUGGINGFACE_TOKEN` or `HF_TOKEN` if your model access requires authentication.

## Training

To reproduce our paper-scale four-GPU configuration:

```bash
GPU_IDS=0,1,2,3 \
VLLM_GPU_MEMORY_UTILIZATION=0.60 \
bash a11yn_train.sh
```

Our launcher defaults to `Qwen/Qwen2.5-Coder-7B-Instruct`, UIReq-6.8K, four GPUs, eight generations per prompt, and output under `outputs/a11yn`.

Common overrides:

```bash
GPU_IDS=0,1 \
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Coder-7B-Instruct \
DATASET_NAME=data/UIReq6.8K/uireq6800.json \
OUTPUT_DIR=outputs/a11yn \
bash a11yn_train.sh
```

The effective training batch size must be divisible by `NUM_GENERATIONS`; the launcher checks this before training. Resume an interrupted run with `RESUME_FROM_CHECKPOINT=/path/to/checkpoint`.

The accessibility reward renders generated HTML in headless Chromium. External network requests are blocked by default, and the vendored Tailwind stylesheet is served locally. Keep `A11YN_BLOCK_EXTERNAL_REQUESTS=1` when evaluating untrusted model output.

## Main Results

We report the following results on RealUIReq-300. Lower is better except for Lighthouse.

| Model | Weighted Violation Score | Lighthouse | IR (Axe) | IR (AChecker) |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-7B-Instruct | 4,479 | 90.88 | 0.40 | 0.62 |
| **A11yn** | **662** | **97.68** | **0.05** | **0.26** |

Compared with the base model, our A11yn model reduces the Axe inaccessibility rate by **87.5%** and the independently measured AChecker rate by **58.1%**, while preserving comparable semantic and visual quality.

## Repository Structure

```text
A11yn_train.py            GRPO training entry point
accessibility_reward.py  Axe-core reward and HTML evaluation
a11yn_train.sh           Reproducible distributed launcher
deepspeed_zero3.yaml     Accelerate/DeepSpeed configuration
data/                    UIReq-6.8K and RealUIReq-300
axe.min.js               Vendored Axe-core runtime
vendor/                  Vendored Tailwind stylesheet
```

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{yoon2026a11yn,
  title     = {A11yn: Aligning LLMs for Web Accessibility-Aware UI Generation},
  author    = {Yoon, Janghan and Cho, Jaegwan and Kim, Junhyeok and Chung, Jiwan and Jeon, Jaehyun and Lim, Seungwon and Yu, Youngjae},
  booktitle = {Conference on Language Modeling},
  year      = {2026}
}
```
