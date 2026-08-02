from datasets import Dataset
import pandas as pd
from trl import GRPOConfig, GRPOTrainer, TrlParser, ScriptArguments, ModelConfig, get_quantization_config, get_peft_config
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import re
import os
from pathlib import Path
import accessibility_reward
import wandb
from huggingface_hub import login
from vllm import SamplingParams


def patch_wandb_working_set() -> None:
    """Make wandb's installed-package scan robust to broken package metadata."""
    from importlib.metadata import distributions
    from wandb.util import InstalledDistribution

    def safe_working_set():
        """Yield package name/version pairs while skipping malformed entries."""
        for dist in distributions():
            try:
                metadata = dist.metadata
                if metadata is None:
                    continue
                name = metadata.get("Name")
                if not name:
                    continue
                yield InstalledDistribution(key=name, version=dist.version)
            except (AttributeError, KeyError, TypeError, UnicodeDecodeError):
                continue

    wandb.util.working_set = safe_working_set


huggingface_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN", "")
if huggingface_token:
    login(token=huggingface_token)

# Apply the wandb metadata workaround before trainer initialization.
patch_wandb_working_set()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "data" / "UIReq6.8K" / "uireq6800.json"

SYSTEM_PROMPT = """
    You are an expert UI designer assistant.
    You should plan the design based on the user request. Show the plan in the `<think>` tag.
        - You must think about the html structure and widgets needed to fulfill the user request.
        - You must think about the Tailwind CSS classes to use for styling.
    Then, you should generate a complete HTML document that includes:
        - A `<head>` section with a `<meta charset="UTF-8">` tag
        - A `<meta name="viewport" content="width=device-width, initial-scale=1.0">` tag
        - A proper tailwind css link tag to load Tailwind CSS from CDN
        - A `<body>` section that contains the complete HTML structure and content
    The HTML document should be visually appealing, well-structured, and content/semantically-rich.
    You must strictly follow the output format shown below:
    <think>
    ...
    </think>

    <answer>
    ```html
    <html>
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        ...
    </head>
    <body>
    ...
    </body>
    </html>
    ```
    </answer>

    User: {user_request}
    Assistant:
    """

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print("This is training args", training_args)

    # Sampling settings used when vLLM generates rollouts for GRPO.
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        repetition_penalty = 1.1,
        stop = ["</answer>"],
        include_stop_str_in_output = True,
        max_tokens=3072,
        temperature=0.7
    )
    training_args.vllm_sampling_params = vllm_sampling_params

    # Resolve model loading options, including optional quantized loading.
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    device_map = None
    if quantization_config is not None:
        from trl import get_kbit_device_map
        device_map = get_kbit_device_map()
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the JSON dataset and convert each query into the fixed training prompt.
    dataset_name = script_args.dataset_name or str(DEFAULT_DATASET_PATH)
    dataset_path = Path(dataset_name)
    if not dataset_path.is_absolute():
        dataset_path = SCRIPT_DIR / dataset_path
    print(f"Loading training dataset from {dataset_path}")
    df = pd.read_json(dataset_path)
    def build_prompt(row):
        """Wrap a user query in the system prompt template expected by the model."""
        return SYSTEM_PROMPT.format(user_request=row["query"])
    df['prompt'] = df.apply(build_prompt, axis=1)
    df['reward_group_id'] = df.index.astype(str)
    train_dataset = Dataset.from_pandas(df[['prompt', 'reward_group_id']], preserve_index=False)
    df_eval = df.sample(n=5, random_state=42).copy()
    df_eval['reward_group_id'] = df_eval.index.astype(str)
    eval_dataset = Dataset.from_pandas(df_eval[['prompt', 'reward_group_id']], preserve_index=False)

    overall_pattern = r"<think>.+</think>.*<answer>.*```html.+```.*</answer>"
    overall_regex = re.compile(overall_pattern, re.DOTALL)
    def format_reward(prompts, completions, **kwargs):
        """Give a format reward of 1.0 only when the completion matches the required output schema."""
        responses = []
        for completion in completions:
            if isinstance(completion, str):
                responses.append(completion)
            elif isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict):
                    responses.append(completion[0]['content'])
                else:
                    responses.append(str(completion[0]))
            elif isinstance(completion, dict):
                responses.append(completion['content'])
            else:
                responses.append(str(completion))
        return [
            1.0 if overall_regex.search(response) else 0.0
            for response in responses
        ]

    # Train with a format reward plus the accessibility reward defined in accessibility_reward.py.
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward,
            accessibility_reward.axe_violation_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args)
    )
    if training_args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
    else:
        print("Starting training from scratch.")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
