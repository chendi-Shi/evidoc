"""Lazy GPU/model setup shared by training and evaluation."""
from __future__ import annotations

from pathlib import Path

MODEL_REVISIONS = {
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
}
HYBRID_TARGETS = ("q_proj", "v_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def resolve_revision(model_id: str, revision: str | None) -> str | None:
    if Path(model_id).is_dir():
        return None
    if revision:
        return revision
    if model_id in MODEL_REVISIONS:
        return MODEL_REVISIONS[model_id]
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id).sha


def load_tokenizer(model_id: str, revision: str | None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Selected model must provide a chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_id: str, revision: str | None, quantization: str, dtype_name: str = "auto"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run the GPU probe first; this training entry point will not silently use CPU.")
    if dtype_name == "auto":
        dtype_name = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if dtype_name == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This PyTorch/CUDA stack did not confirm BF16. Run the GPU probe or explicitly choose --dtype fp16.")
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    loader = AutoModelForCausalLM
    if config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForCausalLM

        config = config.get_text_config()
        loader = Qwen3_5ForCausalLM
    config.use_cache = False
    kwargs = {"config": config, "revision": revision, "trust_remote_code": False,
              "dtype": dtype, "device_map": {"": 0}, "attn_implementation": "eager",
              "output_loading_info": True}
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif quantization != "none":
        raise ValueError("quantization must be none or nf4")
    model, info = loader.from_pretrained(model_id, **kwargs)
    # A bad multimodal-to-text prefix conversion must never train random weights.
    parameters = dict(model.named_parameters())
    missing_parameters = [key for key in info.get("missing_keys", []) if key in parameters]
    if missing_parameters:
        raise RuntimeError(f"Checkpoint did not load model parameters: {missing_parameters[:10]}")
    if any(parameter.is_meta for parameter in parameters.values()):
        raise RuntimeError("Model contains unloaded meta parameters")
    if any(parameter.device.type != "cuda" or parameter.device.index != 0 for parameter in parameters.values()):
        raise RuntimeError("All model parameters must reside on CUDA:0; CPU/disk offload is not allowed in this experiment")
    return model, {"model_id": model_id, "revision": revision, "dtype": dtype_name,
                   "quantization": quantization, "class": type(model).__name__,
                   "attention": "eager", "loading_info": _json_safe(info),
                   "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda,
                   "warning": "Qwen3.5 defaults to nonquantized LoRA; NF4 is an explicit experimental option, not an equivalent-quality guarantee."}


def attach_lora(model, quantization: str, rank: int, targets: str, dropout: float):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    available = {name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                 if hasattr(module, "in_features") and hasattr(module, "out_features")}
    selected = [name for name in HYBRID_TARGETS if name in available] if targets == "auto" else [name.strip() for name in targets.split(",")]
    if not selected or set(selected) - available:
        raise ValueError(f"LoRA targets absent from model: {sorted(set(selected) - available)}")
    module_counts = {target: sum(name.rsplit(".", 1)[-1] == target for name, _ in model.named_modules()) for target in selected}
    if quantization == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=dropout,
                                            target_modules=selected, bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return model, {"rank": rank, "alpha": 2 * rank, "dropout": dropout, "targets": selected,
                   "trainable_parameters": trainable, "total_parameters": total,
                   "adapted_module_counts": module_counts,
                   "scope": "Only the explicitly listed projection modules receive LoRA; all original parameters remain frozen."}
