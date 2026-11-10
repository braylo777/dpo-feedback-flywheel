"""DPO (or ORPO-style via loss_type) LoRA fine-tune with TRL on preference pairs."""

from __future__ import annotations

import json
import time
from pathlib import Path


def train_dpo(
    pairs: list[dict],
    model_id: str,
    out_dir: str,
    beta: float = 0.1,
    epochs: float = 2.0,
    lr: float = 5e-5,
    batch_size: int = 8,
    lora_r: int = 16,
    seed: int = 0,
    max_steps: int = -1,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from .pairs import to_conversational

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    ds = Dataset.from_list([to_conversational(p) for p in pairs])
    args = DPOConfig(
        output_dir=out_dir,
        beta=beta,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_length=256,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=seed,
        lr_scheduler_type="cosine",
        warmup_steps=5,
    )
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0
    adapter_dir = str(Path(out_dir) / "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    log = [h for h in trainer.state.log_history if "loss" in h]
    summary = {
        "model_id": model_id,
        "n_pairs": len(pairs),
        "beta": beta,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "lora_r": lora_r,
        "steps": trainer.state.global_step,
        "train_loss": result.training_loss,
        "wall_seconds": wall,
        "device": str(trainer.model.device),
        "adapter_dir": adapter_dir,
        "log_history": log,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
