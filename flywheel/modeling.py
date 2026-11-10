"""Model loading, greedy generation and sequence log-probs (shared by app, train and eval)."""
from __future__ import annotations

import torch


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(model_id: str, adapter_dir: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.to(device()).eval()
    return model, tok


def _prompt_ids(tok, prompt: str) -> list[int]:
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"]) if isinstance(ids, dict) or hasattr(ids, "keys") else list(ids)


def _full_ids(tok, prompt: str, response: str) -> list[int]:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], tokenize=True
    )
    return list(ids["input_ids"]) if isinstance(ids, dict) or hasattr(ids, "keys") else list(ids)


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int = 96, sample: bool = False,
                   temperature: float = 0.8) -> list[str]:
    tok.padding_side = "left"
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    kw = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if sample else {"do_sample": False}
    out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(g, skip_special_tokens=True).strip() for g in gen]


def generate_one(model, tok, prompt: str, max_new_tokens: int = 96) -> str:
    return generate_batch(model, tok, [prompt], max_new_tokens)[0]


@torch.no_grad()
def response_logprob(model, tok, prompt: str, response: str) -> tuple[float, int]:
    """Sum of log p(response tokens | chat-formatted prompt) and number of response tokens."""
    p_ids = _prompt_ids(tok, prompt)
    full = _full_ids(tok, prompt, response)
    if full[: len(p_ids)] != p_ids:
        raise ValueError("chat template prompt prefix mismatch")
    ids = torch.tensor([full], device=model.device)
    logits = model(ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    lp = torch.log_softmax(logits, -1).gather(-1, targets[:, None])[:, 0]
    resp_lp = lp[len(p_ids) - 1:]
    return float(resp_lp.sum()), int(resp_lp.numel())
