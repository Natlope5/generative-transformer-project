import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/phi-2"

def load_summarization_pipeline(dtype=torch.float32, device="cpu"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,              # ← updated from torch_dtype to dtype
        trust_remote_code=True,
    ).to(device)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    return {
        "tokenizer": tok,
        "model": model,
        "device": device,
    }

def generate_summary(gen, dialogue, prompt_style="baseline", max_new_tokens=60, temperature=0.3):
    if prompt_style == "baseline":
        prompt = (
            "Summarize the following dialogue in one concise sentence.\n\n"
            f"Dialogue:\n{dialogue}\n\nSummary:"
        )
    elif prompt_style == "improved":
        prompt = (
            "You are an expert dialogue summarization system. "
            "Write one short, faithful sentence that states only the main outcome. "
            "Do not add unnecessary words.\n\n"
            f"Dialogue:\n{dialogue}\n\nSummary:"
        )
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")

    tok = gen["tokenizer"]
    model = gen["model"]
    device = gen["device"]

    inputs = tok(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(generated_ids, skip_special_tokens=True).strip()
    text = " ".join(text.split())
    return text