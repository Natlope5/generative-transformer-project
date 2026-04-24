import json
import os
from time import time

from evaluate import load as load_metric

from src.data import load_samsum_subset
from src.model import load_summarization_pipeline, generate_summary

RESULTS_DIR = "results"

def run_experiment(config_path, split="test", num_examples=50, tag="baseline"):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    dialogues, references = load_samsum_subset(split=split, num_examples=num_examples)
    gen = load_summarization_pipeline()

    preds = []
    start_time = time()
    for dlg in dialogues:
        summary = generate_summary(
            gen,
            dlg,
            prompt_style=cfg["prompt_style"],
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
        )
        # take only text after "Summary:" if present
        if "Summary:" in summary:
            summary = summary.split("Summary:")[-1].strip()
        preds.append(summary)
    end_time = time()

    total_time = end_time - start_time
    latency = total_time / len(dialogues)

    rouge = load_metric("rouge")
    rouge_results = rouge.compute(predictions=preds, references=references)

    metrics = {
        "config": cfg,
        "num_examples": len(dialogues),
        "latency_sec_per_example": latency,
        "rouge": rouge_results,
    }

    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{tag}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    samples_path = os.path.join(RESULTS_DIR, f"samples_{tag}.txt")
    with open(samples_path, "w", encoding="utf-8") as f:
        for dlg, ref, pred in zip(dialogues, references, preds):
            f.write("DIALOGUE:\n" + dlg + "\n\n")
            f.write("REFERENCE:\n" + ref + "\n\n")
            f.write("PREDICTION:\n" + pred + "\n")
            f.write("=" * 80 + "\n\n")

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved samples to {samples_path}")
    print(f"ROUGE-L: {rouge_results.get('rougeL', None)}")
    print(f"Latency (sec/example): {latency:.3f}")

if __name__ == "__main__":
    # Baseline
    run_experiment("configs/baseline.json", tag="baseline")
    # Improved
    run_experiment("configs/improved.json", tag="improved")