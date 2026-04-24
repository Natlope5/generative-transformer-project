# Prompt Engineering for Dialogue Summarization with a Pretrained Transformer

This project evaluates how different prompts affect abstractive dialogue summarization using a pretrained transformer model on the SAMSum dataset. It compares a simple baseline prompt to an improved, outcome-focused prompt and measures ROUGE-L and latency.

## Folder structure

- `notebooks/`
  - `baseline_prompting.ipynb` – main experiment notebook for dialogue summarization.
- `results/`
  - `summarization_results.csv` – ROUGE-L scores and summaries for three sample dialogues.
  - `rouge_bar.jpg` – bar chart visualization of ROUGE-L scores.
- `paper/`
  - `Prompt-Engineering-for-Dialogue-Summarization-with-a-Pretrained-Transformer.docx` – full project report.
- `README.md` – this file.

## How to run

1. Open `notebooks/baseline_prompting.ipynb` in Jupyter or Google Colab.
2. Install required libraries if needed:
   - `transformers`
   - `datasets`
   - `evaluate`
   - `torch`
3. Run the notebook cells in order to:
   - Load the Phi-2 model and SAMSum dialogues.
   - Generate baseline and improved summaries.
   - Compute ROUGE-L scores and save results to `results/summarization_results.csv`.

## Reproducible Colab notebook

A reproducible Google Colab notebook for this project is available at:

https://colab.research.google.com/drive/1jve2JRWX7Im5vJkuHWiSEWswKZb0P1-Z?usp=sharing

## Summary of results

- Improved prompt generally produces shorter, more outcome-focused summaries.
- ROUGE-L does not always improve and can be worse when important details are omitted.
- End-to-end runtime for three dialogues is about 31.8 seconds total (≈10.6 seconds per dialogue), including generation and ROUGE-L evaluation.

## Author

Natalie Lopez  
Master’s Program in Advanced Artificial Intelligence  
Full Sail University
