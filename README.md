# Demystify AI

## AI Learning — Language Model Demos

A small, self-contained toolkit for *seeing* how language models predict the
next token. It pairs two interactive tkinter GUIs — one toy model you can read
end-to-end, one real local LLM — so you can watch the same idea (a probability
distribution over the next word) play out at both scales.

| Demo | What it shows |
|------|---------------|
| **`bigram_lm.py`** | A from-scratch bigram model. Trains on a text corpus by counting which word follows which, then lets you explore the learned next-word probabilities and sample a sequence step by step. No neural network — just frequency counts. |
| **`ollama_logprobs.py`** | A logprobs inspector for a real LLM running in [Ollama](https://ollama.com). Sends a prompt, renders the response as clickable token chips colour-coded by confidence (green = certain → red = uncertain), and shows the alternative tokens the model considered at each position. |

Together they form a learning arc: *how a trivial model picks the next word* →
*how a real LLM does the same thing with logprobs*.

The `References/` folder collects foundational papers for further reading:
*Attention Is All You Need* (Vaswani), *Language Models are Few-Shot Learners*
(Brown, GPT-3), *Training LMs to Follow Instructions with Human Feedback*
(Ouyang, InstructGPT/RLHF), *Constitutional AI* (Bai), and *The Illustrated
Transformer* (Alammar). Written walkthroughs live in `bigram_lm_explainer.docx`
and `llm_explainer.docx`.

## Requirements

- **Python 3.8+** with **tkinter** (the GUI toolkit). tkinter ships with
  the standard library but is a separate package on many systems — e.g.
  `python3-tk` on Debian/Ubuntu, `tk` on macOS Homebrew.
- **For `ollama_logprobs.py` only:** a local [Ollama](https://ollama.com)
  server running at `http://localhost:11434` with at least one model pulled
  (e.g. `ollama pull llama3`). The inspector relies on Ollama's `logprobs`
  support, so a recent Ollama version is recommended.

No third-party Python packages are needed — both demos use only the standard
library.

## Usage

### With Nix (recommended)

A `flake.nix` provides a reproducible environment with Python + tkinter:

```sh
nix run            # bigram demo (default)
nix run .#bigram   # bigram demo (explicit)
nix run .#logprobs # Ollama logprobs inspector

nix develop        # drop into a shell with python3 + tkinter on PATH
```

### Without Nix

```sh
python3 bigram_lm.py        # bigram model demo
python3 ollama_logprobs.py  # Ollama logprobs inspector
```

## How each demo works

### Bigram model (`bigram_lm.py`)

1. Edit or keep the default corpus, then train. The model lowercases the text,
   strips punctuation, and counts every adjacent word pair to build a
   next-word probability distribution for each word.
2. Inspect any word's distribution as ranked probability bars.
3. Step through generation: starting from a word, the model samples a
   successor according to the learned probabilities, building a sequence one
   token at a time until it reaches a word with no successors.

### Ollama logprobs inspector (`ollama_logprobs.py`)

1. Pick a model from the dropdown (auto-populated from `/api/tags`; use ↻ to
   refresh) and enter a prompt.
2. Click **Generate**. The response streams in from Ollama's `/api/generate`
   endpoint with per-token logprobs.
3. Each token becomes a chip coloured by its probability. Click a chip to see
   the ranked alternatives the model weighed at that position.

It also understands "thinking" blocks emitted by reasoning models (DeepSeek-R1,
Qwen3, Gemma, etc.) and can optionally suppress them.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'tkinter'`** — install your platform's
  tk package (see Requirements), or use `nix develop` / `nix run`.
- **"Cannot reach Ollama at http://localhost:11434"** — make sure Ollama is
  running (`ollama serve`) and a model is pulled (`ollama list`).
- **Empty model dropdown** — no models are pulled yet; run `ollama pull <model>`
  and click ↻.


## Additional Resources for AI education

External links:
- https://www.3blue1brown.com/?topic=neural-networks
- https://www.youtube.com/watch?v=bgWq678Oed4&list=PLEMXAbCVnmY6U_pA-7GKuP9xiv9utLaP4
- https://bbycroft.net/llm
- https://www.llm-visualized.com
- https://poloclub.github.io/transformer-explainer/
- https://karpathy.ai/zero-to-hero.html
- https://github.com/karpathy/minGPT
- http://neuralnetworksanddeeplearning.com/
- https://github.com/chrishayuk/larql

