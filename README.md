# How Much Context Is Enough? A Comparative Study of Input Scope of Code and Prompting in LLM-Based Vulnerability Detection

Artifact of the APSEC 2026 paper. It evaluates 424 vulnerable/patched function pairs (848 functions in C, C++, Java, and Python, built from [ReposVul](https://github.com/Eshe0922/ReposVul)) at three input scopes (the target function, its whole file, and its direct callers/callees), crossed with three prompting strategies (zero-shot, RAG, SFT), on six open-weight models forming three non-reasoning/reasoning pairs. The SFT training split has 476 functions (238 pairs) and shares no CVE with the evaluation set.

## Installation

Requires **Python ≥ 3.12** (developed on Ubuntu 22.04).

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install --no-deps codebleu==0.7.0   # pins tree-sitter<0.23 but runs on the grammars in requirements.txt
cp .env.example .env                   # LOCAL_API_URL: base URL of the vLLM server (without /v1)
```

## Setup

The datasets and saved runs are shipped as compressed archives in the repository; extract them in place:

```bash
tar xzf data.tar.gz        # -> data/     (evaluation, training, RAG-demonstration, and input-length statistics JSONL)
tar xzf results.tar.gz     # -> results/  (saved runs for every model, used by expr.ipynb)
```

The SFT LoRA adapters (~1 GB of safetensors) do not compress and are hosted on Google Drive. Download them only if you want to serve the fine-tuned (SFT) models:

```bash
pip install gdown
gdown 1G9xSUHUfjpj8M5pCSLBIRqpTIx6Mv2H2 -O models.tar.gz   # https://drive.google.com/file/d/1G9xSUHUfjpj8M5pCSLBIRqpTIx6Mv2H2
shasum -a 256 models.tar.gz   # 6041d82e72772398b734bd39b99ccc019a6017be05c1cb14d19314bcd382b4f6
tar xzf models.tar.gz      # -> models/
```

---

## Reproducing the paper

```bash
# all paper tables, the RQ3 figures, and in-text numbers from the saved runs
python -m src.analysis.paper_tables          # -> paper/tables, paper/figures, paper/paper_numbers.json
python -m src.analysis.sft_truncation        # SFT input-length table (needs the base tokenizers)
```

`expr.ipynb` contains the same metric code cell by cell.

### SFT adapters

One LoRA adapter per base checkpoint (rank 16, alpha 32, dropout 0.05, one epoch, lr 2e-4, effective batch 16, max length 4,096); the two Nemotron modes share one adapter:

```bash
python -m src.core.sft -m Qwen/Qwen3-30B-A3B-Instruct-2507 -d data/FuncFileRepo.train.jsonl -o models/Qwen3_30B_A3B_Instruct_2507
python -m src.core.sft -m nvidia/NVIDIA-Nemotron-Nano-12B-v2 -d data/FuncFileRepo.train.jsonl -o models/NVIDIA_Nemotron_Nano_12B_v2 \
    --control-token /no_think --trust-remote-code
```

Serve a base model with its adapter (`vllm serve <repo> --enable-lora --lora-modules <model>-sft=<adapter dir> --max-lora-rank 32 --max-model-len 131072`) and evaluate with `run.py -p sft -m <model>-sft`.

---

## Running the experiment

Every run evaluates one served model on all three scopes for the given strategies, resuming automatically (rows already scored are skipped; use `-r` to recompute).

Serve each model with vLLM 0.20.2 under the served name below, then run it (`LOCAL_API_URL` in `.env` points at the server):

```bash
vllm serve <checkpoint> --served-model-name <served name> --dtype bfloat16 --max-model-len 131072 \
    --generation-config auto --max-logprobs 20 \
    --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'
# add --trust-remote-code for Nemotron, and --limit-mm-per-prompt '{"image":0}' for Mistral/Magistral

python run.py -m qwen3-30b-instruct -l 32 -e 3 -p zero,rag
python run.py -m qwen3-30b-thinking --reasoning -l 24 -e 3 -p zero,rag
```

| Results label (`results/<label>`) | Served name (`-m`) | Checkpoint | Extra flags | `-l` |
| --- | --- | --- | --- | --- |
| `qwen3-30b-instruct` | `qwen3-30b-instruct` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | | 32 |
| `qwen3-30b-thinking` | `qwen3-30b-thinking` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | `--reasoning` | 24 |
| `mistral-small:24b` | `mistral-small:24b` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | | 32 |
| `magistral-small:24b` | `magistral-small:24b` | `mistralai/Magistral-Small-2509` | `--reasoning` | 24 |
| `nemotron-nano:12b` | `nemotron-nano:12b` | `nvidia/NVIDIA-Nemotron-Nano-12B-v2` | `--control-token /no_think` | 24 |
| `nemotron-nano-think:12b` | `nemotron-nano:12b` | same server as above | `--control-token /think --reasoning --label nemotron-nano-think:12b` | 24 |

For SFT, serve the base checkpoint with its adapter as `<served name>-sft` and keep the base label, e.g. `python run.py -m nemotron-nano:12b-sft --label nemotron-nano:12b --control-token /no_think -p sft -e 3`. The analysis (`src.analysis.paper_tables`) reads `results/<label>/{zero,rag,sft}/`.

### Options

| Short | Long              | Default                          | Description |
| ----- | ----------------- | -------------------------------- | ----------- |
| `-d`  | `--dataset`       | `data/FuncFileRepo.test.jsonl`   | Dataset JSONL (one sample per line). |
| `-s`  | `--savedir`       | `results`                        | Root directory for saved runs. |
| `-m`  | `--model`         | `qwen3-30b-instruct`             | Model id (selects the backend by prefix). |
| `-t`  | `--temperature`   | `None`                           | If unset, the served model's own default applies. |
| `-l`  | `--limit`         | `32`                             | Async concurrency (in-flight requests). |
| `-e`  | `--executions`    | `1`                              | Trials per condition (repeat to average run-to-run noise). |
| `-r`  | `--reset`         | `False`                          | Ignore existing saved runs and recompute. |
| `-p`  | `--strategies`    | `zero,rag`                       | Comma-separated: `zero`, `rag`, `sft`. |
|       | `--reasoning`     | `False`                          | Mark the model as reasoning (larger output budget, free-form). |
|       | `--example`       | auto                             | RAG demonstration file (defaults to `data/FuncFileRepo.example.jsonl`). |
|       | `--control-token` | `None`                           | e.g. `/think` or `/no_think` for the Nemotron hybrid. |
|       | `--label`         | `None`                           | Sub-directory label for this run's results. |

Results are written to `results/{model}/{strategy}/{bench}_{trial}.jsonl` (slim 9-column schema: `id, language, scope, prompts, complexity, label, predict, tokens, time(s)`), and aggregated metrics to `results/{model}/result.csv`.

## Rebuilding the dataset

`data/` is the reference dataset used in the paper. `src.utils.dataset` documents how it was built from ReposVul (pairing, exact-commit call-graph extraction, 128K budget filter); rebuilding needs `data/ReposVul.jsonl` from the ReposVul release, GNU cflow 1.7 under `tools/cflow-1.7` (C/C++), and PyCG (Python):

```bash
python -m src.utils.dataset -i data/ReposVul.jsonl -o data/FuncFileRepo.test.jsonl
```

## License

MIT (see `LICENSE`).

---
