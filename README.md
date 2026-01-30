# How Much Context Is Enough? A Comparative Study of Input Scope of Code in LLM-Based Vulnerability Detection.

<!-- ![image](./overview.png) -->
<div align="center">
  <img src="./overview.png" alt="overview"
       style="width:clamp(320px, 50%, 900px); height:auto; display:block;" />
</div>


## Setup

### Environments

- Ubuntu 22.04
- Python >= 3.12

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/anonymous/llmvd.git
    cd llmvd
    ```

2.  **(Optional) Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API keys:**
    Copy the [.env.example](.env.example) file to `.env` and add your API keys for the models you want to use:
    ```bash
    cp .env.example .env
    ```
    Then, open `.env` and fill in your keys.

### Load Data

* [**Dataset**](https://drive.usercontent.google.com/download?id=1HfBrlC2WHsveyE5GNHyOvakQoS0_Vwn0&confirm=t)
  
    ```bash
    gdown "https://drive.google.com/uc?id=1HfBrlC2WHsveyE5GNHyOvakQoS0_Vwn0" -O data.zip
    unzip data.zip
    ```

* **(Optional)** [**Results**](https://drive.usercontent.google.com/download?id=1fwObdsBjzFMLGQMtW4wMVu1EbTthJAHi&confirm=t)

    ```bash
    gdown "https://drive.google.com/uc?id=1fwObdsBjzFMLGQMtW4wMVu1EbTthJAHi" -O results.zip
    unzip results.zip
    ```
   

## How to Use
   
```bash
python run.py [OPTIONS]
```

### Quick Start

The command below is an example from the experimental setup mentioned in our paper. 

```bash
python run.py -d data/FuncFileRepo.jsonl -m mistral-nemo:12b -l 100 -e 10
```

### Models used from our paper
The `--model` option supports the following identifiers:
- `claude-sonnet-4-5`
- `gemini-2.0-flash`
- `gpt-4o`
- `gpt-5`
- `llama3.1:8b`
- `mistral-nemo:12b`
- `phi3:14b`
- `qwen3-coder:30b`

### OPTIONS
| Short | Long            | Type  | Default                   | Description                                                      |
| ----- | --------------- | ----- | ------------------------- | ---------------------------------------------------------------- |
| `-d`  | `--dataset`     | str   | `data/FuncFileRepo.jsonl` | Path to the dataset JSONL file.                                  |
| `-s`  | `--savedir`     | str   | `results`                 | Directory where experiment results are stored.                   |
| `-m`  | `--model`       | str   | `gpt-5`                   | Model identifier to use for the chosen LLM backend.              |
| `-t`  | `--temperature` | float | `1.0`                     | Temperature setting for the LLM (default: 1.0).                  |
| `-l`  | `--limit`       | int   | `1`                       | Maximum rate limit of LLM API (e.g., max concurrent requests).   |
| `-e`  | `--executions`  | int   | `1`                       | Number of executions per combination (default: 1).               |
| `-r`  | `--reset`       | flag  | `False`                   | Reset the experiment results (overwrite / ignore existing runs). |

## Project Structure

```
.
├── data/                   # Datasets for experiments
├── paper/                  # Research paper source
├── results/                # Experiment results
│   ├── {model_name}/       # Results for a specific model
│   │   ├── *.jsonl         # Raw LLM outputs
│   │   └── result.csv      # Performance metrics for the model
│   ├── *.csv               # Aggregated results across all models
│   └── *.png               # Plots and visualizations
├── src/                    # Source code
│   ├── core/               # Core logic (Detector, Evaluator)
│   ├── llms/               # LLM API clients (GPT, Claude, etc.)
│   ├── prompts/            # Prompt templates
│   └── utils/              # Utility scripts
├── run.py                  # Main script to run experiments
├── expr.ipynb              # Jupyter notebook for experiments and analysis
└── README.md
```
