import argparse
from src.core import Detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM vulnerability-detection experiments.")
    parser.add_argument("-d", "--dataset", default="data/FuncFileRepo.test.jsonl",
                        help="Path to the dataset JSONL file.")
    parser.add_argument("-s", "--savedir", default="results",
                        help="Directory where experiment results are stored.")
    parser.add_argument("-m", "--model", default="llama3.1:8b",
                        help="Model identifier to use for the chosen LLM backend.")
    parser.add_argument("-t", "--temperature", type=float, default=None,
                        help="Sampling temperature. Default: None = use the "
                             "served model's own generation_config default "
                             "(reproducibility; scores are logprob-based so "
                             "temperature does not affect them).")
    parser.add_argument("-l", "--limit", type=int, default=32,
                        help="Async concurrency limit (match vLLM --max-num-seqs).")
    parser.add_argument("-e", "--executions", type=int, default=1,
                        help="Number of trials (reasoning models vary run-to-run).")
    parser.add_argument("-r", "--reset", action="store_true",
                        help="Recompute from scratch, ignoring saved runs.")
    parser.add_argument("-p", "--strategies", default=None,
                        help="Comma-separated strategies (zero,rag,sft). Default: zero,rag.")
    parser.add_argument("--reasoning", action="store_true",
                        help="The served model emits a thinking trace before "
                             "its verdict (parse verdict after the think "
                             "delimiter; larger generation budget; no "
                             "JSON-schema forcing).")
    parser.add_argument("--example", default=None,
                        help="RAG example JSONL (default: auto-discover "
                             "<benchmark>.example.jsonl next to the dataset).")
    parser.add_argument("--control-token", default=None,
                        help="Token prepended to the system prompt for hybrid "
                             "models (e.g. Nemotron '/no_think' or '/think').")
    parser.add_argument("--label", default=None,
                        help="Results subdir name (default: model id). Use to "
                             "separate a hybrid model's two modes.")

    args = parser.parse_args()

    strategies = (
        [s.strip() for s in args.strategies.split(",") if s.strip()]
        if args.strategies else None
    )

    detector = Detector(
        llm=args.model,
        temperature=args.temperature,
        dataset_path=args.dataset,
        save_dir=args.savedir,
        async_limit=args.limit,
        strategies=strategies,
        reasoning=args.reasoning,
        example_path=args.example,
        control_token=args.control_token,
        label=args.label)
    detector.run(executions=args.executions, reset=args.reset)


if __name__ == "__main__":
    main()
