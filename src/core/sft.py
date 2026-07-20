"""Supervised fine-tuning (SFT) — the weight-level intervention of the
scope x strategy study, and the training counterpart to `Detector`.

`Detector` scores a frozen model; `SFT` produces the LoRA adapter that the
`sft` strategy then serves (see `src/prompts/strategies.py`). The two share the
SAME prompt: SFT renders every `(row, scope)` with the `zero/` templates and
`PromptManager` exactly as the detector does at eval, and supervises the
assistant answer `{"vulnerable": 0|1}` the detector reads back. Training input
therefore matches evaluation input token-for-token.

  usage:
    python -m src.core.sft \
      -m Qwen/Qwen3-30B-A3B-Instruct-2507 \
      -d data/FuncFileRepo.train.jsonl \
      -o adapters/qwen3-30b-instruct

Two correctness choices that distinguish this from a naive digit-classifier
fine-tune, both learned the hard way (see the overfit diagnostics in the paper's
threats-to-validity):

  1. COMPLETION loss, not digit-only. We supervise the whole assistant turn
     `{"vulnerable": <d>}` + EOS. Masking everything but the single digit token
     leaves the loss pinned at ln 2 (a constant p=0.5 predictor): the format
     tokens carry no gradient, so the model has no scaffold to condition the
     digit on and never leaves the base's near-uniform state. `digit_only=True`
     reproduces that failure mode for ablation.

  2. The adapter must reach the OUTPUT logits. Adapting attention only is enough
     for a DENSE model (its MLP is a plain set of linear leaves that LoRA also
     wraps), but on a MoE (e.g. Qwen3-30B-A3B) the per-expert FFNs are fused
     Parameters that LoRA cannot wrap, so attention-only LoRA cannot move the
     verdict logits. We therefore (a) target attention AND MLP projections and
     (b) optionally train `lm_head` (--train-lm-head; off by default — the
         shipped adapters are pure LoRA, which vLLM can serve).

Runtime deps (torch + peft + trl + bitsandbytes) are imported lazily inside
`train()` so that importing this module for its prompt-building helpers — or on
a machine without a GPU — costs nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..prompts import PromptManager

SCOPES = ("function", "file", "repository")

# Leaf modules never adapted: vision/mm towers and the token embedding. Note
# `lm_head` is handled separately (via modules_to_save), and MoE expert FFNs are
# fused Parameters that LoRA cannot wrap, so they are excluded here and reached
# instead through lm_head (see the module docstring).
_EXCLUDE = ("vision", "visual", "image", "patch", "mm_", "multi_modal",
            "lm_head", "embed_tokens", "experts")
_LINEAR_CLS = ("Linear", "Linear4bit", "Linear8bitLt")


class SFT:
    def __init__(
        self,
        model: str,
        train_path: str,
        out_dir: str,
        prompt_root: str = "src/prompts",
        control_token: str | None = None,
        max_seq_len: int = 4096,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        epochs: float = 1.0,
        batch: int = 1,
        grad_accum: int = 16,
        lr: float = 2e-4,
        digit_only: bool = False,
        train_lm_head: bool | None = None,
        load_4bit: bool = False,
        trust_remote_code: bool = False,
        limit: int = 0,
        merged_out: str | None = None,
    ):
        self.model_id = model
        self.train_path = train_path
        self.out_dir = out_dir
        self.prompt_root = prompt_root
        self.control_token = control_token
        self.max_seq_len = max_seq_len
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.epochs = epochs
        self.batch = batch
        self.grad_accum = grad_accum
        self.lr = lr
        self.digit_only = digit_only
        # None = auto (decide once the model is loaded and we can see the MoE
        # experts); True/False forces it.
        self.train_lm_head = train_lm_head
        self.load_4bit = load_4bit
        self.trust_remote_code = trust_remote_code
        self.limit = limit
        # If set, also merge the adapter into the base after training and save
        # the full model here. Needed to SERVE a MoE adapter: it carries a
        # trained lm_head (modules_to_save) that vLLM's LoRA path cannot load,
        # and reloading the adapter to merge later trips a peft/transformers
        # version incompatibility — so the merge must happen in-process.
        self.merged_out = merged_out

        self.pm = PromptManager()

    # ------------------------------------------------------------------ #
    # Prompt construction — identical to Detector, so train == eval.
    # ------------------------------------------------------------------ #
    def _tpl(self, name: str) -> str:
        # sft reuses the zero/ templates (it is a model variant, not a new
        # prompt); the hybrid control token is prepended to the system prompt
        # exactly as the detector does at serve time.
        return f"{self.prompt_root}/zero/{name}.md"

    def _system(self) -> str:
        prefix = f"{self.control_token}\n" if self.control_token else ""
        return prefix + self.pm.render(file=self._tpl("system"))

    def _repo_info(self, language, callees, callers) -> str:
        # Same formatting as Detector._repo_info. Neighbours are emitted in
        # natural dataset order here: eval re-ranks them by CodeBLEU only to
        # decide what survives the context budget, which is immaterial for
        # teaching the verdict. Over-long prompts are tail-truncated at
        # tokenization time (see `_encode`).
        info = ""
        if callees:
            info += "### Functions called by the Target Function (Callees):\n"
            for c in callees:
                info += f"#### File: {c['file']}\n```{language}\n{c['function']}\n```\n"
        if callers:
            info += "### Functions that call the Target Function (Callers):\n"
            for c in callers:
                info += f"#### File: {c['file']}\n```{language}\n{c['function']}\n```\n"
        return info

    def _user(self, row: dict, scope: str) -> str:
        language, function = row["language"], row["function"]
        base_kw = dict(language=language, function=function)
        if scope == "function":
            return self.pm.render(file=self._tpl("function"), **base_kw)
        if scope == "file":
            return self.pm.render(file=self._tpl("file"),
                                  file_code=row["file"], **base_kw)
        repo = row.get("repository") or {"callee": [], "caller": []}
        return self.pm.render(
            file=self._tpl("repository"),
            repository=self._repo_info(language, repo.get("callee") or [],
                                       repo.get("caller") or []),
            **base_kw)

    @staticmethod
    def _target(row: dict) -> str:
        return json.dumps({"vulnerable": 1 if row["vulnerable"] else 0})

    def _iter_examples(self):
        """Yield (system, user, target, meta) for every (row, scope)."""
        lines = open(self.train_path, encoding="utf-8").readlines()
        if self.limit:  # balanced overfit subset: interleave head and tail
            half = self.limit // 2
            lines = lines[:half] + lines[-half:]
        system = self._system()
        for line in lines:
            row = json.loads(line)
            for scope in SCOPES:
                yield (system, self._user(row, scope), self._target(row),
                       {"id": row.get("id"), "scope": scope,
                        "vulnerable": bool(row["vulnerable"])})

    # ------------------------------------------------------------------ #
    # Tokenization + loss masking.
    # ------------------------------------------------------------------ #
    def _encode(self, tokenizer, system, user, target, mistral_tok=None):
        """Return (input_ids, labels): prompt masked to -100, completion
        supervised. `digit_only` masks the completion down to the single
        verdict digit (an ablation of the failure mode described in the module
        docstring)."""
        eos = tokenizer.eos_token_id
        if mistral_tok is not None:
            # Mistral ships no HF chat_template (it uses mistral_common, as vLLM
            # does at serve time); build the prompt the same way so training
            # matches eval token-for-token.
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            from mistral_common.protocol.instruct.messages import (
                SystemMessage, UserMessage)
            req = ChatCompletionRequest(messages=[SystemMessage(content=system),
                                                  UserMessage(content=user)])
            prompt_ids = mistral_tok.encode_chat_completion(req).tokens
            base = mistral_tok.instruct_tokenizer.tokenizer
            tgt_ids = base.encode(target, bos=False, eos=False) + [base.eos_id]
            decode1 = lambda tid: base.decode([tid])
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=False)
            if not isinstance(prompt_ids, list):  # some versions return a dict
                prompt_ids = prompt_ids["input_ids"]
            tgt_ids = tokenizer(target, add_special_tokens=False).input_ids
            if eos is not None:
                tgt_ids = tgt_ids + [eos]
            decode1 = lambda tid: tokenizer.decode([tid])

        if self.digit_only:
            tgt_labels = [-100] * len(tgt_ids)
            for i, tid in enumerate(tgt_ids):
                if decode1(tid).strip() in ("0", "1"):
                    tgt_labels[i] = tid
            if all(x == -100 for x in tgt_labels):
                return None  # no isolable digit token; skip
        else:
            tgt_labels = list(tgt_ids)

        # Tail-truncate an over-long prompt: keep the LAST `budget` tokens as a
        # contiguous block so the `<|im_start|>assistant\n{"vulnerable": `
        # prefix that immediately precedes the digit stays intact.
        budget = self.max_seq_len - len(tgt_ids)
        if budget < 16:
            return None
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]
        return prompt_ids + tgt_ids, [-100] * len(prompt_ids) + tgt_labels

    def _build_dataset(self, tokenizer, mistral_tok=None):
        examples, skipped = [], 0
        for system, user, target, _meta in self._iter_examples():
            enc = self._encode(tokenizer, system, user, target, mistral_tok)
            if enc is None:
                skipped += 1
                continue
            input_ids, labels = enc
            examples.append({"input_ids": input_ids, "labels": labels})
        print(f"[data] {len(examples)} examples, {skipped} skipped "
              f"(no digit / no room at max_seq_len={self.max_seq_len})")
        return examples

    # ------------------------------------------------------------------ #
    # Model + LoRA.
    # ------------------------------------------------------------------ #
    def _target_modules(self, model) -> list[str]:
        names: set[str] = set()
        for name, module in model.named_modules():
            if module.__class__.__name__ not in _LINEAR_CLS:
                continue
            if any(k in name.lower() for k in _EXCLUDE):
                continue
            names.add(name.split(".")[-1])
        return sorted(names)

    @staticmethod
    def _is_moe(model) -> bool:
        # MoE checkpoints expose fused expert FFNs (module path contains
        # ".experts") and/or a num_experts field on the config.
        cfg = getattr(model, "config", None)
        if cfg is not None and any(
                getattr(cfg, k, None) for k in
                ("num_experts", "num_local_experts", "n_routed_experts")):
            return True
        return any(".experts" in n for n, _ in model.named_modules())

    def train(self):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, Trainer, TrainerCallback,
                                  TrainingArguments)
        from peft import LoraConfig, get_peft_model

        class LossLog(TrainerCallback):
            """Grep-friendly loss line each logging step (the default progress
            bar swallows loss when stdout is piped to a file)."""
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    print(f"LOSS step={state.global_step}/{state.max_steps} "
                          f"loss={logs['loss']:.4f} "
                          f"lr={logs.get('learning_rate', 0):.2e} "
                          f"epoch={logs.get('epoch', 0):.2f}", flush=True)

        print(f"[sft] {self.model_id}  ctl={self.control_token}  "
              f"digit_only={self.digit_only}")
        tok = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        mistral_tok = None
        if tok.chat_template is None:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            mistral_tok = MistralTokenizer.from_hf_hub(self.model_id)
            print("[tok] no HF chat_template -> using mistral_common")

        if self.load_4bit:
            # 4-bit NF4 (QLoRA). Do NOT pass a torch_dtype: on a MoE the fused
            # experts cannot be 4-bit quantized and a bf16 dtype would force a
            # full materialization on load and OOM.
            load_kw = dict(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True))
        else:
            load_kw = dict(dtype=torch.bfloat16)  # bf16 base (LoRA, not QLoRA)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map={"": 0},
                trust_remote_code=self.trust_remote_code, **load_kw)
        except (ValueError, KeyError) as e:
            # Mistral-Small-3.2 / Magistral resolve to a vision-language wrapper
            # unknown to AutoModelForCausalLM; load the multimodal model — text
            # batches never touch the vision tower and LoRA targets only the LM.
            print(f"[load] CausalLM unsupported ({type(e).__name__}); "
                  f"loading as ImageTextToText")
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, device_map={"": 0},
                trust_remote_code=self.trust_remote_code, **load_kw)
        model.config.use_cache = False
        # Freeze the base explicitly (only LoRA + any modules_to_save train). We
        # deliberately skip peft.prepare_model_for_kbit_training: it upcasts
        # every non-4bit bf16 tensor to fp32, which on a MoE blows the ~30B fused
        # experts up to ~120 GB and OOMs. The fp32 cast is unnecessary when the
        # base is frozen.
        for p in model.parameters():
            p.requires_grad_(False)
        model.enable_input_require_grads()

        # Output head: OFF by default. The shipped adapters are pure LoRA
        # (attention/MLP only), which vLLM's LoRA path can serve; training
        # lm_head (modules_to_save) is kept as an opt-in ablation only.
        is_moe = self._is_moe(model)
        train_head = bool(self.train_lm_head)
        targets = self._target_modules(model)
        save_mods = ["lm_head"] if train_head else None
        print(f"[lora] moe={is_moe}  r={self.lora_r}  "
              f"target_modules({len(targets)})={targets}  "
              f"modules_to_save={save_mods}")
        model = get_peft_model(model, LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout, bias="none",
            task_type="CAUSAL_LM", target_modules=targets,
            modules_to_save=save_mods))
        model.print_trainable_parameters()

        data = self._build_dataset(tok, mistral_tok)

        class Collator:
            def __init__(self, pad_id):
                self.pad_id = pad_id

            def __call__(self, batch):
                maxlen = max(len(b["input_ids"]) for b in batch)
                input_ids, labels, attn = [], [], []
                for b in batch:
                    n = maxlen - len(b["input_ids"])
                    input_ids.append(b["input_ids"] + [self.pad_id] * n)
                    labels.append(b["labels"] + [-100] * n)
                    attn.append([1] * len(b["input_ids"]) + [0] * n)
                return {"input_ids": torch.tensor(input_ids),
                        "labels": torch.tensor(labels),
                        "attention_mask": torch.tensor(attn)}

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=self.out_dir + "/_ckpt",
                per_device_train_batch_size=self.batch,
                gradient_accumulation_steps=self.grad_accum,
                num_train_epochs=self.epochs,
                learning_rate=self.lr,
                bf16=True,
                logging_steps=10,
                logging_first_step=True,
                disable_tqdm=True,
                save_strategy="no",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                # bf16 base optimizes only the small LoRA + head params, so plain
                # adamw_torch suffices and avoids the bitsandbytes dependency.
                optim="paged_adamw_8bit" if self.load_4bit else "adamw_torch",
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                report_to=[],
            ),
            train_dataset=data,
            data_collator=Collator(tok.pad_token_id),
            callbacks=[LossLog()],
        )
        trainer.train()
        model.save_pretrained(self.out_dir)
        tok.save_pretrained(self.out_dir)
        print(f"[done] adapter saved -> {self.out_dir}")
        if self.merged_out:
            # Merge in-process: fold the LoRA deltas into the base and keep the
            # trained lm_head, yielding a plain full model that vLLM can serve
            # directly (no LoRA, no adapter reload).
            merged = model.merge_and_unload()
            merged.save_pretrained(self.merged_out)
            tok.save_pretrained(self.merged_out)
            print(f"[done] merged model saved -> {self.merged_out}")


def main():
    ap = argparse.ArgumentParser(description="LoRA SFT for the sft strategy")
    ap.add_argument("-m", "--model", required=True, help="HF repo id of the base")
    ap.add_argument("-d", "--data", default="data/FuncFileRepo.train.jsonl",
                    help="train-split JSONL (same schema as the eval dataset)")
    ap.add_argument("-o", "--out", required=True, help="adapter output dir")
    ap.add_argument("--control-token", default=None,
                    help="hybrid thinking toggle prepended to the system prompt")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--digit-only", action="store_true",
                    help="ablation: supervise only the verdict digit (collapses "
                         "to a constant p=0.5 predictor — see module docstring)")
    lm = ap.add_mutually_exclusive_group()
    lm.add_argument("--train-lm-head", dest="train_lm_head", action="store_true",
                    default=None, help="also train lm_head (ablation; default off "
                                       "— pure LoRA is what vLLM can serve)")
    lm.add_argument("--no-train-lm-head", dest="train_lm_head",
                    action="store_false", help="never train lm_head")
    ap.add_argument("--load-4bit", action="store_true",
                    help="QLoRA: load the base in 4-bit NF4 (default: bf16 LoRA)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only N rows (balanced) — overfit sanity check")
    ap.add_argument("--merged-out", default=None,
                    help="also save the merged full model here (needed to serve "
                         "a MoE adapter, whose trained lm_head vLLM cannot LoRA-load)")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    SFT(
        model=args.model, train_path=args.data, out_dir=args.out,
        control_token=args.control_token, max_seq_len=args.max_seq_len,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, epochs=args.epochs, batch=args.batch,
        grad_accum=args.grad_accum, lr=args.lr, digit_only=args.digit_only,
        train_lm_head=args.train_lm_head, load_4bit=args.load_4bit,
        trust_remote_code=args.trust_remote_code, limit=args.limit,
        merged_out=args.merged_out,
    ).train()


if __name__ == "__main__":
    main()
