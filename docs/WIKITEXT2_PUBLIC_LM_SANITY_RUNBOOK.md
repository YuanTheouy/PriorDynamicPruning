# WikiText-2 Public LM Sanity Runbook

Date: 2026-06-02

## Scope

This runbook is only for the WikiText-2 PPL sanity benchmark for:

`OPAL-SetBCE = prefix_hk_raw_attn + final Delta_NLL greedy set teacher + BCE set supervision + hard top-K skip`

Do not run HellaSwag, PIQA, ARC, MMLU, SetAttn v1, Exact-K CE, or m10000 for this sanity pass.

## Implemented Files

- `build_wikitext_greedy_set_labels.py`
- `eval_wikitext_opal_ppl.py`
- `wikitext_opal_utils.py`
- `run_wikitext2_public_lm_sanity_gpu01234567.sh`
- `docs/WIKITEXT2_PUBLIC_LM_SANITY_RESULTS.md`

The runner writes results back to `docs/WIKITEXT2_PUBLIC_LM_SANITY_RESULTS.md` on the server repo.

## Server Model Check

Run this first inside the server tmux:

```bash
cd /workspace/PriorDynamicPruning
find /workspace/ckpts -maxdepth 4 -type d \
  | grep -Ei 'qwen|llama|mistral|MiniOneRec' \
  | head -80
```

Selection order:

| priority | model | note |
|---:|---|---|
| 1 | Qwen2.5/Qwen2 base 1.5B or 3B | Best public LM sanity setting. |
| 2 | Qwen2.5/Qwen2 instruct 1.5B or 3B | Acceptable, report as instruct. |
| 3 | MiniOneRec Office checkpoint | Smoke only, not a paper public LM result. |

If there is no Qwen2/Qwen2.5 general LM checkpoint and no cached WikiText-2 dataset, report the blocker instead of substituting another dataset.

## Smoke Command

If no Qwen checkpoint exists locally, download a base LM first:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
mkdir -p /workspace/ckpts

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-1.5B \
  --local-dir /workspace/ckpts/Qwen2.5-1.5B \
  --local-dir-use-symlinks False
```

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
git pull --ff-only origin codex/opal-llm-experiments

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=32 \
WIKITEXT_EVAL_WINDOWS=32 \
WIKITEXT_EPOCHS=1 \
WIKITEXT_SEQ_LEN=512 \
WIKITEXT_DATASET_DISK_PATH=/workspace/datasets/wikitext/wikitext-2-raw-v1 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58000 \
bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
```

Smoke must satisfy:

- Full PPL is finite.
- Static skip PPL is finite.
- OPAL router outputs exactly K skipped layers for every eval window.
- Average kept layers equals `num_layers - K`.
- Eval JSON has `uses_greedy_labels_at_eval: false`.

## Main Seed42 Command

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
git pull --ff-only origin codex/opal-llm-experiments

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=2000 \
WIKITEXT_EVAL_WINDOWS=512 \
WIKITEXT_EPOCHS=40 \
WIKITEXT_SEQ_LEN=1024 \
WIKITEXT_DATASET_DISK_PATH=/workspace/datasets/wikitext/wikitext-2-raw-v1 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58100 \
bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
```

Optional knobs:

```bash
WIKITEXT_ROUTER_PREFIX_TOKENS=256      # default: seq_len / 4
WIKITEXT_SKIP_RATE=0.25
WIKITEXT_PROTECTED_HEAD=4
WIKITEXT_PROTECTED_TAIL=2
WIKITEXT_CANDIDATE_BATCH_SIZE=1        # increase only if memory is safe
WIKITEXT_DATASET_DISK_PATH=/workspace/datasets/wikitext/wikitext-2-raw-v1
WIKITEXT_DATASET_CACHE_DIR=/path/to/hf/cache
WIKITEXT_RUN_RAW=1
```

The runner auto-detects `/workspace/datasets/wikitext/wikitext-2-raw-v1` when it exists. That directory is a HuggingFace `save_to_disk` Arrow dataset and is loaded with `datasets.load_from_disk`, so no network access is needed.

## What The Runner Does

1. Builds train-split Delta_NLL forward-greedy skip-set labels.
2. Trains Raw-SetBCE if `WIKITEXT_RUN_RAW=1`.
3. Trains OPAL-SetBCE with `prefix_hk_raw_attn`.
4. Evaluates Full on test.
5. Evaluates C6 static masks on validation and selects the best by validation NLL.
6. Evaluates Static uniform, Static ends_heavy, Static best-on-val C6 on test.
7. Evaluates Raw-SetBCE and OPAL-SetBCE on test.
8. Runs train-label overlap diagnostics.
9. Writes `docs/WIKITEXT2_PUBLIC_LM_SANITY_RESULTS.md`.

## Leakage Guard

For each WikiText window, the router sees only the first `router_prefix_tokens` tokens. PPL is scored only on suffix tokens. Greedy labels are only built on train windows, static best is selected only on validation, and final PPL is on test.
