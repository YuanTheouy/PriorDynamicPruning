# WikiText-2 Public LM Related Baseline Results

Last updated: 2026-06-02

## Setup

- model path: `/workspace/ckpts/Qwen2.5-1.5B`
- dataset: `/workspace/datasets/wikitext/wikitext-2-raw-v1`
- num_layers: 28
- seq_len: 1024
- router_prefix_tokens: 256
- eval_windows requested / actual: 512 / 289
- eval_tokens: 221,952
- skip_rate: 0.25
- skip_count: 7
- protected_head / protected_tail: 4 / 2
- label_samples: 2000
- candidate library: C16 `uniform`, `ends_heavy`, `first_k`, `last_k`, `middle_heavy`, `random_diverse_seed42`, `random_diverse_v1`...`random_diverse_v10`

## Original WikiText Bad Result

| method | type | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique masks |
|---|---|---:|---:|---:|---:|---:|
| Full | full | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 1 |
| Static uniform | static | 4.0673 | 58.3987 | 1.8519 | 49.2338 | 1 |
| Static ends_heavy | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 |
| Static best-on-val C6 | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 |
| Raw-SetBCE | ablation | 2.7618 | 15.8283 | 0.5464 | 6.6634 | 8 |
| OPAL-SetBCE | ours | 2.7672 | 15.9138 | 0.5518 | 6.7489 | 15 |

Raw-SetBCE is ahead of OPAL-SetBCE by 0.0054 NLL / 0.0855 PPL. OPAL is not currently the best learned method on WikiText-2.

## WikiText-2 Audit

| check | status | note |
|---|---|---|
| score direction | pass | `y_l=1` means skip; BCE uses `binary_cross_entropy_with_logits(-pred, target)`, so skip labels push `pred` lower; inference skips the top-K lowest `pred` layers. |
| mask application | pass | After the `maskcfg` fix, static masks produce large nonzero deltas from Full; the earlier equal-NLL run is invalid and should not be cited. |
| exact budget | pass | Every pasted valid method has `exact_skip_count_rate=1.0` and average kept layers 21 for K=7. |
| protected policy | pass | Full/static/raw/OPAL and the new baselines use the same protected head/tail and K. |
| split leakage | pass | Train labels are train-only; static best uses validation; final PPL uses test; eval paths do not load greedy labels. |
| router input | pass | Routers see only the first `router_prefix_tokens`; PPL and labels score suffix tokens only. |
| full PPL sanity | pass | Full PPL 9.1649 is finite and plausible for Qwen2.5-1.5B on WikiText-2 raw. |
| static best selection | pass | C6 best is selected by validation PPL, not test PPL. |
| label objective | pass | Greedy labels optimize suffix Delta_NLL. New candidate labels also optimize suffix Delta_NLL. |

## Why OPAL Did Not Win Yet

- OPAL beats Static best-on-val C6, so the mask application and learned routing path are not completely broken.
- OPAL loses narrowly to Raw-SetBCE and also loses to the stronger-access layerwise hidden router, which suggests `prefix_hk_raw_attn` is not automatically the best signal for public LM windows.
- OPAL produces more unique masks than Raw, but WikiText-2 PPL rewards mask quality, not mask diversity.
- The raw embedding baseline may be a lower-variance content prior for contiguous LM windows, while H^k may add noisy teacher-prefix features when the training set is only m=2000 windows.
- PuDDing-style and IG-style both collapse to the static `ends_heavy` candidate on all 289 test windows, so OPAL beating them mostly says OPAL beats candidate-library selection, not that it beats all adaptive related-work styles.

## Related Baseline Table

Server run completed:

| method | type | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique masks | exact-K |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | full | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 1 | 1.000 |
| Static uniform | static | 4.0673 | 58.3987 | 1.8519 | 49.2338 | 1 | 1.000 |
| Static ends_heavy | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 |
| Static best-on-val C6 | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 |
| PuDDing-style | related | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 |
| IG-style | related | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 |
| layerwise_hidden_router | related | 2.7568 | 15.7491 | 0.5414 | 6.5842 | 8 | 1.000 |
| Raw-SetBCE | ablation | 2.7618 | 15.8283 | 0.5464 | 6.6634 | 8 | 1.000 |
| OPAL-SetBCE | ours | 2.7672 | 15.9138 | 0.5518 | 6.7489 | 15 | 1.000 |

Dynamic ranking among learned/adaptive methods:

| rank | method | note |
|---:|---|---|
| 1 | layerwise_hidden_router | stronger-access in-framework baseline; not a full Dr.LLM reproduction |
| 2 | Raw-SetBCE | same final Delta_NLL set labels, raw prefix only |
| 3 | OPAL-SetBCE | ours; beats static/PuDDing/IG but not Raw/layerwise |
| 4 | PuDDing-style / IG-style | both select `ends_heavy` for every test window |

Selected candidate distributions:

- PuDDing-style: `{"ends_heavy": 289}`
- IG-style: `{"ends_heavy": 289}`

## Training Diagnostics

Exact train-loss and train-label overlap diagnostics were produced as server-side artifacts, but the scalar values are not present in the compact metric log pasted into this local thread. Do not invent these values. Extract them from `/workspace/PriorDynamicPruning` with the command below and then replace the `pending artifact extract` cells.

| method | loss first ↓ | loss best ↓ | loss last ↓ | overlap@7 ↑ | hamming ↓ | pairwise predicted hamming | exact match ↑ | unique teacher masks | unique predicted masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw-SetBCE | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract |
| OPAL-SetBCE | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract |
| layerwise_hidden_router | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract |

Server artifact extraction command:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate

RUN=wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25
ROOT=results/wikitext2_public_lm_sanity
CKPT_BASE=policy_ckpts/wikitext2_public_lm_sanity/${RUN}
RELATED_CKPT=policy_ckpts/wikitext2_public_lm_related/${RUN}
DIAG=${ROOT}/diagnostics/${RUN}
LABEL=${ROOT}/labels/${RUN}_delta_nll_greedy_set_labels.jsonl
MODEL=/workspace/ckpts/Qwen2.5-1.5B
DATA=/workspace/datasets/wikitext/wikitext-2-raw-v1

mkdir -p "$DIAG"

# Layerwise overlap was not part of the base sanity run; generate it if missing.
if [ -s "${RELATED_CKPT}/layerwise_hidden_bce/risk_router.pt" ] && [ ! -s "${DIAG}/layerwise_hidden_router_train_overlap.jsonl" ]; then
  accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_port 58400 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    ./eval_wikitext_opal_ppl.py diagnose_overlap \
      --teacher_model "$MODEL" \
      --split train \
      --dataset_disk_path "$DATA" \
      --seq_len 1024 \
      --router_prefix_tokens 256 \
      --risk_label_file "$LABEL" \
      --risk_router_ckpt "${RELATED_CKPT}/layerwise_hidden_bce/risk_router.pt" \
      --protected_head 4 \
      --protected_tail 2 \
      --batch_size 1 \
      --output_jsonl "${DIAG}/layerwise_hidden_router_train_overlap.jsonl" \
      --summary_json "${DIAG}/layerwise_hidden_router_train_overlap.summary.json" \
      --precision bf16 \
      --seed 42
fi

python3 - <<'PY'
import itertools, json, math, os

run = "wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25"
root = "results/wikitext2_public_lm_sanity"
diag = f"{root}/diagnostics/{run}"
paths = {
    "Raw-SetBCE": {
        "train": f"policy_ckpts/wikitext2_public_lm_sanity/{run}/raw_embedding_bce/training_metrics.json",
        "overlap": f"{diag}/raw_setbce_train_overlap.jsonl",
    },
    "OPAL-SetBCE": {
        "train": f"policy_ckpts/wikitext2_public_lm_sanity/{run}/prefix_hk_raw_attn_bce/training_metrics.json",
        "overlap": f"{diag}/opal_setbce_train_overlap.jsonl",
    },
    "layerwise_hidden_router": {
        "train": f"policy_ckpts/wikitext2_public_lm_related/{run}/layerwise_hidden_bce/training_metrics.json",
        "overlap": f"{diag}/layerwise_hidden_router_train_overlap.jsonl",
    },
}

def fmt(x):
    if x is None:
        return "NA"
    try:
        x = float(x)
    except Exception:
        return str(x)
    return "NA" if not math.isfinite(x) else f"{x:.6f}"

def loss_summary(path):
    if not os.path.exists(path):
        return None, None, None
    hist = (json.load(open(path)).get("history") or [])
    if not hist:
        return None, None, None
    first = hist[0]
    best = min(hist, key=lambda r: float(r.get("loss", "inf")))
    last = hist[-1]
    return first.get("loss"), best.get("loss"), last.get("loss")

def mask_tuple(layers):
    return tuple(sorted(int(x) for x in layers))

def pairwise_hamming(keys, n_layers=28):
    keys = [set(k) for k in keys]
    if len(keys) < 2:
        return 0.0
    vals = []
    for a, b in itertools.combinations(keys, 2):
        vals.append(len(a ^ b) / n_layers)
    return sum(vals) / len(vals)

def overlap_summary(path):
    if not os.path.exists(path):
        return (None,) * 6
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        return (None,) * 6
    overlap = sum(float(r["overlap_ratio"]) for r in rows) / len(rows)
    hamming = sum(float(r["hamming_ratio"]) for r in rows) / len(rows)
    exact = sum(float(r["exact_match"]) for r in rows) / len(rows)
    teacher = [mask_tuple(r["label_skipped_layers"]) for r in rows]
    pred = [mask_tuple(r["predicted_skipped_layers"]) for r in rows]
    return overlap, hamming, pairwise_hamming(pred), exact, len(set(teacher)), len(set(pred))

print("| method | loss first | loss best | loss last | overlap@7 | hamming | pairwise predicted hamming | exact match | unique teacher masks | unique predicted masks |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, p in paths.items():
    first, best, last = loss_summary(p["train"])
    overlap, hamming, pairwise, exact, uniq_teacher, uniq_pred = overlap_summary(p["overlap"])
    print(f"| {name} | {fmt(first)} | {fmt(best)} | {fmt(last)} | {fmt(overlap)} | {fmt(hamming)} | {fmt(pairwise)} | {fmt(exact)} | {uniq_teacher or 'NA'} | {uniq_pred or 'NA'} |")
PY
```

## OPAL Tuning Table

| setting | status | Full NLL/PPL | Raw-SetBCE NLL/PPL | OPAL-SetBCE NLL/PPL | conclusion |
|---|---|---:|---:|---:|---|
| prefix tokens 256 | complete | 2.2154 / 9.1649 | 2.7618 / 15.8283 | 2.7672 / 15.9138 | OPAL beats static/PuDDing/IG but loses Raw by 0.0054 NLL. |
| prefix tokens 512 | complete | 2.1834 / 8.8763 | 2.7443 / 15.5530 | 2.7586 / 15.7777 | OPAL still loses Raw by 0.0143 NLL; do not expand SetBCE to 3 seeds. |
| validation checkpoint selection | pending | NA | NA | NA | Only worthwhile if an OPAL variant first beats Raw on seed42. |
| static prior / swap q=1/2 | pending | NA | NA | NA | Defer unless old OPAL baselines or prefix variants show a path to beat Raw. |

Do not compare absolute Full PPL across prefix lengths: `prefix256` scores 768 suffix tokens per window, while `prefix512` scores 512 suffix tokens per window. Compare methods within the same prefix setting using Delta_NLL/Delta_PPL.

## Gap Table

| missing item | status | priority | why it matters | action |
|---|---|---:|---|---|
| random_dynamic_hash | missing | P2 | sanity lower-bound for prompt-conditioned dynamic mask diversity | add/run only after old OPAL baselines are in |
| OPAL-PrefixLast old | missing on WikiText-2 | P0 | old submission-style OPAL reference; needed to know whether SetBCE is worse than previous OPAL formulation | run seed42 first |
| OPAL-Attn old one-layer | missing on WikiText-2 | P0 | old one-layer attention OPAL ablation; needed for continuity with prior tables | run seed42 first |
| static best-on-val C16 | optional missing | P2 | PuDDing/IG use C16; static best currently C6, so C16 static can check whether candidate library itself contains a stronger static mask | optional after P0 |
| 3 seeds | missing | P2 | needed only if a seed42 setting becomes paper-worthy | do not run until prefix512 or old OPAL wins Raw |
| prefix length sensitivity | seed42 prefix512 complete | P1 | likely rescue axis for OPAL-SetBCE on long LM windows | did not rescue SetBCE; OPAL still loses Raw |
| validation checkpoint selection | missing | P1 | current training may not choose best validation-PPL checkpoint | add only if prefix512 still looks promising |

## Next Minimal Experiments

Priority order:

1. Run OPAL-PrefixLast old and OPAL-Attn old one-layer on WikiText-2 seed42. These are the most important missing historical OPAL baselines.
2. Do not expand OPAL-SetBCE prefix512 to three seeds: seed42 still loses Raw-SetBCE.
3. Expand to three seeds only if an old OPAL baseline, static-prior variant, or another clearly specified OPAL variant beats Raw-SetBCE on seed42.

Prefix512 rescue command already run:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
git pull --ff-only origin codex/opal-llm-experiments

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=2000 \
WIKITEXT_EVAL_WINDOWS=512 \
WIKITEXT_SEQ_LEN=1024 \
WIKITEXT_ROUTER_PREFIX_TOKENS=512 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58300 \
bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
```

Old OPAL baselines are not wired into the current WikiText-2 runner yet. Do not claim they were run until a WikiText-specific old-baseline harness reports `OPAL-PrefixLast old` and `OPAL-Attn old one-layer` with the same model, split, K, protected policy, and suffix-PPL evaluator.

## Final Judgment

- current decision: **appendix only**
- reason: OPAL beats Static best-on-val C6, PuDDing-style, and IG-style, but loses to Raw-SetBCE and the stronger-access layerwise_hidden_router. The prefix512 seed42 rescue also loses Raw-SetBCE.
- recommended wording: WikiText-2 is a public LM sanity / partial generalization result, not a main OPAL superiority claim.
- do not write: "OPAL is best on WikiText-2." The table does not support that.
- safe write: "On WikiText-2, OPAL improves over fixed/candidate-library skipping baselines but remains slightly behind a raw-prefix set router and a stronger-access layerwise hidden router."

## Commands

Run related baselines:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
git pull --ff-only origin codex/opal-llm-experiments

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=2000 \
WIKITEXT_EVAL_WINDOWS=512 \
WIKITEXT_SEQ_LEN=1024 \
WIKITEXT_ROUTER_PREFIX_TOKENS=256 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58200 \
WIKITEXT_RUN_PUDDING=1 \
WIKITEXT_RUN_IG=1 \
WIKITEXT_RUN_LAYERWISE=1 \
bash ./run_wikitext2_related_baselines_gpu01234567.sh
```

Minimal OPAL correction if needed:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=2000 \
WIKITEXT_EVAL_WINDOWS=512 \
WIKITEXT_SEQ_LEN=1024 \
WIKITEXT_ROUTER_PREFIX_TOKENS=512 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58300 \
bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
```

## Artifact Paths

- base metrics: `results/wikitext2_public_lm_sanity/metrics/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/`
- greedy labels: `results/wikitext2_public_lm_sanity/labels/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_delta_nll_greedy_set_labels.jsonl`
- candidate labels: `results/wikitext2_public_lm_sanity/related_labels/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_c16_candidate_delta_nll.jsonl`
- related metrics: `results/wikitext2_public_lm_sanity/related_metrics/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/`
- related checkpoints: `policy_ckpts/wikitext2_public_lm_related/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/`
