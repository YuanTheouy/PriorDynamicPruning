# Related Baseline Reproduction Results

Date: 2026-05-30

## Submission Convergence Baseline Matrix

The submission round freezes the main method as:

```text
OPAL-PrefixLast = prefix_hk_last + one-layer Delta_NLL labels + hard top-K skip.
```

`prefix_hk_raw_attn` is retained as `OPAL-Attn` for ablation/analysis, not as
the main method.

Runner:

```bash
cd /workspace/PriorDynamicPruning

SUBMISSION_TASKS=office25 \
SUBMISSION_SEEDS="42" \
bash ./run_submission_convergence_experiments_gpu01234567.sh 2>&1 | tee /tmp/submission_office25_seed42.log
```

Full priority-1 run:

```bash
cd /workspace/PriorDynamicPruning

SUBMISSION_TASKS=office25 \
SUBMISSION_SEEDS="42 13 3407" \
bash ./run_submission_convergence_experiments_gpu01234567.sh 2>&1 | tee /tmp/submission_office25_3seeds.log
```

Related baseline rows produced by the runner:

| baseline | evaluator method | training/calibration signal | inference context | same budget |
|---|---|---|---|---|
| random dynamic mask | `input_guided` with prompt hash | none | prompt-only hash features | 7 skipped / 21 kept |
| PuDDing-style prompt candidate | `pudding_prompt_candidate` | offline candidate-mask Delta_NLL labels | prompt-only raw embedding mean | 7 skipped / 21 kept |
| IG-style cluster mask | `ig_cluster_mask` | cluster-average one-layer Delta_NLL risk | prompt-only raw embedding mean + nearest cluster | 7 skipped / 21 kept |
| layerwise_hidden_router / Dr.LLM-style | `layerwise_hidden_router` | one-layer Delta_NLL risk labels | prompt-only per-layer hidden states | 7 skipped / 21 kept |
| static best-on-val | `static` with selected mask library | validation candidate-mask Delta_NLL | fixed global mask on test | 7 skipped / 21 kept |

Status:

| item | status |
|---|---|
| Office_Products 25% skip seed42 | runner ready; pending server run |
| Office_Products 25% skip seeds 42/13/3407 | runner ready; pending server run |
| Industrial_and_Scientific 25% skip seed42 | runner ready; pending server run |
| Office_Products second skip rate seed42 | runner ready; pending server run |

Leakage rule:

```text
Evaluation is prompt-only. Target/answer tokens can be used only offline for
training labels or validation selection, never inside per-example test-time
routing.
```
