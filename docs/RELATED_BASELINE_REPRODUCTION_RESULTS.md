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

### Office Products 25% Skip Related Baseline Results

Completed on `Office_Products`, seeds `42 / 13 / 3407`, same 7 skipped / 21 kept
budget. Three-seed averages:

| method | mean Delta_NLL ↓ | mean Delta_PPL ↓ | mean KL ↓ | mean NDCG@10 | mean retention | mean unique_masks |
|---|---:|---:|---:|---:|---:|---:|
| OPAL-PrefixLast | **-0.224583** | **-4.882415** | 1.697355 | 0.1598 | 0.8810 | 54.3 |
| raw_input_risk | -0.149916 | -3.334785 | 1.738470 | **0.1619** | **0.8923** | 29.3 |
| OPAL-Attn | -0.137910 | -3.115666 | 1.663331 | 0.1503 | 0.8286 | 9.3 |
| static best-on-val | -0.005696 | 0.187665 | **1.474916** | 0.1593 | 0.8780 | 1.0 |
| IG-style cluster | -0.000690 | -0.001417 | 1.818229 | 0.1532 | 0.8445 | 7.3 |
| PuDDing-style candidate | 0.089038 | 2.318122 | 1.711482 | 0.1494 | 0.8235 | 2.3 |
| layerwise hidden router | 0.099545 | 2.663132 | 1.937508 | 0.1562 | 0.8609 | 23.3 |
| random dynamic hash | 1.764607 | 123.841179 | 2.690828 | 0.1429 | 0.7877 | 12.0 |

Related-baseline conclusion:

- OPAL-PrefixLast beats all implemented related baselines on mean `Delta_NLL` and `Delta_PPL`.
- static best-on-val has the best mean KL, mostly due to seed 3407 selecting a strong validation mask, but it does not preserve likelihood as well as OPAL-PrefixLast.
- PuDDing-style, IG-style, layerwise hidden router, and random dynamic hash do not outperform OPAL-PrefixLast under the fixed budget.
