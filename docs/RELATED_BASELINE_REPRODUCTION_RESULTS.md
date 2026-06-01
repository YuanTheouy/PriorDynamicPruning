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
| Office_Products 25% skip seeds 42/13/3407 | completed and recorded |
| Office_Products second skip rate seeds 42/13/3407 | completed and recorded |
| Industrial_and_Scientific 25% skip seed42 | not yet present in this local note; paste or sync the summary table to record |

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

### Industrial and Scientific 25% Skip Related Baseline Results

Completed on `Industrial_and_Scientific`, seeds `42 / 13 / 3407`, same 7
skipped / 21 kept budget.

| method | mean Delta_NLL ↓ | mean Delta_PPL ↓ | mean KL ↓ | mean NDCG@10 | mean retention | mean unique_masks |
|---|---:|---:|---:|---:|---:|---:|
| static best-on-val | **-0.299121** | **-5.042637** | **0.988811** | 0.1134 | 0.8356 | 1.0 |
| static ends_heavy | **-0.299121** | **-5.042637** | **0.988811** | 0.1134 | 0.8356 | 1.0 |
| raw_input_risk | -0.002367 | 0.611307 | 1.786387 | 0.1206 | 0.8882 | 82.7 |
| OPAL-PrefixLast | 0.053427 | 2.859912 | 1.873676 | 0.1185 | 0.8729 | 155.0 |
| OPAL-Attn | 0.122975 | 2.689344 | 1.955186 | 0.1188 | 0.8748 | 157.0 |
| IG-style cluster | 0.123106 | 2.558113 | 1.772026 | 0.1163 | 0.8569 | 7.0 |
| static uniform | 1.029266 | 35.089726 | 1.769962 | **0.1248** | **0.9194** | 1.0 |
| random dynamic hash | 1.609992 | 83.726653 | 2.409829 | 0.1153 | 0.8490 | 12.0 |

Related-baseline second-dataset conclusion:

- Industrial is a mixed result: validation-selected `static best-on-val` collapses to `ends_heavy_k21` and is the strongest likelihood/KL row.
- `raw_input_risk` is the strongest dynamic row on `Delta_NLL`, while OPAL-PrefixLast and OPAL-Attn remain competitive but do not beat the static ends-heavy mask.
- static uniform has the best NDCG/retention but poor likelihood preservation, so it is not the primary row under the LLM acceleration objective.
- This table is useful for paper honesty: OPAL-PrefixLast wins Office25, but the second dataset exposes strong static-mask behavior.

### Office Products 35.7% Skip Related Baseline Stress Results

Completed on `Office_Products`, seeds `42 / 13 / 3407`, same 10 skipped / 18
kept budget. This is the second skip-rate stress test.

| method | mean Delta_NLL ↓ | mean Delta_PPL ↓ | mean KL ↓ | mean NDCG@10 | mean retention | mean unique_masks |
|---|---:|---:|---:|---:|---:|---:|
| static best-on-val | **0.328888** | **10.766920** | 2.281277 | **0.1282** | **0.7066** | 1.0 |
| raw_input_risk | 0.509111 | 16.592211 | 2.503581 | 0.1174 | 0.6469 | 50.7 |
| OPAL-PrefixLast | 0.512090 | 16.749792 | 2.489585 | 0.1149 | 0.6338 | 92.7 |
| OPAL-Attn | 0.514535 | 16.959152 | 2.491641 | 0.1190 | 0.6560 | 133.7 |
| IG-style cluster | 0.648491 | 22.758193 | 2.545443 | 0.1061 | 0.5847 | 6.7 |
| static ends_heavy | 0.873638 | 34.731965 | **2.270256** | 0.0924 | 0.5095 | 1.0 |
| static uniform | 2.498949 | 277.975715 | 4.468617 | 0.0818 | 0.4510 | 1.0 |
| random dynamic hash | 3.302239 | 692.385318 | 4.219182 | 0.1165 | 0.6420 | 13.3 |

Related-baseline stress-test conclusion:

- Validation-selected `static best-on-val` is the strongest average row at this heavier skip rate.
- OPAL-PrefixLast, OPAL-Attn, and raw_input_risk are closely matched and remain much better than random dynamic, IG-style cluster, and static uniform on likelihood preservation.
- OPAL-PrefixLast has the best dynamic KL, while OPAL-Attn has the best dynamic NDCG/retention in this stress setting.
- This should be reported as a stress-test tradeoff rather than as the main evidence table; the 25% Office table remains the cleanest main-result setting for OPAL-PrefixLast.
