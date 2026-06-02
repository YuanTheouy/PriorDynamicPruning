# LLM Eval Downstream Results

Last updated: 2026-06-02

## Status

Pending server run. The downstream runner is:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
git pull --ff-only origin codex/opal-llm-experiments
bash ./run_llmeval_downstream_gpu01234567.sh
```

This is a no-compensation main table run through `lm-evaluation-harness` (`simple_evaluate`). It consumes existing WikiText-trained routers/artifacts and does not rerun WikiText PPL, label construction, or router training.

## Fixed Setup

- model: `/workspace/ckpts/Qwen2.5-1.5B`
- evaluator: `lm-evaluation-harness`
- tasks: `piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge`
- seeds: `42,13,3407`
- max_length: `1024`
- router_prefix_tokens: `256`
- skip_rate / skip_count: `0.25 / 7`
- protected_head / protected_tail: `4 / 2`
- compensation: `none`

## Required Methods

- Full
- Static ends_heavy
- Static best-on-val
- PuDDing-style
- IG-style
- layerwise_hidden_router
- Raw-SetBCE best-on-val
- OPAL-SetBCE best-on-val

## Required Readout

When the server run finishes, this file must contain:

- every task x method x seed result
- three-seed mean/std by task
- six-task average
- OPAL best-on-val comparison against Raw, layerwise, PuDDing, IG, and static baselines
- PuDDing/IG candidate-selection distributions
- OPAL best-on-val `unique_masks` caveat if it is low
