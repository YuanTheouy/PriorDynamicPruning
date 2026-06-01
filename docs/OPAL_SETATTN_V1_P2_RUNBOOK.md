# OPAL-SetAttn v1 P2 Runbook

日期：2026-06-01

## 当前明确任务

只跑一个任务：

```text
Office_Products
seed = 42
teacher = Delta_NLL forward greedy skip set
label samples = 2000
skip_rate = 0.25
keep = 21 / skip = 7
protected_head = 4
protected_tail = 2
router = prefix_hk_raw_setattn
losses = BCE set supervision + Exact K CE
epochs = 40
eval max batches = 500
```

不要现在跑 seed13 / seed3407，不要跑 Industrial25 / Office36。

## 服务器命令

在 8 卡 A100 tmux 中执行：

```bash
cd /workspace/PriorDynamicPruning
git pull --ff-only origin codex/opal-llm-experiments
bash ./run_opal_setattn_v1_p2_seed42_gpu01234567.sh
```

如果端口冲突：

```bash
OPAL_BASE_PORT=56400 bash ./run_opal_setattn_v1_p2_seed42_gpu01234567.sh
```

## 脚本会做什么

1. 生成或复用 `Office_Products_delta_nll_greedy_set_m2000_seed42_head4_tail2.jsonl`。
2. 训练 `prefix_hk_raw_setattn_bce`。
3. 训练 `prefix_hk_raw_setattn_exact_k_ce`。
4. 评估 BCE checkpoint。
5. 评估 Exact K CE checkpoint。
6. 对两者跑 train-label overlap diagnostic。
7. 写 summary csv，并在终端打印 compact rows。

## 关键输出路径

```text
policy_ckpts/opal_setattn_v1/opal_setattn_v1_delta_nll_greedy_set_m2000_seed42/
results/planrec_experiments/tables/summary_opal_setattn_v1_delta_nll_greedy_set_m2000_seed42.csv
results/opal_greedy_set_diagnostics/opal_setattn_v1_delta_nll_greedy_set_m2000_seed42_prefix_hk_raw_setattn_bce_train_overlap.summary.json
results/opal_greedy_set_diagnostics/opal_setattn_v1_delta_nll_greedy_set_m2000_seed42_prefix_hk_raw_setattn_exact_k_ce_train_overlap.summary.json
```

## 跑完后回填

把脚本最后的 `Compact result rows` 和 `overlap summary` 两段贴回给 Codex。然后写入：

```text
docs/OPAL_SETATTN_V1_P2_RESULTS.md
```

判断只看：

```text
Delta_NLL
Delta_PPL
NDCG@10 / retention
KL_full_to_skip 作为次指标
overlap@7 / hamming / pairwise order accuracy
unique_predicted_masks
```

## 决策规则

如果 `prefix_hk_raw_setattn_bce` 没有明显超过旧 `OPAL-Attn + BCE` seed42：

```text
old OPAL-Attn + BCE Delta_NLL = -0.660716
old OPAL-Attn + BCE Delta_PPL = -15.340786
```

就先不要扩 seed，先分析训练/overlap/unique masks。

如果 SetAttn BCE 明显超过旧 OPAL-Attn BCE，再继续 seed13 / seed3407。

Exact K CE 只作为兼容和对照；不要因为 Exact K CE 输给 BCE 就否定 SetAttn。
