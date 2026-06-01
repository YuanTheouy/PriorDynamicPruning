# OPAL-SetAttn v1 P2 Results

日期：2026-06-01

## 任务

```text
dataset = Office_Products
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

## 主结果

按 `Delta_NLL` 排序，越低越好：

| method | loss | Delta_NLL ↓ | Delta_PPL ↓ | KL ↓ | NDCG@10 | retention |
|---|---|---:|---:|---:|---:|---:|
| OPAL-SetAttn v1 | Exact K CE | **-0.657853** | **-15.293803** | 1.265450 | **0.129351** | **0.933027** |
| OPAL-SetAttn v1 | BCE | -0.587349 | -14.093351 | **1.101963** | 0.126038 | 0.909129 |

## 和 P2 seed42 旧最强对照

旧最强 row：

```text
OPAL-Attn + BCE
Delta_NLL = -0.660716
Delta_PPL = -15.340786
KL = 1.151651
NDCG@10 = 0.127205
retention = 0.917551
```

SetAttn v1 Exact K CE 与旧 OPAL-Attn+BCE 的主指标差距：

```text
Delta_NLL gap = +0.002863  # SetAttn less negative, slightly worse
Delta_PPL gap = +0.046983  # SetAttn less negative, slightly worse
```

但 SetAttn v1 Exact K CE 的 downstream 更好：

```text
NDCG@10:   0.129351 vs 0.127205
retention: 0.933027 vs 0.917551
```

## Train-label overlap

| loss | exact match | overlap@7 | hamming | pairwise order acc | unique predicted masks |
|---|---:|---:|---:|---:|---:|
| BCE | 0.0045 | 4.0445 | 5.9110 | **0.8332** | 101 |
| Exact K CE | **0.0180** | **4.3820** | **5.2360** | 0.8004 | **782** |

## 判断

1. SetAttn v1 结构没有明显超过旧 `OPAL-Attn + BCE` 的主指标。
2. SetAttn v1 Exact K CE 是本轮 SetAttn 内部赢家，和旧最强几乎打平，并且 NDCG/retention 更好。
3. SetAttn v1 BCE 明显弱于 SetAttn Exact K CE，也弱于旧 OPAL-Attn+BCE。
4. Exact K CE 在 SetAttn 上表现反转：它不再像旧 P2 里那样弱于 BCE，而是明显优于 SetAttn BCE。
5. 先不要扩 seed13 / seed3407；下一步应先回填训练曲线、检查 SetAttn BCE 为什么退化、确认 Exact K CE 是否因为 unique masks 更高而提升。

## 下一步只读提取命令

在服务器新终端中执行，用来复制完整 rows / metrics / overlap metadata：

```bash
cd /workspace/PriorDynamicPruning

RUN_GROUP=opal_setattn_v1_delta_nll_greedy_set_m2000_seed42
OUT=results/planrec_experiments
DIAG=results/opal_greedy_set_diagnostics
CKPT=policy_ckpts/opal_setattn_v1/${RUN_GROUP}

echo "=== summary csv ==="
cat "${OUT}/tables/summary_${RUN_GROUP}.csv"

echo "=== BCE training metrics ==="
cat "${CKPT}/prefix_hk_raw_setattn_bce/training_metrics.json"

echo "=== Exact K CE training metrics ==="
cat "${CKPT}/prefix_hk_raw_setattn_exact_k_ce/training_metrics.json"

echo "=== BCE overlap summary ==="
cat "${DIAG}/${RUN_GROUP}_prefix_hk_raw_setattn_bce_train_overlap.summary.json"

echo "=== Exact K CE overlap summary ==="
cat "${DIAG}/${RUN_GROUP}_prefix_hk_raw_setattn_exact_k_ce_train_overlap.summary.json"

echo "=== raw eval json names ==="
ls -lh "${OUT}/raw_json/"*"${RUN_GROUP}"*.json
```

如果只想复制最短汇总：

```bash
cd /workspace/PriorDynamicPruning
RUN_GROUP=opal_setattn_v1_delta_nll_greedy_set_m2000_seed42
python3 - <<'PY'
import csv, json
from pathlib import Path

run_group = "opal_setattn_v1_delta_nll_greedy_set_m2000_seed42"
out = Path("results/planrec_experiments")
diag = Path("results/opal_greedy_set_diagnostics")

print("=== compact summary rows ===")
for row in csv.DictReader((out / "tables" / f"summary_{run_group}.csv").open()):
    print(json.dumps({
        "run_name": row.get("run_name"),
        "Delta_NLL": row.get("Delta_NLL"),
        "Delta_PPL": row.get("Delta_PPL"),
        "KL_full_to_skip": row.get("KL_full_to_skip"),
        "NDCG@10": row.get("NDCG@10"),
        "retention_NDCG@10": row.get("retention_NDCG@10"),
        "risk_router_input": row.get("risk_router_input"),
    }, ensure_ascii=False))

print("=== compact overlap rows ===")
for loss in ("bce", "exact_k_ce"):
    path = diag / f"{run_group}_prefix_hk_raw_setattn_{loss}_train_overlap.summary.json"
    data = json.load(path.open())
    print(json.dumps({
        "loss": loss,
        "exact_match_rate": data.get("exact_match_rate"),
        "mean_overlap_count": data.get("mean_overlap_count"),
        "mean_hamming_count": data.get("mean_hamming_count"),
        "mean_pairwise_order_accuracy": data.get("mean_pairwise_order_accuracy"),
        "unique_predicted_masks": data.get("unique_predicted_masks"),
    }, ensure_ascii=False))
PY
```
