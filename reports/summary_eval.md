# 离线评测汇总（reports 自动统计）

- 统计目录：`reports`
- 文件数量：**6**
- like 阈值：3, 3.5, 4
- K 列表：10, 20
- 模型：emb, itemcf, popularity

## like = 3

### K = 10（users = 2717）

| model | P@K | R@K | NDCG@K | Hit@K | source |
|---|---:|---:|---:|---:|---|
| emb | 0.002356 | 0.023555 | 0.012411 | 0.023555 | `offline_eval_k10_like3.0.csv` |
| itemcf | 1.47e-04 | 0.001472 | 8.52e-04 | 0.001472 | `offline_eval_k10_like3.0.csv` |
| popularity | 5.52e-04 | 0.005521 | 0.003485 | 0.005521 | `offline_eval_k10_like3.0.csv` |

### K = 20（users = 2717）

| model | P@K | R@K | NDCG@K | Hit@K | source |
|---|---:|---:|---:|---:|---|
| emb | 0.002264 | 0.045271 | 0.017838 | 0.045271 | `offline_eval_k20_like3.0.csv` |
| itemcf | 1.47e-04 | 0.002944 | 0.001223 | 0.002944 | `offline_eval_k20_like3.0.csv` |
| popularity | 5.15e-04 | 0.010305 | 0.00471 | 0.010305 | `offline_eval_k20_like3.0.csv` |

## like = 3.5

### K = 10（users = 2377）

| model | P@K | R@K | NDCG@K | Hit@K | source |
|---|---:|---:|---:|---:|---|
| emb | 0.002524 | 0.025242 | 0.01336 | 0.025242 | `offline_eval_k10_like3.5.csv` |
| itemcf | 2.52e-04 | 0.002524 | 0.001577 | 0.002524 | `offline_eval_k10_like3.5.csv` |
| popularity | 7.15e-04 | 0.007152 | 0.004054 | 0.007152 | `offline_eval_k10_like3.5.csv` |

### K = 20（users = 2377）

| model | P@K | R@K | NDCG@K | Hit@K | source |
|---|---:|---:|---:|---:|---|
| emb | 0.002419 | 0.04838 | 0.019161 | 0.04838 | `offline_eval_k20_like3.5.csv` |
| itemcf | 1.68e-04 | 0.003366 | 0.001796 | 0.003366 | `offline_eval_k20_like3.5.csv` |
| popularity | 5.68e-04 | 0.011359 | 0.005096 | 0.011359 | `offline_eval_k20_like3.5.csv` |

## like = 4

### K = 10（users = 1952）

| model | P@K | R@K | NDCG@K | Hit@K | source |
|---|---:|---:|---:|---:|---|
| emb | 0.003074 | 0.030738 | 0.016269 | 0.030738 | `offline_eval_k10_like4.0.csv` |
| itemcf | 4.10e-04 | 0.004098 | 0.001667 | 0.004098 | `offline_eval_k10_like4.0.csv` |
| popularity | 7.17e-04 | 0.007172 | 0.004508 | 0.007172 | `offline_eval_k10_like4.0.csv` |

### K = 20（users = 1952）

| model | P@K | R@K | NDCG@K | Hit@K | source |
|---|---:|---:|---:|---:|---|
| emb | 0.002946 | 0.058914 | 0.023333 | 0.058914 | `offline_eval_k20_like4.0.csv` |
| itemcf | 3.84e-04 | 0.007684 | 0.002572 | 0.007684 | `offline_eval_k20_like4.0.csv` |
| popularity | 7.17e-04 | 0.014344 | 0.006328 | 0.014344 | `offline_eval_k20_like4.0.csv` |

## 自动结论（便于写报告/答辩）

> 规则：在同一个 like、K 下，用 **NDCG@K** 选最优模型（越大越好）。

| like | K | best_model_by_ndcg | best_ndcg | notes |
|---:|---:|---|---:|---|
| 3 | 10 | emb | 0.012411 | — |
| 3 | 20 | emb | 0.017838 | — |
| 3.5 | 10 | emb | 0.01336 | — |
| 3.5 | 20 | emb | 0.019161 | — |
| 4 | 10 | emb | 0.016269 | — |
| 4 | 20 | emb | 0.023333 | — |

