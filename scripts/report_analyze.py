#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reports 评测结果汇总 & 可视化工具

功能：
- 扫描 reports/ 下的离线评测结果文件：offline_eval_k{K}_like{LIKE}.csv
- 合并为一张总表：reports/summary_eval.csv
- 生成一份 Markdown 汇总：reports/summary_eval.md
- 生成可视化图表（PNG）：reports/plots/

依赖：
- 仅标准库即可生成汇总表与 Markdown
- 若环境安装了 matplotlib，则会额外生成 PNG 图（推荐）

用法（在项目根目录）：
  python3 scripts/report_analyze.py
  python3 scripts/report_analyze.py --reports_dir reports --out_dir reports
  python3 scripts/report_analyze.py --no_plots
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# 注意：这里用正则从文件名解析出 k 与 like
# e.g. offline_eval_k20_like3.5.csv
FILE_RE = re.compile(r"offline_eval_k(?P<k>\d+)_like(?P<like>[0-9.]+)\.csv$")


@dataclass(frozen=True)
class Row:
    k: int
    like: float
    model: str
    users: int
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    hit_at_k: float
    source_file: str


def parse_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        # 兜底：有些科学计数法可能带奇怪空格
        return float(s.replace(" ", ""))


def discover_report_files(reports_dir: str) -> List[Tuple[str, int, float]]:
    files = sorted(glob.glob(os.path.join(reports_dir, "offline_eval_k*_like*.csv")))
    out: List[Tuple[str, int, float]] = []
    for fp in files:
        name = os.path.basename(fp)
        m = FILE_RE.match(name)
        if not m:
            continue
        k = int(m.group("k"))
        like = float(m.group("like"))
        out.append((fp, k, like))
    return out


def read_one_report(fp: str, k: int, like: float) -> List[Row]:
    rows: List[Row] = []
    with open(fp, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line in r:
            model = (line.get("model") or "").strip()
            if not model:
                continue
            rows.append(
                Row(
                    k=k,
                    like=like,
                    model=model,
                    users=int(parse_float(line.get("users") or "0")),
                    precision_at_k=parse_float(line.get("precision_at_k")),
                    recall_at_k=parse_float(line.get("recall_at_k")),
                    ndcg_at_k=parse_float(line.get("ndcg_at_k")),
                    hit_at_k=parse_float(line.get("hit_at_k")),
                    source_file=os.path.basename(fp),
                )
            )
    return rows


def fmt_float(x: float, ndigits: int = 6) -> str:
    # 小数很小的时候，统一用科学计数法更易读；否则用定点
    if x == 0.0:
        return "0"
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.{ndigits}f}".rstrip("0").rstrip(".")


def write_summary_csv(rows: List[Row], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r.like, r.k, r.model))
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "like",
                "k",
                "model",
                "users",
                "precision_at_k",
                "recall_at_k",
                "ndcg_at_k",
                "hit_at_k",
                "source_file",
            ]
        )
        for r in rows_sorted:
            w.writerow(
                [
                    fmt_float(r.like, 3),
                    r.k,
                    r.model,
                    r.users,
                    r.precision_at_k,
                    r.recall_at_k,
                    r.ndcg_at_k,
                    r.hit_at_k,
                    r.source_file,
                ]
            )


def build_markdown(rows: List[Row], reports_dir: str) -> str:
    if not rows:
        return "# 离线评测汇总\n\n未发现可用的 `offline_eval_k*_like*.csv`。\n"

    # 分组：like -> k -> model
    likes = sorted({r.like for r in rows})
    ks = sorted({r.k for r in rows})
    models = sorted({r.model for r in rows})

    lines: List[str] = []
    lines.append("# 离线评测汇总（reports 自动统计）")
    lines.append("")
    lines.append(f"- 统计目录：`{reports_dir}`")
    lines.append(f"- 文件数量：**{len({r.source_file for r in rows})}**")
    lines.append(f"- like 阈值：{', '.join(fmt_float(x, 3) for x in likes)}")
    lines.append(f"- K 列表：{', '.join(str(k) for k in ks)}")
    lines.append(f"- 模型：{', '.join(models)}")
    lines.append("")

    def table_for_like(like: float) -> None:
        subset = [r for r in rows if r.like == like]
        if not subset:
            return
        lines.append(f"## like = {fmt_float(like, 3)}")
        lines.append("")
        # 每个 K 一张表，方便截图放 PPT
        for k in sorted({r.k for r in subset}):
            t = [r for r in subset if r.k == k]
            if not t:
                continue
            # users 可能在不同 model 一致（通常一致），取最大做展示
            users = max(r.users for r in t)
            lines.append(f"### K = {k}（users = {users}）")
            lines.append("")
            lines.append("| model | P@K | R@K | NDCG@K | Hit@K | source |")
            lines.append("|---|---:|---:|---:|---:|---|")
            t_sorted = sorted(t, key=lambda r: r.model)
            for r in t_sorted:
                lines.append(
                    f"| {r.model} | {fmt_float(r.precision_at_k)} | {fmt_float(r.recall_at_k)} | {fmt_float(r.ndcg_at_k)} | {fmt_float(r.hit_at_k)} | `{r.source_file}` |"
                )
            lines.append("")

    for like in likes:
        table_for_like(like)

    # 简要建议：自动挑一个表现最好模型（按 NDCG@K）
    # 取“like, k”维度内的最佳模型（每组一条）
    lines.append("## 自动结论（便于写报告/答辩）")
    lines.append("")
    lines.append("> 规则：在同一个 like、K 下，用 **NDCG@K** 选最优模型（越大越好）。")
    lines.append("")
    lines.append("| like | K | best_model_by_ndcg | best_ndcg | notes |")
    lines.append("|---:|---:|---|---:|---|")
    for like in likes:
        for k in ks:
            group = [r for r in rows if r.like == like and r.k == k]
            if not group:
                continue
            best = max(group, key=lambda r: r.ndcg_at_k)
            notes = "—"
            # 如果最佳和第二名差异很小，提示“差距不大”
            sorted_by = sorted(group, key=lambda r: r.ndcg_at_k, reverse=True)
            if len(sorted_by) >= 2 and (sorted_by[0].ndcg_at_k - sorted_by[1].ndcg_at_k) < 1e-3:
                notes = "差距不大（<1e-3）"
            lines.append(
                f"| {fmt_float(like, 3)} | {k} | {best.model} | {fmt_float(best.ndcg_at_k)} | {notes} |"
            )
    lines.append("")

    return "\n".join(lines) + "\n"


def try_plot(rows: List[Row], plots_dir: str) -> Tuple[bool, Optional[str]]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # noqa: BLE001
        return False, f"matplotlib 不可用：{e}"

    os.makedirs(plots_dir, exist_ok=True)

    likes = sorted({r.like for r in rows})
    models = sorted({r.model for r in rows})

    def metric_getter(metric: str):
        if metric == "precision":
            return lambda r: r.precision_at_k
        if metric == "recall":
            return lambda r: r.recall_at_k
        if metric == "ndcg":
            return lambda r: r.ndcg_at_k
        if metric == "hit":
            return lambda r: r.hit_at_k
        raise ValueError(metric)

    metrics = [("precision", "P@K"), ("recall", "R@K"), ("ndcg", "NDCG@K"), ("hit", "Hit@K")]

    # 每个 like 生成一个 2x2 总图（更适合答辩截图）
    for like in likes:
        subset = [r for r in rows if r.like == like]
        if not subset:
            continue
        ks = sorted({r.k for r in subset})
        if not ks:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
        fig.suptitle(f"Offline Eval (like={fmt_float(like, 3)})", fontsize=14)
        axes = axes.flatten()

        for ax, (metric, label) in zip(axes, metrics):
            get = metric_getter(metric)
            for model in models:
                pts = sorted([r for r in subset if r.model == model], key=lambda r: r.k)
                if not pts:
                    continue
                xs = [r.k for r in pts]
                ys = [get(r) for r in pts]
                if len(xs) == 1:
                    ax.scatter(xs, ys, label=model)
                else:
                    ax.plot(xs, ys, marker="o", label=model)
            ax.set_title(label)
            ax.set_xlabel("K")
            ax.grid(True, alpha=0.3)
            # 自动用科学计数法（数值很小）
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

        # 统一图例放右上角
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.tight_layout(rect=(0, 0, 0.95, 0.95))

        out = os.path.join(plots_dir, f"offline_eval_like{fmt_float(like, 3)}.png")
        fig.savefig(out)
        plt.close(fig)

    return True, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports", help="评测结果目录（默认：reports）")
    ap.add_argument("--out_dir", default="reports", help="输出目录（默认：reports）")
    ap.add_argument("--no_plots", action="store_true", help="只生成汇总表，不生成图")
    args = ap.parse_args()

    reports_dir = args.reports_dir
    out_dir = args.out_dir
    plots_dir = os.path.join(out_dir, "plots")

    files = discover_report_files(reports_dir)
    if not files:
        print(f"[warn] 未发现评测文件：{reports_dir}/offline_eval_k*_like*.csv")
        out_md = os.path.join(out_dir, "summary_eval.md")
        os.makedirs(out_dir, exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(build_markdown([], reports_dir))
        return

    all_rows: List[Row] = []
    for fp, k, like in files:
        all_rows.extend(read_one_report(fp, k, like))

    out_csv = os.path.join(out_dir, "summary_eval.csv")
    out_md = os.path.join(out_dir, "summary_eval.md")
    write_summary_csv(all_rows, out_csv)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(build_markdown(all_rows, reports_dir))

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_md}")

    if args.no_plots:
        print("[skip] no plots")
        return

    ok, err = try_plot(all_rows, plots_dir)
    if ok:
        print(f"[ok] plots saved to: {plots_dir}")
    else:
        print(f"[warn] plots not generated: {err}")
        print("      你可以安装 matplotlib：pip install matplotlib")


if __name__ == "__main__":
    main()


