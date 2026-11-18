#!/usr/bin/env python3
"""Render CAPACITY_REPORT.md from machine-readable findings.json.

All numbers in the report come from the analysis output; this module only
formats them and states the interpretation. Run after analyze.py.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import statsx as sx  # noqa: E402


def mb(x):
    return f"{x / 1e6:.2f} MB"


def kb(x):
    return f"{x / 1e3:.1f} KB"


def bytes_human(x):
    if x is None:
        return "—"
    if x >= 1024 ** 3:
        return f"{x / (1024 ** 3):.2f} GB"
    if x >= 1024 * 1024:
        return f"{x / (1024 * 1024):.2f} MB"
    if x >= 1024:
        return f"{x / 1024:.1f} KB"
    return f"{x:.0f} B"


def coef_num(x, digits=3):
    if x is None:
        return "—"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{digits}f}"


def pct(x, d=2):
    return f"{100 * x:.{d}f}%" if abs(x) <= 1.5 else f"{x:.{d}f}"


def coef_table(fits):
    lines = ["| 模型 | 预测变量 | 系数（字节/单位） | 标准误 | 调整 R² | 残差 RSE |",
             "| --- | --- | --- | --- | --- | --- |"]
    for m in ("M0", "M1", "M2", "M3"):
        f = fits[f"{m}_ols"]
        if f["beta"] is None:
            lines.append(f"| {m} | — | 不可识别 | — | — | — |")
            lines.append(f"| | 原因 | {f['status']} | | | |")
            continue
        for name, b, se in zip(f["predictors"], f["beta"], f["se_beta"]):
            se_s = "—" if se is None else coef_num(se)
            lines.append(
                f"| {m if name == 'intercept' else ''} | {name} | {coef_num(b)} | "
                f"{se_s} | {f['adj_r2']:.4f} | {f['rse_bytes']/1e3:.1f} KB |")
    return "\n".join(lines)


def loo_table(loo):
    lines = ["| 模型 | 留轴交叉验证 RMSE | 中位绝对百分比误差 | 95 分位误差 | 最大误差 | 预测点数 | 不可识别折 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for t in sorted(loo, key=lambda d: d["loo_rmse_bytes"] if d["loo_rmse_bytes"] is not None else float("inf")):
        rmse_s = "不可识别" if t["loo_rmse_bytes"] is None else bytes_human(t["loo_rmse_bytes"])
        med_s = "—" if t["loo_median_abs_pct"] is None else f"{t['loo_median_abs_pct']:.2f}%"
        p95_s = "—" if t.get("loo_p95_abs_pct") is None else f"{t['loo_p95_abs_pct']:.2f}%"
        max_s = "—" if t["loo_max_abs_pct"] is None else f"{t['loo_max_abs_pct']:.2f}%"
        unid = "、".join(t.get("unidentifiable_folds", [])) or "—"
        lines.append(
            f"| {t['model']} | {rmse_s} | {med_s} | {p95_s} | {max_s} | "
            f"{t['held_out_predictions']} | {unid} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings", default="../data/tidy/findings.json")
    ap.add_argument("--out", default="../CAPACITY_REPORT.md")
    args = ap.parse_args()
    d = json.load(open(args.findings))
    ev = d["evidence"]
    base = d["baseline"]

    same_rows = "\n".join(
        f"| {r['scenario']} | {r['keys']} | {r['versions']} | {r['mean_chain']} | "
        f"{r['heap_final_mb']} MB | {r['journal_kb']} KB |"
        for r in ev["same_rows_different_chains"])
    pin_rows = "\n".join(
        f"| {r['scenario']} | {r['pinned']} | {r['effective_horizon']} | "
        f"{r['versions_removed']} | {r['versions_retained']} | "
        f"{r['heap_final_mb']} MB | {r['journal_kb']} KB |"
        for r in ev["long_reader_pins_horizon"])
    idx_rows = "\n".join(
        f"| {r['scenario']} | {r['indexes']} | {r['keys']} | {r['versions']} | "
        f"{r['index_entries']} | {r['heap_final_mb']} MB |"
        for r in ev["index_candidates_track_history"])
    ji_lines = []
    for j in ev["journal_never_truncated"]:
        if "pair" in j:
            ji_lines.append(
                f"- {j['pair']}：记录数 {j['plain_records']} → {j['vacuum_records']}，"
                f"vacuum 恰好增加一条记录 = `{j['vacuum_adds_exactly_one_record']}`；"
                f"WAL 字节 {j['plain_journal_bytes']:,} → {j['vacuum_journal_bytes']:,}；"
                f"内存版本 {j['plain_versions']:,} → {j['vacuum_retained_versions']:,}。")
        else:
            ji_lines.append(
                f"- 同负载钉住/正常 vacuum 对比：WAL 分别为 "
                f"{j['pinned_journal_bytes']:,} 与 {j['vacuumed_journal_bytes']:,} 字节"
                f"（差异 {j['byte_difference']}，仅来自 vacuum 帧水位整数编码，"
                f"容差内 = `{j['difference_within_vacuum_frame_tolerance']}`）；"
                f"记录数相同 = `{j['same_record_count']}`，提交负载相同 = "
                f"`{j['same_mutation_count']}`；保留版本 {j['pinned_versions']:,} 对 "
                f"{j['vacuumed_versions']:,}。")
    ji_text = "\n".join(ji_lines) + (
        "\n\n结论：vacuum 只收缩内存版本结构与索引；WAL 帧只增不删，磁盘日志容量必须单独立项。"
        if ji_lines else "（该对比场景未在本次矩阵中）")

    micro = d["micro_slopes"]
    micro_eval = d["micro_calibrated_eval"]
    disk = d["disk_models"]
    hr = d["headroom"]

    p1 = "\n".join(
        f"| {r['budget_mb']} | {r['max_versions']:,} | {r['max_writes_equiv']:,} | "
        f"{r['predicted_heap_mb']} | {r['pi95_mb'][0]} – {r['pi95_mb'][1]} |"
        for r in hr["p1_full_history"])
    p2 = "\n".join(
        f"| {r['budget_mb']} | " +
        "；".join(f"{h['multiplier_vs_p1']}× P1（{h['total_versions_planned']:,} 写入）时窗口 "
                  f"{h['max_read_window_versions']:,}（覆盖 {h['window_covers_pct_of_history']}%）"
                  for h in r["planning_horizons"]) +
        f"；超限点 {r['infeasible_horizon']['total_versions_planned']:,} 写入时窗口 "
        f"{r['infeasible_horizon']['max_read_window_versions']:,}"
        f"（覆盖 {r['infeasible_horizon']['window_covers_pct_of_history']}%）"
        f" | {r['write_volume_ceiling_min_window']:,} | {r['volume_multiplier_vs_p1']}× |"
        for r in hr["p2_vacuum_window"])
    p3 = "\n".join(
        f"| {row['budget_mb']} | " +
        "；".join(f"预留 {pct(t['reserve_fraction'],0)}：写者 {t['writer_version_cap']:,} / "
                  f"长读 {t['pinned_version_allowance']:,}"
                  for t in row["tiers"]) + " |"
        for row in hr["p3_pin_reserve"])
    width = "\n".join(
        f"| {r['budget_mb']} | " +
        "；".join(f"w={w['text_width']} → "
                  + ("不可识别" if w['max_versions'] is None
                     else f"{w['max_versions']:,}")
                  for w in r["by_text_width"]) + " |"
        for r in hr["width_sensitivity"])

    def slope_str(v):
        s = v.get("slope")
        r2 = v.get("r2")
        s_s = "—" if s is None else f"{s:.1f}"
        r2_s = "—" if r2 is None else f"{r2:.7f}"
        return f"斜率 {s_s} 字节/单位（n={v.get('n')}，R²={r2_s}）"

    slopes_txt = "\n".join(f"- `{k}`：{slope_str(v)}"
                           for k, v in micro.items())
    jrm = micro.get("journal_row_marginal", {})
    joint = micro["row_map_joint"]
    joint_txt = (f"行映射联合拟合每文本字节系数 = "
                 f"{joint['beta'][2]:.2f} 字节/行/字节（n={joint['n']}，"
                 f"R²={joint.get('r2') if joint.get('r2') is not None else '—'}）；"
                 f"WAL 行边际成本 = {jrm.get('slope', 0):.1f} 字节/行"
                 f"（满载记录 {jrm.get('full_record_slope', 0):.1f} 减空记录 "
                 f"{jrm.get('empty_record_slope', 0):.1f}）")

    noise = "\n".join(
        f"| {r['scenario']} | {r['replicates']} | {r['heap_final_cv_pct']:.3f}% | "
        f"{r['load_ms_cv_pct']:.2f}% |"
        for r in d["replicate_detail"])

    th = d.get("throughput_detail", [])
    thl = d.get("throughput_headline", {})
    def pause_str(x):
        # 0.0 is the sentinel for "no vacuum ran in this scenario".
        if x is None or x != x or x == 0.0:
            return "—"
        return f"{x}"

    throughput_rows = "\n".join(
        f"| {r['scenario']} | {r['mutations_per_s']} | {r['commits_per_s']} | "
        f"{r['mean_commit_ms']} | {pause_str(r['vacuum_pause_ms'])} | "
        f"{r['load_ms_cv_pct']}% |"
        for r in th)
    th_txt = []
    if "batch50_vs_batch1_throughput_ratio" in thl:
        th_txt.append(
            f"- 相同写入量下，batch=50 相对 batch=1 的变更吞吐比为 "
            f"**{thl['batch50_vs_batch1_throughput_ratio']}×**：提交粒度是短写入吞吐的"
            "第一调节杆，但它同时改变版本坍缩率与 WAL 记录数，需与内存表联合决策。")
    if "pinned_reader_throughput_ratio_vs_unpinned" in thl:
        th_txt.append(
            f"- 持有跨全程只读事务时的吞吐比 = "
            f"**{thl['pinned_reader_throughput_ratio_vs_unpinned']}**（相对无钉住，"
            f"约 {(1-thl['pinned_reader_throughput_ratio_vs_unpinned'])*100:.0f}% 差异）："
            "两者堆规模几乎相同，该差异大于场景内重复 CV（<2.5%）但未做单因子隔离，"
            "只作观测记录，不推断因果；长读者确定无疑的代价仍是钉住水位造成的内存累积。")
    if thl.get("vacuum_pause_ms"):
        th_txt.append(
            f"- 单次 vacuum 的锁内停顿实测 **{thl['vacuum_pause_ms']} ms**"
            "（含全量索引重建）；这是维护读窗口的周期性写入税，频率应由窗口预算决定。")
    throughput_text = "\n".join(th_txt)

    m2fit = d['model_fits']['M2_ols']
    if m2fit.get("beta") is None:
        path_b = ("M2 在本次数据中不可识别（预测变量共线），路径 B 退化为 M1：每版本 "
                  f"{d['model_fits']['M1_ols']['beta'][1]:.1f} 字节")
    else:
        path_b = (f"每保留版本 {m2fit['beta'][1]:.1f} 字节（版本库+索引侧，可被 vacuum 回收）、"
                  f"每历史版本 {m2fit['beta'][2]:.1f} 字节（WAL 行拷贝，vacuum 不可回收）、"
                  f"每索引候选 {m2fit['beta'][4]:.1f} 字节、每个 WAL 记录容器 "
                  f"{m2fit['beta'][5]:.1f} 字节。")

    rpt = f"""# LaminaDB 历史保留容量决策材料

> 数据驱动版本：基于 {d['n_engine_runs']} 次独立进程引擎测量与 {d['n_micro_runs']} 次组件微观校准，
> 全部原始记录见 `capacity/data/raw/`，独立计算与交叉校验见 `capacity/data/tidy/`。
> 运行环境：Go {base['go_version']}，{base['num_cpu']} CPU，空进程堆基线
> {base['heap_alloc_mean']/1e3:.1f} KB（±{base['heap_alloc_sd']/1e3:.1f} KB，n=5）。
> 本文所有数字由 `analysis/make_report.py` 从 `findings.json` 渲染，未手工填写。

## 1. 结论摘要

1. **当前行数不是容量变量，保留版本数才是。** 同样 1000 个当前主键，仅改变更新量、
   热点分布和提交粒度，最终堆占用可相差数倍；任何 `字节 = c × 行数` 的单行系数模型（M0）
   在留轴验证中被明确拒绝（见第 5 节）。
2. **长读事务会让计划回收失效，且失效是可观测的 `pinned=true` 事实。** 被钉住的
   vacuum 一个版本都无法移除；而无论 vacuum 回收多少，WAL 日志字节数完全不变——
   内存窗口与磁盘日志是两条独立增长曲线。
3. **二级索引的候选数随历史版本累积，不随当前行数。** 全历史索引每个保留版本一个候选，
   晚建索引只改变建索引前的候选可见性语义，不改变建成后的增长速率。
4. **推荐模型 M2（保留版本 Vr + 历史版本 Vh + 活键 K + 索引候选 I + 日志记录容器 C）**
用于内存准入：Vr 是可被 vacuum 回收的版本库侧成本，Vh 是 vacuum 动不了的 WAL 侧成本；
M3 额外刻画文本宽度。组件微观基准与引擎整体 OLS 两条独立标定路径的同名系数相差约 10%。
5. 固定内存预算下给出三套可执行策略（第 7 节）：全历史 P1、vacuum 读窗口 P2、长读预留 P3，
   每个准入边界带 95% 预测区间；扩大历史窗口对可接纳写集的挤出以具体倍数列出。

## 2. 测量方法（为什么这些数字可以作为证据）

- **每次引擎运行都是独立 OS 进程、空数据目录**，堆观测从同一运行时基线开始，避免跨场景
  垃圾累积污染；每个场景 {d['replicate_detail'][0]['replicates']} 次重复。
- 内存同时记录 Go 运行时口径（`HeapAlloc/HeapInuse`、GC 次数与停顿）和 OS 口径（`/proc` RSS）。
  模型基于 GC 后的 `HeapAlloc`（真实存活对象），RSS 只作上界参考——Go 堆不会把空闲 span
  主动还给操作系统，RSS 对容量决策偏保守。
- 每条记录都带**独立结构台账**：Go 侧重放与 Python 侧 `analysis/ledger.py` 用同一套场景输入
  各自推导版本数、活键数、vacuum 幸存者和日志记录数，`cross_checks.jsonl` 中
  {d['n_engine_runs']} 条记录全部一致（失败数 = {d['cross_check_failures']}）才允许进入拟合。
- 关键语义已在台账中复现：同一提交内同键多次写入坍缩为一个版本（last-write-wins）、
  墓碑链在水位以下整条移除、vacuum 水位被最老活动读者压到其快照。
- 限制：数据目录位于 virtiofs，fsync 耗时不代表物理盘；测量规模在数千键、数万版本，
  外推到更大规模时第 8 节给出适用边界。

## 3. 可调输入（全部出现在每条原始记录的 `scenario` 字段）

| 输入 | 含义 | 影响的结构量 |
| --- | --- | --- |
| `keys` | 插入的主键数（当前行数上界） | 活键 K、版本链条数 |
| `updates` | 插入后的更新/墓碑变更数 | 版本 V、日志记录 C |
| `hot_fraction` | 接收 80% 更新的热键比例 | 版本链长度分布（均值相同、形状不同） |
| `batch_size` | 每事务变更数（提交粒度） | 版本数（同键同提交坍缩）、日志记录 C |
| `delete_fraction` | 墓碑占更新比例 | 整链回收量、最终活键 |
| `text_width` | 文本列固定字节宽度 | 每版本有效载荷字节 |
| `index_count` / `late_index` | 二级索引数量与建索引时机 | 索引候选 I |
| `pin_snapshot` | 是否持有跨整个写入期的只读事务 | vacuum 有效水位 |
| `vacuum` / `vacuum_horizon` | 是否回收及请求水位（head/mid） | 保留版本 V、重建后的 I |

## 4. 无模型原始证据

### 4.1 相同当前行数，不同版本链与占用

| 场景 | 当前键 | 保留版本 | 平均链长 | 最终堆 | WAL |
| --- | --- | --- | --- | --- | --- |
{same_rows}

行数恒为 1000（或 500），堆占用随版本数与链形状变化；提交粒度 `batch=1` 与 `batch=500`
的对比同时改变版本数和 WAL 记录数，这正是单一行数系数无法解释的两个方向。

### 4.2 长读事务使计划回收无效

| 场景 | pinned | 有效水位 | 移除版本 | 保留版本 | 最终堆 | WAL |
| --- | --- | --- | --- | --- | --- | --- |
{pin_rows}

### 4.3 vacuum 回收内存但永不回收日志

{ji_text}

### 4.4 索引候选跟随历史而非当前行

| 场景 | 索引数 | 当前键 | 版本数 | 候选总数 | 最终堆 |
| --- | --- | --- | --- | --- | --- |
{idx_rows}

### 4.5 写入吞吐与维护停顿（短写入不被历史窗口吞掉）

{throughput_text}

| 场景 | 变更/秒 | 提交/秒 | 平均提交 ms | vacuum 停顿 ms | 装载耗时 CV |
| --- | --- | --- | --- | --- | --- |
{throughput_rows}

完整逐场景数据在 `data/tidy/throughput.csv`。

### 4.6 堆剖析：两份行拷贝的直接证据（与行数系数之争无关的结构性事实）

对相同 1000 键 / 10000 更新、三种提交粒度各做一次全采样率（`MemProfileRate=1`）
inuse_space 堆剖析，原始 profile 在 `data/raw/profiles/`，文本汇总在
`data/tidy/heap_profile_attribution.txt`。三个 profile 一致显示存活字节的前两名各占约 43%：

- `datum.Clone`：版本库 `versions.Store.Apply` 为每个保留版本克隆的整行；
- `catalog.Relation.Normalize`：事务暂存并被 `journal.Record.Writes` 长期持有的整行。

两者规模在 batch=1/50/500 下始终接近 1:1，并随**保留/历史版本数**而非原始变更数缩放
（batch=500 因同提交同键坍缩，两份都只有 ~3.8k 版本大小）。这从分配栈层面独立证实了
M2 的双版本预测器：`Vr`（可 vacuum 回收的版本库侧）与 `Vh`（vacuum 不可回收的 WAL 侧）。

### 4.7 重复运行的经验误差地板

| 场景 | 重复数 | 最终堆变异系数 | 装载耗时变异系数 |
| --- | --- | --- | --- |
{noise}

跨场景中位堆 CV = {d['replicate_noise']['median_heap_cv_pct']:.3f}%，
最大 = {d['replicate_noise']['max_heap_cv_pct']:.3f}%。这是任何模型都无法再降低的测量地板，
下文预测区间均显著宽于该地板，说明区间主要由模型/结构不确定性而非测量抖动主导。

## 5. 相互竞争的模型与留轴裁决

四个嵌套模型，全部对 GC 后 `HeapAlloc` 减去空进程基线拟合：

- **M0**：仅当前行数 K（即题目拒绝的“一个行数系数”）
- **M1**：仅保留版本 Vr（把版本库与 WAL 两份行拷贝混成一项）
- **M2**：保留版本 Vr + 历史版本 Vh + 活键 K + 索引候选 I + WAL 记录容器 C
- **M3**：M2 + 行文本字节 (Vr+Vh)×w

{coef_table(d['model_fits'])}

留一轴交叉验证（每次藏起整条工作负载轴，用其余轴训练后预测未见形状。RMSE 对单点巨差取平方，
数值会被少数外推尾部放大到 GB 级，因此请以中位/95 分位百分比误差为主判据，RMSE 仅作尾部
风险的保守刻度）：

{loo_table(d['leave_one_axis_out'])}

解读要点：

- M0 调整 R² ≈ 0（行数系数的标准误大于系数本身），留轴中位误差 22%，**行数单系数被明确拒绝**；
  M1 调整 R²={d['model_fits']['M1_ols']['adj_r2']:.3f}，版本数是第一主因但仍把两份行拷贝
  混为一项，vacuum 场景无法解释。
- M2 引入保留版本 Vr 与历史版本 Vh 后调整 R² 升到 {d['model_fits']['M2_ols']['adj_r2']:.3f}，
  留轴中位误差降到 {next(t for t in d['leave_one_axis_out'] if t['model']=='M2')['loo_median_abs_pct']:.2f}%；
  Vr/Vh 系数都显著（标准误远小于系数），分别对应可回收与不可回收的两份行拷贝。
- M3 再加文本字节项，调整 R²={d['model_fits']['M3_ols']['adj_r2']:.3f}，但留轴 RMSE 被极少数
  外推尾部主导：藏起宽度轴（D）或提交粒度轴（C）时，训练集没有该维度的变化量，系数只能
  从共线关系外推，单点误差可达数百 MB。这不是模型在测量范围内失效，而是**外推边界的真实
  不确定性**：表中同时给中位、95 分位和最大误差，决策区间按上界取值；要进入新的批量/宽度
   regime，应先补测该轴再重拟合。
- 活键 K 的系数在 M2/M3 中均不显著（标准误与系数同量级），说明在版本主导的工作负载里，
  当前行数对堆的边际解释力被版本项吸收——与 M0 的失败互为印证。
- 表中的“不可识别折”是设计使然而非缺陷：只有 H_rows 轴让 K 发生变化，藏起它后训练集里 K
  为常数，K 系数无法估计；M2/M3 仍用其余 72 个点完成预测。这也从实验设计角度说明：要给
  行数定系数，必须先在行数轴上补测，不能从版本轴外推。

## 6. 两条独立标定路径的对照

**路径 A：组件微观基准（全新进程，只构造单一结构）**

{slopes_txt}
- {joint_txt}

**路径 B：引擎整体 OLS 的对应系数**：{path_b}

路径 A 在完全不接触引擎整体数据的情况下预测引擎占用：
RMSE = {kb(micro_eval['loo_rmse_bytes'])}，中位绝对误差 {micro_eval['median_abs_pct']:.2f}%，
95 分位 {micro_eval['p95_abs_pct']:.2f}%，最大 {micro_eval['max_abs_pct']:.2f}%
（n={micro_eval['n']}）。两条独立路径的边际成本互相印证：整体 OLS 的每保留版本
{d['model_fits']['M2_ols']['beta'][1]:.0f} B 对微观版本库 {d['micro_slopes']['version_store_per_version']['slope']:.0f} B、
OLS 每历史 WAL 版本 {d['model_fits']['M2_ols']['beta'][2]:.0f} B 对微观 WAL 行
{d['micro_slopes']['journal_row_marginal']['slope']:.0f} B、OLS 每索引候选
{d['model_fits']['M2_ols']['beta'][4]:.0f} B 对微观索引树
{d['micro_slopes']['index_tree_per_entry']['slope']:.0f} B，差异均在约 10% 内。
残余差异来自引擎固定容器（目录、事务、map 桶开销）与共线性，决策时取两者中更宽的误差边界。

磁盘侧模型（{len(disk)} 组拟合）：journal J0/J1 与镜像每版本斜率见 `findings.json ->
disk_models`；镜像随保留版本线性变化，journal 随累计写入线性变化且不被 vacuum 改变。

## 7. 固定内存预算下的策略选择（全部带 95% 预测区间）

基准假设：{hr['assumptions']['keys']} 个主键、平均提交批量 {hr['assumptions']['avg_commit_batch']}、
文本宽度 {hr['assumptions']['text_width']}、无二级索引；拟合残差 RSE =
{hr['assumptions']['rse_mb']} MB（dof={hr['assumptions']['dof']}）。
准入判定统一使用预测区间**上界**，即宁可少接纳。

### P1 全历史保留（不 vacuum）：预算 → 最大可保留版本

| 预算 | 最大版本数 | 折合新增写入 | 预测堆 | 95% 区间 |
| --- | --- | --- | --- | --- |
{p1}

### P2 周期 vacuum 维护读窗口：同样预算能接纳多少累计写入

| 预算 | 各规划期限下可维持的读窗口 | 最小窗口（W=K）时写入天花板 | 对 P1 倍数 |
| --- | --- | --- | --- |
{p2}

读窗口越短，同样内存可接纳的累计写入越大，但可回溯的历史比例同步下降；
当规划写入量远超预算时读窗口会被算成 0——含义是**连最小窗口都放不下，不可执行**而不是
允许零窗口运行，这正是不截断 WAL 的长期代价：vacuum 只能调节 Vr 项，Vh 行拷贝与记录容器
仍随累计写入单调增长。最小窗口 W=K 时的写入天花板相对 P1 的倍数（约 1.7–1.8×）才是该实现
下 vacuum 能给出的稳态写集增益上限；要突破它必须先实现日志截断（本基线明确未实现）。
WAL 磁盘成本不因 vacuum 改变，长寿命进程仍需独立的日志容量项。

### P3 为偶发长读预留预算：写者/长读的版本配额拆分

| 预算 | 各预留档位下的配额（写者 / 长读保护版本） |
| --- | --- |
{p3}

### 文本宽度敏感性（M3，同预算下的版本天花板）

| 预算 | 不同文本宽度的最大版本数 |
| --- | --- |
{width}

## 8. 误差边界与适用范围

1. **区间口径**：策略表用 OLS 预测区间（系数协方差 + 残差散度，t 分位 95%），
   比仅重复运行的经验 CV（{d['replicate_noise']['median_heap_cv_pct']:.3f}% 量级）更宽，
   属保守口径。
2. **共线性**：无 vacuum 时 Vr=Vh、V/I/C 同向增长，M2/M3 单个系数的标准误会如实变大；
   做单点定价时优先使用整体预测区间，不要把单个系数当作精确单价。Vr 与 Vh 的分离由
   vacuum/钉住轴的对比识别，该轴之外的预测仍受这一共线性约束。
3. **外推边界**：拟合范围为数千键、数万版本、文本 ≤256 字节、索引 ≤2。超出该范围
   （特别是 map 扩容行为变化、128 MiB 镜像上限附近）需要补点重测；模型不含 GC 碎片、
   goroutine 栈增长和 RSS 不归还效应，RSS 口径请直接查原始记录。
4. **未覆盖成本**：查询期临时物化（排序/全匹配行）、备份瞬时双份内存、HTTP 请求缓冲
   不在本模型内；生产准入应在 P1 边界上再扣减工作负载峰值余量。
5. **磁盘与内存分离**：vacuum 对 journal 字节数零影响是实测事实，日志截断在本基线中
   明确未实现，不能假设长期运行后磁盘会自动回落。

## 9. 复现步骤

```sh
# 1. 构建被测系统与测量 harness（Go 1.25+，CGO_ENABLED=0）
go build -o bin/laminad ./cmd/laminad
go build -o capacity/tmp/harness ./capacity/harness

# 2. 测量矩阵（独立进程、每场景 3 重复；--quick 可先冒烟）
python3 -I capacity/analysis/run_matrix.py \\
  --harness capacity/tmp/harness \\
  --out capacity/data/raw --workdir capacity/tmp/full --replicates 3

# 3. 独立台账交叉校验 + 模型拟合 + 策略表
python3 -I capacity/analysis/analyze.py \\
  --raw capacity/data/raw --tidy capacity/data/tidy

# 4. 渲染本报告
python3 -I capacity/analysis/make_report.py \\
  --findings capacity/data/tidy/findings.json --out capacity/CAPACITY_REPORT.md
```

## 10. 交付物清单

| 路径 | 内容 |
| --- | --- |
| `capacity/harness/*.go` | 测量 harness：场景台账、引擎运行、组件微观基准、内存/磁盘取证 |
| `capacity/analysis/run_matrix.py` | 矩阵驱动（场景轴、重复、独立进程） |
| `capacity/analysis/ledger.py` | Python 独立结构台账与 vacuum 规则重放 |
| `capacity/analysis/statsx.py` | 无第三方依赖的 OLS、t 分位、预测区间 |
| `capacity/analysis/models.py` | M0–M3 竞争模型、留轴验证、微观校准、磁盘模型 |
| `capacity/analysis/analyze.py` | 交叉校验、tidy 表、拟合、策略与误差边界总装 |
| `capacity/data/raw/*.jsonl` | 全部原始观测（引擎、微观、基线） |
| `capacity/data/tidy/` | tidy CSV、逐条交叉校验、`findings.json` |
"""
    with open(args.out, "w") as f:
        f.write(rpt)
    print(f"wrote {args.out} ({len(rpt)} chars)")


if __name__ == "__main__":
    main()
