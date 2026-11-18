# LaminaDB 历史保留容量决策包

本目录是题目要求的容量决策材料：用**真实运行**校准相互竞争的模型，交付原始数据、
独立计算和误差边界，而不是一张没有证据的容量图。所有脚本仅依赖 Go 标准库与
Python 3 标准库（无 numpy/pandas）。

## 目录结构

```
capacity/
├── run_all.sh                # 一键复现（构建 → 测量 → 独立校验/拟合 → 报告 → 计算器冒烟）
├── CAPACITY_REPORT.md        # 最终决策报告（由 findings.json 渲染，勿手改）
├── harness/                  # Go 测量程序，直接调用 internal/engine 真实内核
│   ├── main.go               #   engine / micro / baseline 三个子命令
│   ├── workload.go           #   可调场景、确定性负载、Go 侧结构台账
│   ├── run.go                #   引擎运行、阶段内存、vacuum/钉住/索引、磁盘取证、交叉校验
│   ├── micro.go              #   组件级微观基准（行映射/版本库/索引树/日志记录）
│   └── mem.go                #   runtime.MemStats + /proc RSS + GC 控制
├── analysis/
│   ├── run_matrix.py         # 矩阵驱动：每场景独立 OS 进程、空目录、3 重复
│   ├── ledger.py             # Python 独立结构台账（重放负载 + 重写 vacuum 规则）
│   ├── statsx.py             # 手写 OLS / 矩阵求逆 / t 分位 / 预测区间
│   ├── models.py             # M0–M3 竞争模型、留轴验证、微观校准、磁盘模型
│   ├── analyze.py            # 交叉校验 → tidy → 拟合 → 策略表 → findings.json
│   ├── make_report.py        # findings.json → CAPACITY_REPORT.md
│   └── calculator.py         # 可调输入 → 带 95% 区间的占用预测与预算准入判定
├── data/
│   ├── raw/                  # 原始 JSONL（engine_runs / microbench / baseline）+ profiles/ 堆剖析
│   └── tidy/                 # tidy CSV、逐条 cross_checks、findings.json、剖析汇总
└── tmp/                      # 构建产物与运行期目录（git 忽略，可随时删除）
```

## 证据链（每条数字都可追溯）

1. **原始观测** `data/raw/engine_runs.jsonl`：每次运行一个 JSON 对象，含完整场景输入、
   引擎 `stats`、分阶段内存（GC 后）、WAL/镜像/备份字节、vacuum 事实、耗时和交叉校验位。
2. **独立台账** `ledger.py`：不读取 harness 的任何计数，只从场景输入重放确定性负载，
   独立推出版本数（含同提交同键坍缩）、活键、vacuum 幸存者和日志记录数，再与引擎
   `stats` 逐条比对。任何一条不一致，`analyze.py` 拒绝拟合并退出非零。
3. **竞争模型**：M0 行数单系数（基线反例）、M1 单版本项、M2 双版本结构项
   （Vr 可回收 + Vh 不可回收，加 K/I/C）、M3 再加文本字节。裁决用**留一轴交叉验证**
   （藏起整条负载轴预测未见形状），不是样本内 R²。
4. **第二条独立标定路径**：组件微观基准在全新进程里只构造单一结构，得到每元素边际成本，
   再在零拟合的情况下预测引擎整体占用；与整体 OLS 系数对照，差异计入不确定性。
5. **误差边界**：重复运行给出测量地板（经验 CV）；OLS 预测区间（系数协方差 + 残差散度，
   t 分位 95%）给出决策口径；策略准入统一取区间上界。

## 三条核心容量事实（详见报告第 4 节）

- 相同当前行数可以对应数量级不同的版本链：版本数、链形状、提交粒度共同决定占用。
- 长读事务把 vacuum 有效水位钉在自己的快照（`pinned=true`、`removed=0` 是实测输出）。
- vacuum 收缩内存版本与索引，但 WAL 帧只增不删：内存窗口与磁盘日志是两条独立曲线。

## 模型口径（M0–M3，留轴交叉验证裁决）

- **Vr 保留版本**：版本库 + 重建后索引，vacuum 可回收；**Vh 历史版本**：WAL 长期持有的
  行拷贝，vacuum 不可回收。无 vacuum 时 Vr=Vh，vacuum/钉住轴的对比是分离两者的唯一证据。
- 堆剖析（全采样率 pprof，`data/raw/profiles/`）独立证实每个存活版本对应两份约等大的整行
  （`datum.Clone` 与 `catalog.Relation.Normalize`，各占存活字节约 43%）。
- 微观基准与整体 OLS 两条独立路径的同名边际成本相差约 10%，报告同时给出两者与 95% 预测区间。

## 快速使用

```sh
# 完整复现（约 15–20 分钟）；若 go 不在 PATH，用 GO_BIN=/path/to/go ./run_all.sh
./run_all.sh
# 冒烟（约 1–2 分钟，样本不足，只验证管线不验证结论）
QUICK=1 REPLICATES=2 ./run_all.sh

# 单点预算准入（假设 1000 键、12000 保留版本、260 条日志、128 MiB 预算）
python3 -I analysis/calculator.py --findings data/tidy/findings.json \
    --keys 1000 --versions 12000 --journal-records 260 --budget-mb 128
```

## 已知边界

- 测量在 virtiofs 上进行，fsync 与磁盘延迟不代表物理盘；内存口径不受影响。
- 拟合范围：数千键、数万版本、文本 ≤256 字节、索引 ≤2；外推需补点重测。
- 模型不含查询期临时物化、备份瞬时双份内存与 RSS 不归还效应（RSS 在原始记录中可查）。
- 本基线未实现日志自动截断，长期运行的 journal 增长没有自动回落机制。
