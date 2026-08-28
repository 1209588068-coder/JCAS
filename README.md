# JCAS v6.0

本仓库仅用于本地 Argoverse 2 数据和研究者自训练图神经网络的离线模型鲁棒性研究，不涉及第三方系统。

## 当前主实验

v6.0 是当前唯一主线：

- 任务：车辆对无向 pair 风险分类；
- 标签：动态风险距离，参数为 `d0=5 m`、`tau=1 s`、`decel=4 m/s²`；
- 图：`graph-v4` 字节资产与 `grouped-v5` 防泄漏划分；
- 训练场景变换比例：5%；
- 目标 pair：继承冻结的严格交叉拟合选择结果；
- 轨迹变化：0.2 m、10 个间隔、minimum-jerk、velocity residual；
- 双端分配：固定对称，`alpha=0.5`；
- 三个训练种子：20260621、20260622、20260623。

正式 test 的三种子均值：

| 指标 | 均值 |
|---|---:|
| Incremental ASR | 62.29% |
| Conditional flip | 63.49% |
| Absolute ASR | 64.16% |
| Clean activation | 1.89% |
| Pair AUC | 0.98608 |
| Nonincident incremental FP | 0.129% |
| Adjacent incremental FP | 1.343% |

完整结果见 `record/v6/contracts/v6_final_test_20260814.metadata.json`。

## 代码结构

```text
jcas/
├── core/       # 模型、标签、划分、轨迹变化和清单校验
├── workflows/  # 构图、训练、评估及清单生成入口
└── release/    # 冻结、独立复算和发布完整性
tests/          # v6 核心回归测试
docs/           # v6 运行说明
graphs/         # v6 使用的冻结图和 split manifest
record/v6/      # v6 模型、评估证据和正式 contract
```

`record/v5` 与 `record/v5_1_1` 中保留的少量文件是 v6 冻结 contract 明确绑定的 clean reference、目标选择来源和 validation 选择证据，并非继续维护的旧实验。

## 运行入口

无需安装即可从仓库根目录运行：

```bash
python3 -m jcas.workflows.graph_builder --help
python3 -m jcas.workflows.trainer --help
python3 -m jcas.workflows.evaluator --help
python3 -m jcas.workflows.fixed_symmetric_manifest --help
```

也可以执行 `python3 -m pip install -e .`，随后使用 `jcas-train`、`jcas-evaluate` 等命令。

详细命令见 `docs/V6_RUNBOOK.md`。v6.0 与低速 pair 优先 v6.4-S 的独立
冻结结果、精确源码快照和复运行命令统一见
`docs/FROZEN_V6_EXPERIMENTS.md`。

## 冻结与代码整理

正式 v6.0 结果所使用的精确源码保存在：

```text
record/v6/contracts/v6_source_snapshot_20260814.tar.gz
```

冻结的正式主实验仍为 v6.0。当前工作树为 `v6.4.0.dev0`；低速 pair
优先 v6.4-S 已完成三种子 validation，并冻结为 development-only release。
它没有访问 test，不具备正式 test 结果身份，也不会覆盖 v6.0。
