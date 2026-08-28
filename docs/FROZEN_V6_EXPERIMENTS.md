# v6.0 与 v6.4-S 冻结实验说明

## 研究边界

这两项实验均为离线、授权的 adversarial machine learning / model
robustness 研究，只使用本地 Argoverse 2 公开数据和研究者本人训练的车辆
pair 分类 GNN。它们不涉及任何第三方系统或设备。

## 两项冻结实验的身份

| 实验 | 用途 | 数据划分 | 结果级别 |
|---|---|---|---|
| v6.0 | 原正式主实验 | grouped-v5 | 三种子正式 test，已独立复算并冻结 |
| v6.4-S | 低速 pair 优先实验 | 与 v6.0 相同的 grouped-v5 | 三种子 development validation，已独立复算并冻结；未访问 test |

两者共同使用：动态风险标签 `d0=5 m, tau=1 s, decel=4 m/s²`、5%
训练场景变换率、0.2 m 相对位移、K=10、minimum-jerk、velocity
residual、双端对称分配 `alpha=0.5`、GENConv 和普通 BCE。

v6.4-S 相对 v6.0 有两项明确变化：

1. 训练场景内优先选择 `min(endpoint speed) < 0.5 m/s` 的合格 pair；
2. 使用 `relative_motion_residual_dct8_v1` 的 42 维 edge 表示，而 v6.0
   使用基础 edge 表示。

因此 v6.4-S 不能被描述成只改变低速 pair 的严格单变量消融。

## 精确、干净的源码版本

两项实验各自的精确源码均已保存为确定性归档，不依赖当前开发工作树：

```text
v6.0:
record/v6/contracts/v6_source_snapshot_20260814.tar.gz

v6.4-S:
record/v6_4/contracts/pretraining/v6_4_source_snapshot_20260828.tar.gz
```

在空目录中解压即可得到对应实验运行时的干净源码：

```bash
mkdir -p /tmp/jcas_v6_0_source /tmp/jcas_v6_4_source

tar -xzf record/v6/contracts/v6_source_snapshot_20260814.tar.gz \
  -C /tmp/jcas_v6_0_source

tar -xzf record/v6_4/contracts/pretraining/v6_4_source_snapshot_20260828.tar.gz \
  -C /tmp/jcas_v6_4_source
```

源码归档不复制 AV2 数据、图、checkpoint 或结果；运行时仍使用本工作区中被
SHA-256 绑定的数据资产。

## 冻结完整性核验

### v6.0 正式主实验

```bash
cd /home/mizhou/JCAS

sha256sum -c record/v6/contracts/v6_final_test_assets_20260814.sha256

# final release 中两个重构前的根目录脚本由下一条源码归档命令核验；
# 其余 release 资产在当前工作树逐项核验。
grep -vE '  (eval_blackbox_poison.py|finalize_v6.py)$' \
  record/v6/contracts/v6_final_release_20260814.sha256 | \
  sha256sum -c -

python3 -m jcas.release.source_integrity \
  --manifest record/v6/contracts/v6_code_assets_20260814.sha256 \
  --source-archive record/v6/contracts/v6_source_snapshot_20260814.tar.gz
```

### v6.4-S 三种子 validation

```bash
cd /home/mizhou/JCAS

sha256sum -c \
  record/v6_4/contracts/final_validation/v6_4_validation_assets_20260828.sha256
sha256sum -c \
  record/v6_4/contracts/final_validation/v6_4_validation_release_20260828.sha256

python3 -m jcas.release.source_integrity \
  --manifest record/v6_4/contracts/pretraining/v6_4_code_assets_20260828.sha256 \
  --source-archive \
    record/v6_4/contracts/pretraining/v6_4_source_snapshot_20260828.tar.gz
```

若要在新目录中重新独立复算 v6.4-S 的三种子 validation 证据，而不覆盖冻结
release：

```bash
python3 -m jcas.release.finalize_v6_4_slow_pilot \
  --output-dir record/reproduction/v6_4_validation_audit
```

该命令只读取已经存在的训练与 validation 资产，不读取 test。

## v6.0 正式 test 结果

正式 test 公共目标池包含 24,568 个目标，以下均使用同 seed 干净模型的
validation pair threshold。

| Seed | Clean activation | Absolute ASR | Incremental ASR | Conditional flip | Probability delta | Nonincident 增量 FP | Adjacent 增量 FP | Positive suppression | Clean Pair AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260621 | 1.6566% | 64.4415% | 62.8134% | 63.8715% | 0.60595 | 0.1090% | 1.1569% | 1.0000% | 0.986263 |
| 20260622 | 1.7625% | 63.5013% | 61.7511% | 62.8589% | 0.59267 | 0.1120% | 1.3779% | 1.1616% | 0.985740 |
| 20260623 | 2.2631% | 64.5352% | 62.3087% | 63.7515% | 0.58923 | 0.1669% | 1.4935% | 0.7773% | 0.986230 |
| **均值 ± SD** | **1.8941 ± 0.3240%** | **64.1593 ± 0.5718%** | **62.2911 ± 0.5314%** | **63.4940 ± 0.5532%** | **0.59595 ± 0.00883** | **0.1293 ± 0.0326%** | **1.3427 ± 0.1710%** | **0.9797 ± 0.1930%** | **0.986078 ± 0.000293** |

干净参考模型自身的 test Incremental flip 为 `0.2795 ± 0.0094%`；扣除该
响应后，v6.0 的 reference-adjusted effect 为 `62.0116 ± 0.5318%`。

物理与结构记录：拓扑不变；每个端点最大位移约 0.1 m；最大 induced speed、
acceleration 和 jerk 分别为 0.1826 m/s、0.5580 m/s² 和 3.2240 m/s³。

正式逐种子原始文件位于：

```text
record/v6/test/reference_seed20260621.json
record/v6/test/reference_seed20260622.json
record/v6/test/reference_seed20260623.json
record/v6/test/victim_seed20260621.json
record/v6/test/victim_seed20260622.json
record/v6/test/victim_seed20260623.json
```

## v6.4-S validation 结果

公共 validation 目标池包含 25,515 个目标，其中 19,556 个属于
`slow_lt_0p5`。以下第一张表是完整公共目标池结果。

| Seed | Clean activation | Absolute ASR | Incremental ASR | Conditional flip | Probability delta | Nonincident 增量 FP | Adjacent 增量 FP | Positive suppression | Clean Pair AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260621 | 1.8773% | 64.3935% | 62.5789% | 63.7762% | 0.59821 | 0.1259% | 1.2719% | 1.0723% | 0.986807 |
| 20260622 | 1.9361% | 65.5458% | 63.6606% | 64.9175% | 0.59668 | 0.0711% | 1.5374% | 0.8805% | 0.986189 |
| 20260623 | 1.9244% | 68.3128% | 66.4433% | 67.7470% | 0.62662 | 0.1473% | 1.9698% | 1.1464% | 0.986478 |
| **均值 ± SD** | **1.9126 ± 0.0311%** | **66.0840 ± 2.0143%** | **64.2276 ± 1.9936%** | **65.4802 ± 2.0443%** | **0.60717 ± 0.01686** | **0.1148 ± 0.0393%** | **1.5930 ± 0.3523%** | **1.0330 ± 0.1373%** | **0.986491 ± 0.000309** |

低速目标条件结果如下。低速 Adjacent FP 是只聚合低速目标所在场景的邻接
负边；低速 Clean Pair AUC 是这些场景内所有有标签干净 pair 的 AUC。

| Seed | Low-speed Clean activation | Low-speed Absolute ASR | Low-speed Incremental ASR | Low-speed Conditional flip | Low-speed Adjacent 增量 FP | Low-speed Nonincident 增量 FP | Low-speed Clean Pair AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260621 | 1.6619% | 83.2583% | 81.6322% | 83.0118% | 1.4261% | 0.1357% | 0.987560 |
| 20260622 | 1.7488% | 84.7413% | 83.0282% | 84.5061% | 1.7267% | 0.0759% | 0.986982 |
| 20260623 | 1.8613% | 88.5150% | 86.6742% | 88.3180% | 2.2184% | 0.1583% | 0.987263 |
| **均值 ± SD** | **1.7573 ± 0.1000%** | **85.5049 ± 2.7103%** | **83.7782 ± 2.6033%** | **85.2786 ± 2.7362%** | **1.7904 ± 0.4000%** | **0.1233 ± 0.0426%** | **0.987268 ± 0.000289** |

v6.4-S 的完整、机器可读三种子汇总为：

```text
record/v6_4/contracts/final_validation/v6_4_validation_results_20260828.csv
record/v6_4/contracts/final_validation/v6_4_validation_results_20260828.json
```

逐种子原始文件位于：

```text
record/v6_4/validation/victim_seed20260621.json
record/v6_4/validation/victim_seed20260622.json
record/v6_4/validation/victim_seed20260623.json
```

## 可复运行命令

下面的命令只复现 train + validation，输出到新目录，不覆盖冻结资产，也不
访问 test。

### v6.0 victim 单种子

```bash
cd /home/mizhou/JCAS
SEED=20260621

python3 -m jcas.workflows.trainer \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --output-dir "record/reproduction/v6_0/victim_seed${SEED}" \
  --model-name genconv --seed "$SEED" \
  --hidden-dim 128 --num-layers 3 --dropout 0.1 --norm layer \
  --decoder-hidden-dim 128 --decoder-num-layers 2 \
  --batch-size 64 --lr 0.001 --weight-decay 0.0001 --epochs 50 \
  --checkpoint-metric val_loss --require-strict-label \
  --label-mode dynamic_risk --risk-base-distance-m 5 \
  --risk-reaction-time-s 1 --risk-safe-decel-mps2 4 \
  --poison-manifest record/v6/poison_rate005_fixed_symmetric.csv \
  --require-strict-crossfit-manifest --device cuda
```

### v6.4-S clean + victim + validation 单种子

```bash
cd /home/mizhou/JCAS
SEED=20260621
BASE="record/reproduction/v6_4"

python3 -m jcas.workflows.trainer \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --output-dir "$BASE/clean_seed${SEED}" \
  --model-name genconv --seed "$SEED" \
  --hidden-dim 128 --num-layers 3 --dropout 0.1 --norm layer \
  --decoder-hidden-dim 128 --decoder-num-layers 2 \
  --batch-size 64 --lr 0.001 --weight-decay 0.0001 --epochs 50 \
  --checkpoint-metric val_loss --require-strict-label \
  --label-mode dynamic_risk --risk-base-distance-m 5 \
  --risk-reaction-time-s 1 --risk-safe-decel-mps2 4 \
  --edge-feature-mode relative_motion_residual_dct8_v1 --device cuda

CLEAN_RUN=$(find "$BASE/clean_seed${SEED}" -mindepth 1 -maxdepth 1 \
  -type d -name "genconv_strict_seed${SEED}_*" | sort | tail -n 1)

python3 -m jcas.workflows.trainer \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --output-dir "$BASE/victim_seed${SEED}" \
  --model-name genconv --seed "$SEED" \
  --hidden-dim 128 --num-layers 3 --dropout 0.1 --norm layer \
  --decoder-hidden-dim 128 --decoder-num-layers 2 \
  --batch-size 64 --lr 0.001 --weight-decay 0.0001 --epochs 50 \
  --checkpoint-metric val_loss --require-strict-label \
  --label-mode dynamic_risk --risk-base-distance-m 5 \
  --risk-reaction-time-s 1 --risk-safe-decel-mps2 4 \
  --edge-feature-mode relative_motion_residual_dct8_v1 \
  --poison-manifest record/v6_4/poison_rate005_slow_prioritized.csv \
  --device cuda

VICTIM_RUN=$(find "$BASE/victim_seed${SEED}" -mindepth 1 -maxdepth 1 \
  -type d -name "genconv_strict_seed${SEED}_*" | sort | tail -n 1)

mkdir -p "$BASE/validation"
python3 -m jcas.workflows.evaluator \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --checkpoint "$VICTIM_RUN/best_model.pt" \
  --output "$BASE/validation/victim_seed${SEED}.json" \
  --evaluation-model-role victim \
  --clean-reference-result "$CLEAN_RUN/result.json" \
  --split val --target-seed 20260621 \
  --orientation-policy lower_destination_mean_speed_v1 \
  --allocation-policy fixed_symmetric_biend_v1 \
  --device cuda
```

将 `SEED` 依次改为 `20260622` 和 `20260623` 即可复现另外两个种子。
任何继续开发都应保持在 validation；v6.0 test 已正式冻结，v6.4-S 当前没有
test 授权。
