# v6.0 运行与核验说明

## 研究边界

本流程仅处理本地 Argoverse 2 数据和研究者自训练 GNN，用于车辆 pair 分类的离线模型鲁棒性研究。

## 冻结配置

```text
label                  = dynamic_risk(5 m, 1 s, 4 m/s²)
poison scenario rate   = 5%
relative displacement  = 0.2 m
perturb window         = 10 intervals
ramp                   = minimum_jerk
velocity               = residual
allocation             = fixed_symmetric_biend_v1
allocation alpha       = 0.5
epochs                 = 50
checkpoint metric      = val_loss
```

训练清单：

```text
record/v6/poison_rate005_fixed_symmetric.csv
record/v6/poison_rate005_fixed_symmetric.csv.metadata.json
record/v6/poison_rate005_fixed_symmetric.fixed_alpha_audit.csv
```

它包含 5,776 个训练场景。所有场景均使用 `alpha=0.5`；其中 2,110 行超出旧 pair-feature budget，这一点属于已披露的方法差异。

## 训练复现

不要覆盖现有正式目录，不要添加 `--evaluate-test`：

```bash
python3 -m jcas.workflows.trainer \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --output-dir record/reproduction/v6_seed20260621 \
  --model-name genconv \
  --seed 20260621 \
  --hidden-dim 128 \
  --num-layers 3 \
  --dropout 0.1 \
  --norm layer \
  --decoder-hidden-dim 128 \
  --decoder-num-layers 2 \
  --batch-size 64 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --epochs 50 \
  --checkpoint-metric val_loss \
  --require-strict-label \
  --label-mode dynamic_risk \
  --risk-base-distance-m 5 \
  --risk-reaction-time-s 1 \
  --risk-safe-decel-mps2 4 \
  --poison-manifest record/v6/poison_rate005_fixed_symmetric.csv \
  --require-strict-crossfit-manifest \
  --device cuda
```

## Validation 复核

下面仅展示 victim 侧命令。输出必须写入新目录：

```bash
CLEAN_RUN=record/v5/clean_seed20260621/genconv_strict_seed20260621_20260812_123153
V6_RUN=$(find record/reproduction/v6_seed20260621 -mindepth 1 -maxdepth 1 \
  -type d -name 'genconv_strict_seed20260621_*' | sort | tail -n 1)

python3 -m jcas.workflows.evaluator \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --checkpoint "$V6_RUN/best_model.pt" \
  --output record/reproduction/validation/victim_seed20260621.json \
  --evaluation-model-role victim \
  --clean-reference-result "$CLEAN_RUN/result.json" \
  --split val \
  --target-seed 20260621 \
  --orientation-policy lower_destination_mean_speed_v1 \
  --allocation-policy fixed_symmetric_biend_v1 \
  --device cuda
```

正式 test 已经冻结并发布，不应为了继续调参而覆盖或反复运行。

## 正式结果核验

```bash
sha256sum -c record/v6/contracts/v6_final_test_assets_20260814.sha256
python3 -m jcas.release.source_integrity \
  --manifest record/v6/contracts/v6_final_release_20260814.sha256 \
  --source-archive record/v6/contracts/v6_source_snapshot_20260814.tar.gz
```

正式源码的精确副本为：

```text
record/v6/contracts/v6_source_snapshot_20260814.tar.gz
```

当前冻结的正式主实验仍为 v6.0；原结果仍绑定原始冻结源码，不需要重新训练
或重跑 test。v6.4-S 低速 pair 优先实验已经完成三种子 validation 并作为
development-only release 冻结。两项实验的独立结果、源码快照和可复运行命令
统一见 `docs/FROZEN_V6_EXPERIMENTS.md`。

## v6.4-S 低速 pair 优先 pilot

该开发实验保持 v6.1 的 42 维表示、GENConv、普通 BCE、5% 场景、
`0.2 m / K10 / minimum-jerk / velocity residual / alpha=0.5` 不变。
它只在冻结的 5,776 个训练场景内优先选择
`min(endpoint_speed) < 0.5 m/s` 的合格 pair；不存在合格低速 pair 时使用
稳定哈希回退到原候选集合。选择过程只读取 train，不读取模型输出、validation
或 test。

首个 seed 固定为 `20260621`，其五项冻结继续门槛全部通过后，另外两个 seed
也已完成。三种子证据已独立复算并冻结在
`record/v6_4/contracts/final_validation/`。该实验仍不授权 test。
