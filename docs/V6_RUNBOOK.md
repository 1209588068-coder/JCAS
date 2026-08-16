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

当前 `jcas/` 包是 v6.0.1 代码目录整理版，不改变既有实验结果。
