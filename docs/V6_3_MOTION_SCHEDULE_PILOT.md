# v6.3：运动状态分段时间窗 pilot

本实验仅用于本地 Argoverse 2 与自训练 GNN 的离线模型鲁棒性研究。
冻结的 v6.0 主实验、K10 变换代码和历史结果保持不变；v6.3 仅允许
validation 开发评估，不允许访问 test。

## 唯一方法改动

v6.3 保留 v6.2-D 的 5,776 个 train 场景、目标 pair、方向与标签，只按
干净轨迹末帧两端速度选择时间窗：

```text
min endpoint speed < 0.5 m/s  -> K=10
min endpoint speed >= 0.5 m/s -> K=4
```

保持不变：相对位移 `0.2 m`、双端 `alpha=0.5`、minimum-jerk、velocity
residual、动态风险标签、42 维运动归一化边表示、GENConv 和普通 BCE。

该选择不读取模型概率、梯度、validation 或 test。清单生成后必须逐行从
train 图重新计算 motion regime、K、目标身份与变换结果，再冻结训练前合同。

## 单种子停止规则

仅先运行 seed `20260621`。四项同时满足才允许追加种子：

- overall incremental ASR `>= 0.6736`；
- moving incremental ASR `>= 0.45`；
- adjacent incremental FP `<= 0.015`；
- clean pair AUC `>= 0.9849`。

禁止 `--evaluate-test`、`--split test`、任何 `--max-*` 和训练/评估阶段的
`--force`。未通过全部门槛时立即停止该方向。

## 运行顺序

### 1. 生成、逐行验证并冻结清单

```bash
cd /home/mizhou/JCAS

python3 -m jcas.workflows.motion_schedule_manifest \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --base-manifest record/v6_2/poison_rate005_motion_prioritized.csv \
  --output record/v6_3/poison_rate005_motion_k4_k10.csv

python3 -m jcas.release.verify_motion_schedule_manifest \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --poison-manifest record/v6_3/poison_rate005_motion_k4_k10.csv \
  --output record/v6_3/verification/motion_schedule_verification.json

python3 -m jcas.release.freeze_v6_3_pilot \
  --output-dir record/v6_3/contracts/pretraining
```

### 2. 只训练 seed 20260621

```bash
python3 -m jcas.workflows.trainer \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --output-dir record/v6_3/victim_seed20260621 \
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
  --poison-manifest record/v6_3/poison_rate005_motion_k4_k10.csv \
  --edge-feature-mode relative_motion_residual_dct8_v1 \
  --experimental-trigger-schedule motion_regime_k4_k10_v1 \
  --device cuda
```

### 3. Validation

```bash
CLEAN_RUN=record/v6_1/clean_seed20260621/genconv_strict_seed20260621_20260827_174820
VICTIM_RUN=$(find record/v6_3/victim_seed20260621 -mindepth 1 -maxdepth 1 \
  -type d -name 'genconv_strict_seed20260621_*' | sort | tail -n 1)

mkdir -p record/v6_3/validation

python3 -m jcas.workflows.evaluator \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --checkpoint "$CLEAN_RUN/best_model.pt" \
  --output record/v6_3/validation/reference_seed20260621.json \
  --evaluation-model-role clean_reference \
  --clean-reference-result "$CLEAN_RUN/result.json" \
  --split val \
  --target-seed 20260621 \
  --orientation-policy lower_destination_mean_speed_v1 \
  --allocation-policy fixed_symmetric_biend_v1 \
  --experimental-trigger-schedule motion_regime_k4_k10_v1 \
  --device cuda

python3 -m jcas.workflows.evaluator \
  --graph-dir graphs/av2mf_graph_v4 \
  --graph-manifest graphs/av2mf_graph_v4/manifest_grouped_v5.csv \
  --checkpoint "$VICTIM_RUN/best_model.pt" \
  --output record/v6_3/validation/victim_seed20260621.json \
  --evaluation-model-role victim \
  --clean-reference-result "$CLEAN_RUN/result.json" \
  --split val \
  --target-seed 20260621 \
  --orientation-policy lower_destination_mean_speed_v1 \
  --allocation-policy fixed_symmetric_biend_v1 \
  --experimental-trigger-schedule motion_regime_k4_k10_v1 \
  --device cuda
```

训练和评估命令均不得添加 `--evaluate-test`、`--split test`、`--max-*` 或
`--force`。
