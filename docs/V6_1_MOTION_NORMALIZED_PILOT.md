# v6.1 运动归一化相对轨迹 pilot

本 pilot 仅用于本地 Argoverse 2 和自训练 GENConv 的离线模型鲁棒性研究。

## 不变项

- 动态风险标签：`d0=5 m, tau=1 s, decel=4 m/s²`；
- grouped-v5 图划分；
- GENConv 与普通 BCE；
- 5% 训练场景变化；
- `0.2 m / K10 / minimum-jerk / velocity residual / alpha=0.5`；
- 公共 validation 目标池和 `target_seed=20260621`；
- 不访问 test。

## 唯一新增表示

`relative_motion_residual_dct8_v1` 在原34维 edge feature 后追加8维：

1. 对最后11个状态的 pair 相对位置拟合常速度趋势；
2. 从相对位置中减去该趋势；
3. 在终点 LOS/切向局部坐标系内计算3个 LOS 与2个切向低频系数；
4. 增加终点相对速度残差的 LOS/切向分量；
5. 增加完整窗口有效标记。

表示由轨迹直接确定，不读取标签、模型输出、梯度、validation 或 test。

## 两阶段单变量顺序

1. v6.1a：沿用 `record/v6/poison_rate005_fixed_symmetric.csv`，只改变特征表示。
2. v6.1b：仅当 v6.1a 显示运动目标收益时，才使用速度分层清单。

速度分层配额固定为：`40% / 10% / 15% / 20% / 15%`，对应双端最小速度
`<0.5 / 0.5–2 / 2–5 / 5–10 / >=10 m/s`。

## 继续门槛

- 总体 Incremental ASR `>= 67.36%`；
- 运动目标 Incremental ASR `>= 45%`；
- Adjacent incremental FP `<= 1.5%`；
- clean Pair AUC `>= 0.9849`。

首个 seed 未通过时停止，不运行另外两个 seed，不访问 test。

## 训练前冻结

在任何 v6.1 训练前运行：

```bash
python3 -m jcas.release.freeze_v6_1_pilot
```

该命令绑定当前源码、原 v6.0 训练清单、grouped-v5 图契约、单种子配置和上述四项继续门槛。它不是正式 test contract。
