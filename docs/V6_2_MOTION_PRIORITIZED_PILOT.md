# v6.2-D：运动 pair 优先的数据投毒 pilot

本实验只用于本地 AV2 与自训练 GNN 的离线模型鲁棒性研究。冻结的正式主实验仍是 v6.0；本 pilot 不访问 test。

## 方法边界

v6.2-D 只改变训练清单中的目标 pair：

- 保留 v6.0 的同一批 5,776 个投毒场景和 5% 比例；
- 每个场景重新枚举满足动态负标签、双向监督、物理可行和双端连续帧要求的无向 pair；
- 若存在两端末帧速度都不低于 `0.5 m/s` 的 pair，则在这类 pair 中用稳定哈希选一个；
- 否则在全部合格 pair 中用稳定哈希选一个；
- 不读取模型概率、梯度、validation 或 test。

保持不变：动态风险标签、0.2 m 相对位移、`K=10`、minimum-jerk、velocity residual、`alpha=0.5`、GENConv、42 维 v6.1 表示和普通 BCE。

## 单种子门槛

只运行 seed `20260621`。四项必须同时满足才允许继续另外两个种子：

- overall incremental ASR `>= 0.6736`；
- moving (`min endpoint speed >= 0.5 m/s`) incremental ASR `>= 0.45`；
- adjacent incremental FP `<= 0.015`；
- clean pair AUC `>= 0.9849`。

## 禁止事项

- 不添加 `--evaluate-test`；
- 不切换到 `test`；
- 不使用 `--max-*` 或 `--force` 运行训练/评估；
- 单种子未通过全部门槛时，不运行 seed `20260622/20260623`。
