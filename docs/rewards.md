# 稠密奖励与归因

语义通道包括 Checklist、Search query、Browse source focus、State、Final 与可评价的 Stop。Tool 是单独的任务收益通道。冻结配置示例见 training/config.json。

| 信号 | 职责 |
|---|---|
| Checklist | 子任务覆盖与原题范围忠实度 |
| Search | query 是否针对当前信息缺口 |
| Browse source focus | 打开正文前的选源质量 |
| State | 证据状态判断及实际 evidence attachment |
| Final | 完整性、证据忠实度与实际 citation 支持 |
| Evidence gain | 代码根据可信 before/after receipt 计算覆盖变化 |
| Tool cost / policy event | 可信执行成本或有 authority 证明的违规 |
| Stop | 可信且可评价的主动停止；forced termination 是 N/A |

Stop 只有在模型主动生成、存在可信 termination receipt 且能精确绑定对应 Token 时才具备训练资格。`forced / quota_exhausted / error / cancelled` 只保留终止审计，不把 Stop 通道补成 0 分，也不进入 Stop advantage。

不能用读后 evidence gain 反改事前 Browse 评分。Provenance、schema 和 scorer identity 是放行条件，不是正奖励来源。Pending/unobservable 不偷偷补成零；固定四轨迹题组中任一必需核心奖励缺失，整组不得计算 advantage。

## 增量 Coverage

shared/gain_contract.py 将 prior receipt 固定为不可变状态。下一次 Browse 的 before 引用上一步 after，在冻结 requirements 下评价 retained/new evidence。原子字段描述目标选项/成员、outcome、人群适用性、合并后完整性、反证和新增支持 ID。

代码映射 unknown=0、partial=0.5、direct=1。无关新证据保留既有 partial，gain 为零；新增实质支持可将 unknown 推进至 partial；充分的合并证据可将 partial 推进至 direct。Contradiction 单独记录，不作为正收益；撤销证据需要可信 invalidation event，不能靠自由标签降级。

## Advantage 与 Loss

同题各 rollout 与其余 rollout 做 leave-one-out 比较。局部通道采用配置定义的 baseline；Tool 任务收益计入合格的后续 gain/cost event。概念公式为：

```text
A(channel) = local_weight × local_relative_credit
           + final_weight × final_relative_credit
ratio = exp(current_logprob - behavior_logprob)
loss = -mean(min(ratio × A, clip(ratio, 1-epsilon, 1+epsilon) × A))
```

这里是 GRPO 风格的组内相对优化，不声称复现所有 GRPO 论文的标准化。精确 baseline 和 row weight 见 training/core.py。示例配置只有 Tool 接收 0.5 × Final，不是所有前序阶段都传播 Final credit。

局部分数非负也能产生负 advantage；工具成本和可信违规事件还可产生负收益。初始 ratio 接近 1、平均 loss 接近零，不等于梯度为零。

采集与 replay 必须一致复现声明的采样分布和 logits processor。奖励绑定精确生成 Token；工具返回文本或附近猜测的 Token 不可替代。
