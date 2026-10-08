# 架构与可信边界

共享策略生成 Checklist、工具决策、State 和 Final。SFT 学习接口，RL 对指定生成 Token 分配独立奖励通道。Search/Browse 是策略动作；工具返回是 observation，不是可直接优化的模型生成 Token。

1. 冻结原题、requirements、constraints、policy、tokenizer、scorer 和 reward config。
2. 捕获精确 prompt、生成 token IDs、采样配置和 behavior log probabilities。
3. 执行工具，保存证据正文、source IDs、坐标、哈希和 execution receipts。
4. 构造分阶段 Judge view；Judge 判断语义，代码验证 schema、绑定与允许使用的 ID。
5. 继承上一份已验证 coverage receipt，评价新增证据并确定性合并。
6. 先把可信评分/工具事件绑定到 record/capture/token，再编译相对 advantage。
7. 对共享 LoRA 进行 clipped policy optimization，原子保存新 checkpoint。

哈希用于可信 authority 边界内的完整性检查，本身不提供独立认证。不能让 rollout 自己制造可信评分或执行 registry。离线演示使用明确标记的测试 authority；真实训练必须由独立可信服务提供查找接口。

## Citation 协议

保留完整 chunk ID 和已有的句后 citation 格式：

```xml
<answer>
证据支持的主张。
<cite id="WEB:example#s0-c0">来源说明</cite>
</answer>
```

State 引用 evidence IDs；Final 自己生成实际 attachment。代码验证标签、ID 成员资格与 attachment 位置，不能仅凭字符串证明改写后的主张被支持。

Citation 看实际附着的证据；Fidelity 看允许的证据是否支持已经写出的主张；Completeness 看是否覆盖用户要求。三者独立，不用一个总体印象替代。

无需自动补 citation、迁移短 ID、合并两套 LoRA，或另引入一个答案撰写模型。

修改 judge/effective_config.json 的 provider/model 后，执行 python -B tools/refreeze.py，再跑离线检查。新 scorer identity 不代表语义验收通过，也不能让旧回执自动变成新 scorer 的评分。

## 可移植与部署专用代码

公开版提供核心算法、契约与工具后端，不包含私有 fixture 绑定的集群采集服务。根 CLI 用于离线演示/检查，不是生产 rollout 服务。部署时应接入自己的 capture/execution 服务，同时保持导出的接口与放行规则。
