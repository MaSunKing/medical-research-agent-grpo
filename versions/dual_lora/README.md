# 版本二：双 LoRA 框架

[返回项目首页](../../README.md) · [单 LoRA](../single_lora/README.md) · [Process / Final](../process_final_sft/README.md)

## 迭代原因

单 LoRA 把检索决策、State 和答案生成交给同一套可训练参数，难以单独判断或优化两类能力。双 LoRA 将 Process 与 Final 分成独立 adapter，分别管理训练和输入，检验角色分离能否改善任务耦合。角色分离本身不证明效果更好，需同题验证。

本页定义复用已有 **972 道 teacher 轨迹** 的双角色重训方案，不重新采集。972 指该轮清洗后的轨迹集合，不把早期说明中的 974 原始训练题数量混为同一个数据版本。新的 972 题双 LoRA 重训与配对 50 题结果尚待执行。

## 角色与数据

```mermaid
flowchart LR
    Data[已有 972 道 teacher 轨迹] --> Split[按阶段拆分并保留因果历史]
    Split --> P[Process / Retrieval LoRA]
    Split --> F[Final LoRA]
    P --> Tools[Checklist → Decision → Search/Browse → State → Stop]
    Tools --> Input[真实 Pre-Final 输入]
    Input --> F
    F --> Answer[有证据引用的答案]
```

- 两套 adapter 共享 Qwen3-8B frozen base，参数与优化器分别管理；不在同一个正在更新的 adapter 上仅屏蔽 Final loss 就声称 Final 已冻结。
- Process 学 Checklist、具体 Search/Browse 动作、State 和 Stop；Final 学实际证据输入到答案。输入侧的历史、状态、工具结果不参与 target loss。
- 先使用已有 teacher 的阶段记录建立基线。teacher Pre-Final 与真实 Process 产生的输入有分布差异；后续 on-policy Pre-Final 适配属于下一版本，不回填成本版本已经完成。
- 切换 adapter 必须重新构建输入与 prefill，不能复用另一个 adapter 的 KV cache。
- 单 LoRA 与本版本共用 [冻结旧检索](../legacy_retrieval/README.md)，保持同一历史工具、预算和 citation 合同，避免同时改模型与检索造成归因混乱。

## 本版本参数与验收

| 项目 | 本版本记录 |
|---|---|
| Base | Qwen3-8B |
| Teacher 题目 | 972；复用已有轨迹，不重新采集 |
| 验证题目 | 同50题配对验证 |
| Process / Final 样本数 | — |
| LoRA rank / alpha / dropout | — |
| LR / batch / accumulation | — |
| Epoch / updates | — |
| 训练执行与结果 | — |

未确认的信息留空，不从单 LoRA 借用参数或结果。本版本训练入口的接线与验收记录也待补齐。验收应包括阶段 mask、无未来历史、State→下一输入一致、证据 ID 可见、最长样本前反向、参数确有更新、checkpoint 完整性以及新进程续跑。

拆分以阶段capture为边界：Process仅保留Checklist、具体工具Decision、State与Stop target；Final仅保留其当时可见证据输入及Final target。两套adapter均采用completion-only，不监督输入历史与Observation。各自loss的样本/token归一化、阶段权重及batch口径待实现冻结，不从单LoRA或后续Process700训练器推断已经采用同一算法。

## 同 50 题验证

比较 Base、旧单 LoRA SFT、新双 LoRA，固定题目、工具后端、预算、seed、温度和评分协议；同时记录流程失败和有效配对。历史结果留在单 LoRA 页面，不替代本轮结果。

既看完整答案，也分开核查 query、真实获取的证据、State 覆盖和 Final 证据绑定。若完整答案与检索指标不一致，报告差异，不据此单独宣称“模型更好”。50 题若已用于调参，应称开发验证集，不再视为未触碰的最终测试集。
