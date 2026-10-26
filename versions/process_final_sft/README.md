# 版本三｜Process与Final解耦训练

**让小模型先找对证据，再可靠地使用证据回答。** 本版本把真实检索过程与答案综合分开训练、分开评价；当前已完成Process-SFT、Final-SFT对照，以及冻结Base Final下的一轮Process-GRPO。

[项目首页](../../README.md) · [源码入口](current/README.md) · [历史归档](archive/README.md)

## 阅读顺序

| 章节 | 主要内容 |
|---|---|
| **[1. 系统设计与题集](design.md)** | 模型、节点、检索、Pre-final、题目来源与数据隔离 |
| **[2. Process-SFT](process_sft.md)** | 972题如何变成阶段样本，mask、loss与真实训练参数 |
| **[3. Final-SFT实验](final_sft.md)** | gold构建、500/100划分、Base对照与暂不加训的依据 |
| **[4. Process-GRPO](grpo.md)** | 400条轨迹、本地奖励、概率校正、25批更新及后续设计 |
| **[5. 评测方法与结果](evaluation.md)** | 指标含义、完整对照表、失败、统计边界与模型选择 |
| **[6. 逐题评分与问题](records.md)** | 50题Process四组、100题Final四组；每条均有分数与文字 |

## 研究路线

```mermaid
flowchart TD
    A[Qwen3-8B量化基座] --> B[972题Process-SFT]
    B --> C[冻结Pre-final与gold]
    C --> D[Final-SFT与Base对照]
    D --> E[保留Base Final]
    B --> F[100训练题各4条轨迹]
    E --> F
    F --> G[本地Process奖励与25批GRPO更新]
    G --> H[GRPO13与GRPO25重新检索评测]
```

上图是训练演进；实际单题问答的节点与数据流见[系统设计](design.md#单题如何运行)。Final-SFT没有接入本轮GRPO，后者始终冻结Base Final。

## 已完成的实验

| 实验 | 规模 | 当前结果 |
|---|---|---|
| Process-SFT | 972题、10575阶段样本、1322次更新 | 作为后续Process策略起点 |
| Final-SFT | 500训练题；dev100×Base及三个checkpoint | SFT的1/1.5/2轮均未超过Base Final |
| Process-GRPO | 100题×4条，99组可用、25批更新 | 保留第13批中途与第25批最终checkpoint |
| Process开发评测 | 50题×Base/SFT/GRPO13/GRPO25 | 200条逐题审核；188个有效HealthBench-style答案分 |

**当前的内容效果候选是GRPO13。** 它的实际证据交付覆盖为69.75%，高于历史SFT的55.83%；GRPO25完成率更高。State重大错误仍存在，综合质量值改善不明确；同题区间跨0、历史运行配置存在差异，尚不能宣称GRPO稳定或显著优于SFT。[查看完整结果与解释](evaluation.md#process四组结果)

**当前答案模块仍选Base Final。** 在相同冻结输入的共同有效97题上，Base为77.06分，Final-SFT的1/1.5/2轮分别为58.40、61.57、65.98分；这些是项目评分，不是HealthBench分。[查看为何暂停加训](final_sft.md#为什么没有继续增加轮次)

## 如何理解这份记录

- **设计**：单Process策略、显式节点与真实工具，Final独立综合证据。
- **训练**：Process-SFT、Final-SFT、Process-GRPO分别记录初始化、数据、目标与checkpoint。
- **评测**：Process的信息交付与State可靠性单独打分，答案另用原生HealthBench rubric；不将两者都称为准确率。
- **后续**：新独立题集同配置验证尚未启动；下一轮采集和训练须另行冻结，不在此预报结果。

公开内容包括方法、参数、脱敏评分及逐题问题摘要。完整题目、gold、模型答案、证据原句、权重与回执保存在私有归档。源码快照的适用身份与复现入口见[代码说明](design.md#实际交付与完整存档)。
