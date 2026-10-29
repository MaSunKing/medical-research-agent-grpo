# 系统架构导航

版本三的完整架构说明集中在[系统架构与执行机制](../versions/process_final_sft/engineering.md)。检索与证据整理策略（Process）负责取得并判断材料，答案生成模块（Final）读取原问题和实际交付材料，生成回答、引用及未解决边界。本页按阅读目的导航；参数训练关系、运行流程和实验结果分别进入对应页面。

| 阅读目的 | 入口 |
|---|---|
| 单题运行：需求拆解、检索循环、材料汇集与答案生成 | [运行流程](../versions/process_final_sft/engineering.md#runtime-flow) |
| 上下文、缓存、预算、校验、证据身份与恢复 | [共享执行机制](../versions/process_final_sft/engineering.md#shared-mechanisms) |
| 实际请求、固定输入快照与训练／评测模式 | [交接与接入边界](../versions/process_final_sft/engineering.md#training-evaluation-interface) |
| 模型、量化、题集及配置身份 | [数据与实验设置](../versions/process_final_sft/design.md) |
| 检索策略训练主线与答案模块单独对照 | [版本三实验地图](../versions/process_final_sft/README.md) |
| 三项评测与逐题证据 | [评测地图](../versions/process_final_sft/evaluation.md) · [逐题记录](../versions/process_final_sft/records.md) |
| 当前维护源码、冻结实验实现和历史组件 | [当前源码入口](../versions/process_final_sft/current/README.md) · [代码导航](code_navigation.md) |

Process与Final是角色，Base／SFT／GRPO是策略参数条件，LangGraph／MCP／SQLite是工程设施；它们不属于同一个层级。当前答案模块使用同一Qwen3-8B并禁用项目的Process适配器，称为基座答案生成器（Base Final）；本轮参数不更新，但输入材料和回答仍可变化。职责和参数单独控制不要求另部署一套基座。[术语解释](../versions/process_final_sft/glossary.md)

<a id="一个策略循环显式节点与工具边界"></a>
<a id="各阶段能看到什么"></a>
<a id="身份与证据"></a>
<a id="上下文与失败"></a>
<a id="双层pre-final"></a>
<a id="basesft与grpo怎样作用于同一个流程"></a>
<a id="工程约束如何支持策略比较"></a>
<a id="持久化与恢复的边界"></a>

相关主题由完整架构页统一说明，阅读时按上表进入对应部分。
