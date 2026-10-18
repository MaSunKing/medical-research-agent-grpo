# 当前工程说明与归档同步记录

## 当前设计（2026-10-18）

当前为LangGraph显式节点、MCP真实工具的单策略循环；原始972 Process训练与50题对照见[版本三](../versions/process_final_sft/README.md)。已实现本题同query缓存重显、未读优先/部分已读后置、作用域耗尽与失败候选隐藏、保守JSON语法修复，以及软12,288/硬15,000完整请求预算。模型原文抽取仍最后兜底，Search预览与Browse正文隔离。

600题预分500 train/100 dev，全部只采Process→Pre-Final，保存完整原包和真实Final构建器冻结的MESSAGES。三题服务端试采和输入完整性通过，但有乱码/State待审问题，不能自动认定金标合格或全量采集完成。程序修复和完整导出不等于医学语义正确。

本次只同步当前文档，不声称下面的公开组件已被替换为新运行包。[架构](architecture.md)、[检索](retrieval.md)、[Final输入](final_contract.md)和[代码导航](code_navigation.md)为当前说明；后续计划不调整。

## 归档：2026-09公开源码同步记录（不是当前配置）

以下原文仅说明当时已公开源码和工程验收，包含旧8K预算及配对回执机制；不作为当前缓存范围、预算或部署身份。历史原文保留，不追溯改写。

## 1. Browse 到 State / Final 的证据结构不再丢失

每个打开的 evidence chunk 保留：

```text
source_id / text / 原始坐标与哈希
structure_kind
boundary_incomplete
table_integrity_verified
structure_audit
```

Runtime 不再根据 Final 文本二次猜测证据结构。字段缺失会被合同拒绝，而不是静默降级。

## 2. 通用表格完整性门控

- 普通统计正文不会仅因出现 patients、events、RR、CI 和数字而被判成表格。
- 有显式表格结构、表头和完整数据行的片段可进入可引用证据。
- 缺表头、半行或边界截断的表格保留 provenance 回执，但不冒充完整 evidence。
- 规则依赖结构信号和句子边界，不针对单个网站或单道题写特例。

## 3. 共享检索回执用于公平对照

Base/SFT 评测将完全相同的 tool + arguments 绑定到内容寻址 request key。首次执行冻结响应；第二次相同请求重放相同字节。它只用于配对评测，避免网络波动、来源回退和网页更新被误记成模型差异。

## 4. 8K 预算与证据卡

共享合同是 8192 tokens（输入加预留输出）。未超限时保持完整原始上下文；只有累积 evidence 使请求超限时，才使用由原文完整句子组成、带 source identity 的证据卡。若证据继续增长，Decision 可使用审计回执，Final 可使用明确 `citable=false` 的预算回执；canonical trajectory 仍保留完整原文。

## 5. Final 重复保护与正式尝试选择

Final guard 检测长 answer unit 的第二次精确重复，记录 `abnormal_repetition` 并停止继续生成。它不修改 logits、不使用 repetition penalty、不替模型重写答案。多次 Final capture 由独立审计器按生成顺序与协议字段选择正式结果，首次失败与纠正尝试均保留。

## 6. 本次公开同步范围

公开仓库同步上述检索、证据合同、预算和重复审计模块及其离线回归。服务器私有题集、模型权重、完整 token/logprob capture、API 回执、凭据与绝对路径没有上传。
