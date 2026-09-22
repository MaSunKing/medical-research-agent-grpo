# 近期工程同步（2026-09）

本页记录已经同步到公开源码、并在冻结真实轨迹或离线合同测试中验证的工程改动。它们用于减少评测噪声，不替模型自动修改医学答案。

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

Raw/SFT 评测将完全相同的 tool + arguments 绑定到内容寻址 request key。首次执行冻结响应；第二次相同请求重放相同字节。它只用于配对评测，避免网络波动、来源回退和网页更新被误记成模型差异。

## 4. 8K 预算与证据卡

共享合同是 8192 tokens（输入加预留输出）。未超限时保持完整原始上下文；只有累积 evidence 使请求超限时，才使用由原文完整句子组成、带 source identity 的证据卡。若证据继续增长，Decision 可使用审计回执，Final 可使用明确 `citable=false` 的预算回执；canonical trajectory 仍保留完整原文。

## 5. Final 重复保护与正式尝试选择

Final guard 检测长 answer unit 的第二次精确重复，记录 `abnormal_repetition` 并停止继续生成。它不修改 logits、不使用 repetition penalty、不替模型重写答案。多次 Final capture 由独立审计器按生成顺序与协议字段选择正式结果，首次失败与纠正尝试均保留。

## 6. 本次公开同步范围

公开仓库同步上述检索、证据合同、预算和重复审计模块及其离线回归。服务器私有题集、模型权重、完整 token/logprob capture、API 回执、凭据与绝对路径没有上传。
