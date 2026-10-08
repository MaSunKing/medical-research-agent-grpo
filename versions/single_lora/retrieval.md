# 检索与证据漏斗

```text
Search 候选 → Browse 选源 → HTML/XML section
→ 通用模板噪声过滤 → 保留坐标的 chunks
→ BM25/BGE 召回 → MiniLM 重排
→ 按预算返回正文片段 → State / Final context
```

后端支持网页与医学 XML。排序前做通用结构/模板清洗，而不是针对某个域名写特例。保留正文、短证据陈述及原始坐标/哈希；不按“参考文献”标题一刀切删除内容。

Chunk 边界兼顾中文/全角标点、闭合符号、省略号、URL 与小数。这是多语言切分支持，不代表所有语言的检索质量已验证。默认 BGE-small-en 和 MS-MARCO MiniLM 偏英语，应按实际部署语言评价效果。

BM25 与 BGE 提供候选，MiniLM 重排后执行片段/token 预算。PubMed、Semantic Scholar、web search 和文档解析服务需自行配置；公开版不携带凭证或权重。

当前 Browse 回执把 `structure_kind`、`boundary_incomplete`、`table_integrity_verified` 与 `structure_audit` 一起传入 Runtime。普通统计正文即使包含 patients、events、RR 和 95% CI，只要保持完整句子边界，就不会仅凭这些词被重分类为表格；完整表格保留，缺表头或被截断的表格只保留 provenance 审计，不作为完整可引用 evidence。

评测时，相同 tool 与参数使用内容寻址的冻结回执，第二个 arm 复用相同响应字节，避免网络波动污染 Raw/SFT 对照。8K 合同预算内保持原始完整上下文；只有累积证据超限时才使用有来源绑定的证据卡，并在必要时降级为明确不可引用的预算回执，完整原文仍保留在 canonical trajectory。

排查证据质量时，逐层比较原始/解析文档、全部 section/chunk、实际选中片段和 Final 真正收到的输入。合法引用 ID 或干净正文都不等于实际主张被支持。
