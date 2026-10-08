# 第三方代码与许可证声明

本项目采用 Apache-2.0，完整许可证见 LICENSE；许可证原文保持不变。

retrieval/dr_agent 含有适配自 [rlresearch/dr-tulu](https://github.com/rlresearch/dr-tulu) 的 Agent/工具代码，既有版权与许可证声明保留。此公开版本的适配涉及医学检索、证据 provenance、模板过滤与 chunk-selection 契约。项目名称描述自身稠密奖励与本地部署方向，不取消复用代码的必要署名。

`versions/legacy_retrieval/dr_agent` 是同一来源的历史冻结副本，适用上述声明；它不是新的独立第三方项目。新版源码与冻结副本分别记录发布身份，不混用实验结果。

PyTorch、Transformers、PEFT、bitsandbytes、BeautifulSoup、lxml、sentence-transformers、FastMCP 等依赖单独分发，受各自许可证约束。Qwen、embedding 和 reranker 权重不包含在仓库中，需自行获取并遵守相关条款。外部 API 服务独立配置和授权。
