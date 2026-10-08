# 当前检索源码

`dr_agent/` 是当前工具后端，`common.py` 是共享工具加载组件；`runtime/` 放置当前工程合同与模型可见展示组件。历史后端单独冻结在 [legacy_retrieval](../versions/legacy_retrieval/README.md)，不随本目录更新。

主要入口位于 `dr_agent/mcp_backend/apis/`：

- [pubmed_apis.py](dr_agent/mcp_backend/apis/pubmed_apis.py)：PubMed 检索与自然语言查询回退。
- [dual_paper.py](dr_agent/mcp_backend/apis/dual_paper.py)：论文后端合并。
- [paper_semantic_ranking.py](dr_agent/mcp_backend/apis/paper_semantic_ranking.py)：MedCPT Search 重排。
- [medical_web_apis.py](dr_agent/mcp_backend/apis/medical_web_apis.py)、[web_semantic_ranking.py](dr_agent/mcp_backend/apis/web_semantic_ranking.py)：网页搜索、候选与 MiniLM 重排。
- [web_preflight.py](dr_agent/mcp_backend/apis/web_preflight.py)：访问探针。
- [browse_preprocess.py](dr_agent/mcp_backend/apis/browse_preprocess.py)、[medical_document_parser.py](dr_agent/mcp_backend/apis/medical_document_parser.py)：清洗及结构解析。
- [medical_passage_retriever_v28.py](dr_agent/mcp_backend/apis/medical_passage_retriever_v28.py)、[hybrid_passage_retriever.py](dr_agent/mcp_backend/apis/hybrid_passage_retriever.py)：正文片段检索。
- [pdf_policy_v54.py](dr_agent/mcp_backend/apis/pdf_policy_v54.py)、[table_integrity.py](dr_agent/mcp_backend/apis/table_integrity.py)：PDF 路径与表格结构。

以调用点实际传入的 limit 为准。后端可返回多于 8 条结果，8 是 Runtime 可见窗口，不是累计池容量。

公开仓库提供源代码，不附语义模型权重或服务器环境。真实运行须配置本地 BGE、网页 MiniLM、论文 MedCPT 模型与相应 API；源码身份见 [当前导出清单](../SOURCE_EXPORT_CURRENT.json)。
