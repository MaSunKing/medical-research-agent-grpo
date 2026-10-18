# 已公开检索源码与当前部署边界

本目录保留既有工具后端与组件快照。**[最新完整检索源码在版本三runtime/retrieval_snapshot_v2](../versions/process_final_sft/current/runtime/retrieval_snapshot_v2)**，由当前trainable_agent的Execute/MCP调用，并附共享上下文哈希。旧入口不覆盖、不与当前包混用。当前60:40排序、缓存重显、可用性及软硬预算见[检索设计](../docs/retrieval.md)和[架构](../docs/architecture.md)。历史版本目录保持不动。

主要入口位于 `dr_agent/mcp_backend/apis/`：

- [pubmed_apis.py](dr_agent/mcp_backend/apis/pubmed_apis.py)：PubMed 检索与自然语言查询回退。
- [dual_paper.py](dr_agent/mcp_backend/apis/dual_paper.py)：论文后端合并。
- [paper_semantic_ranking.py](dr_agent/mcp_backend/apis/paper_semantic_ranking.py)：MedCPT Search 重排。
- [medical_web_apis.py](dr_agent/mcp_backend/apis/medical_web_apis.py)、[web_semantic_ranking.py](dr_agent/mcp_backend/apis/web_semantic_ranking.py)：网页搜索、候选与 MiniLM 重排。
- [web_preflight.py](dr_agent/mcp_backend/apis/web_preflight.py)：访问探针。
- [browse_preprocess.py](dr_agent/mcp_backend/apis/browse_preprocess.py)、[medical_document_parser.py](dr_agent/mcp_backend/apis/medical_document_parser.py)：清洗及结构解析。
- [medical_passage_retriever_v28.py](dr_agent/mcp_backend/apis/medical_passage_retriever_v28.py)、[hybrid_passage_retriever.py](dr_agent/mcp_backend/apis/hybrid_passage_retriever.py)：正文片段检索。
- [pdf_policy_v54.py](dr_agent/mcp_backend/apis/pdf_policy_v54.py)、[table_integrity.py](dr_agent/mcp_backend/apis/table_integrity.py)：PDF 路径与表格结构。

以各冻结调用点实际传入的limit为准。8是模型可见窗口，不是累计池容量；当前重复Search只重显原回执前8候选，不联网凑数。当前候选按未读优先/部分已读后置稳定分组，明确耗尽/失败按作用域过滤。

公开仓库不附语义模型权重或服务器环境，实际embedding/reranker路径与API按部署配置登记。现有[导出清单](../SOURCE_EXPORT_CURRENT.json)只代表已公开快照，不替代新运行包的逐文件哈希；详见[代码导航](../docs/code_navigation.md)。
