# 源码阅读导航

| 功能 | 当前源码 |
|---|---|
| 论文 Search | [pubmed_apis.py](../retrieval/dr_agent/mcp_backend/apis/pubmed_apis.py)、[dual_paper.py](../retrieval/dr_agent/mcp_backend/apis/dual_paper.py) |
| 论文语义重排 | [paper_semantic_ranking.py](../retrieval/dr_agent/mcp_backend/apis/paper_semantic_ranking.py) |
| 网页 Search / 重排 | [medical_web_apis.py](../retrieval/dr_agent/mcp_backend/apis/medical_web_apis.py)、[web_semantic_ranking.py](../retrieval/dr_agent/mcp_backend/apis/web_semantic_ranking.py) |
| 候选窗口 | [candidate_window.py](../retrieval/runtime/candidate_window.py)、[candidate_latest_window_v13.py](../retrieval/runtime/candidate_latest_window_v13.py) |
| Checklist / State / Decision | [checklist_feedback_v55.py](../retrieval/runtime/checklist_feedback_v55.py)、[state_compact_v38.py](../retrieval/runtime/state_compact_v38.py)、[sft_interface_v20.py](../retrieval/runtime/sft_interface_v20.py) |
| 失败预算 | [environment_failure_budget_v1.py](../retrieval/runtime/environment_failure_budget_v1.py) |
| 证据 freshness | [evidence_freshness_v57.py](../retrieval/runtime/evidence_freshness_v57.py) |
| Process / Final 实际训练代码 | [版本三](../versions/process_final_sft/README.md) |
| Final citation / 固定输入对照 | [final_citation_alias_v1.py](../versions/process_final_sft/final/final_citation_alias_v1.py)、[evaluate.py](../versions/process_final_sft/final/evaluate.py) |
| 历史共享 LoRA / dense reward / replay | [历史单 LoRA](../versions/single_lora/README.md)、[training/](../training/)、[judge/](../judge/) |

带版本号的文件常是依赖关系，不是多个可随意选择的算法。v13 依赖 v11 的 local_scores，不启用 v11 全池重排。公开代码不带私有训练数据、captures和服务器环境；版本框架不等于所有未来步骤已经实现。
