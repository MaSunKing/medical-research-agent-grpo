# 当前设计与公开源码导航

当前节点式实现及最新检索已在[版本三current目录](../versions/process_final_sft/current/README.md)独立公开，附原始/发布文件SHA。保留此前组件和训练入口，不替换版本一、二、不修改旧规模gate。当前目录为源码公开包，不是附带题目/权重/凭据的服务器部署包；UTF-8/LF规范化、共享manifest重绑和prepare显式路径适配均有记录。

## 当前运行的模块边界

下表中src、tests、scripts及retrieval_snapshot_v2位于[公开runtime](../versions/process_final_sft/current/runtime)；批量采集/导出入口位于其上层current目录。

| 冻结包模块 | 职责 |
|---|---|
| src/trainable_agent/engine.py | LangGraph节点、工具执行、State传递、停止与Pre-Final |
| contracts.py / prompts.py | 动作schema、R/E/来源校验与英文阶段Prompt |
| search_cache.py | 本题重复Search重显、阅读状态分组和作用域可用性 |
| model_json.py | 保守JSON语法修复；不修医学内容或错误ID |
| context_processing/pipeline.py | 完整请求计量、阶段输入、软硬线及最后兜底 |
| context_processing/ | 候选/正文隔离、重复工程包装投影、表格和引用保护 |
| tools.py / retrieval_deadlines.py | MCP工具桥接、worker和超时/未知结果回执 |
| storage.py / recovery.py / training.py | SQLite回执、显式恢复、真实当步输入与合法target导出 |
| retrieval_snapshot_v2/backend/agent/dr_agent/ | PubMed/S2/Serper、正文解析、排序、清洗与片段证据 |
| retrieval_snapshot_v2/source_registry.py / worker.py | 从已完成Search回执恢复来源映射，保持Browse引用连续；不额外搜索 |
| packaging/prefinal600/ | 600题500/100划分校验、Process-only队列、真实Final输入冻结、一致性导出 |

原始972训练代码单独公开于[original972_training](../versions/process_final_sft/current/original972_training)，按10,575行、1,322更新和effective-batch target-token mean核对；不能以本页下方的既有训练文件身份替代。

当前Process更新实现见[Process-GRPO源码](../versions/process_final_sft/current/process_grpo_update/README.md)，包括轨迹等权目标、概率校正、确定性和断点校验；根目录training保留其既有实现身份。[逐步设计](../versions/process_final_sft/engineering.md)解释各模块为何存在以及限制，[算法说明](../versions/process_final_sft/grpo.md)将奖励连接到实际参数更新。

从实际行为开始阅读，可先看[v3完整案例与失败定位](../examples/version3_cases/README.md)。`examples/trajectory_demo`保留原有历史身份，不作为v3/GRPO的执行案例。

## 已公开组件与参考入口

| 功能 | 已公开快照/参考源码 |
|---|---|
| 论文 Search | [pubmed_apis.py](../retrieval/dr_agent/mcp_backend/apis/pubmed_apis.py)、[dual_paper.py](../retrieval/dr_agent/mcp_backend/apis/dual_paper.py) |
| 论文语义重排 | [paper_semantic_ranking.py](../retrieval/dr_agent/mcp_backend/apis/paper_semantic_ranking.py) |
| 网页 Search / 重排 | [medical_web_apis.py](../retrieval/dr_agent/mcp_backend/apis/medical_web_apis.py)、[web_semantic_ranking.py](../retrieval/dr_agent/mcp_backend/apis/web_semantic_ranking.py) |
| 候选窗口 | [candidate_window.py](../retrieval/runtime/candidate_window.py)、[candidate_latest_window_v13.py](../retrieval/runtime/candidate_latest_window_v13.py) |
| Checklist / State / Decision | [checklist_feedback_v55.py](../retrieval/runtime/checklist_feedback_v55.py)、[state_compact_v38.py](../retrieval/runtime/state_compact_v38.py)、[sft_interface_v20.py](../retrieval/runtime/sft_interface_v20.py) |
| 失败预算 | [environment_failure_budget_v1.py](../retrieval/runtime/environment_failure_budget_v1.py) |
| 证据 freshness | [evidence_freshness_v57.py](../retrieval/runtime/evidence_freshness_v57.py) |
| Process / Final 训练边界与已完成状态 | [版本三](../versions/process_final_sft/README.md)；其中现有脚本保持原快照身份 |
| Final citation / 固定输入对照 | [final_citation_alias_v1.py](../versions/process_final_sft/final/final_citation_alias_v1.py)、[evaluate.py](../versions/process_final_sft/final/evaluate.py) |
| 历史共享 LoRA / dense reward / replay | [历史单 LoRA](../versions/single_lora/README.md)、[training/](../training/)、[judge/](../judge/) |

带版本号的文件常是依赖关系，不是多个可随意选择的算法。v13依赖v11的local_scores，不启用v11全池重排。公开清单SOURCE_EXPORT_CURRENT.json核验的是已发布快照，不证明当前冻结包与其相同。运行必须以各自源码/配置/模型哈希、messages和回执为准。版本一、二及后续计划原样保留。
