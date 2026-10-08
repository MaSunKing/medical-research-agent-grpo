# 当前实现：从哪里开始读代码

公开仓库保留了一部分带版本号的模块，目的是维持实际导入关系、scorer 身份与合同回归。它们不是让使用者从多个版本中任选其一。下面按功能列出当前入口。

## 推荐阅读顺序

| 你想看什么 | 当前入口 |
|---|---|
| 无模型可运行验证 | [run_pipeline.py](../../docs/../run_pipeline.py)、[离线例子](../../docs/../examples/offline_demo.py) |
| 一道真实问题怎么走完整流程 | [静态轨迹案例](../../docs/../examples/trajectory_demo/README.md) |
| 工具动作结束边界 | [agent/action_boundary_v46.py](../../docs/../agent/action_boundary_v46.py) |
| Final citation 解析与完整 ID | [agent/citation_ids_v4_1.py](../../docs/../agent/citation_ids_v4_1.py) |
| 候选预览与预算选择 | [agent/preview_selection_v23.py](../../docs/../agent/preview_selection_v23.py)、[preview_tokens_v22.py](../../docs/../agent/preview_tokens_v22.py) |
| 共享阶段输入与预算 | [agent/shared_interface.py](../../docs/../agent/shared_interface.py)、[shared_budget_v52.py](../../docs/../agent/shared_budget_v52.py)、[evidence_cards_v32.py](../../docs/../agent/evidence_cards_v32.py) |
| Final 重复与多次生成审计 | [final_repetition_guard.py](../../docs/../agent/final_repetition_guard.py)、[final_attempt_audit.py](../../docs/../agent/final_attempt_audit.py) |
| Browse→Runtime 证据合同 | [evidence_contract_v14.py](../../docs/../agent/evidence_contract_v14.py)、[evidence_exact_v31.py](../../docs/../agent/evidence_exact_v31.py) |
| 正文提取与表格完整性 | [medical_document_parser.py](../legacy_retrieval/dr_agent/mcp_backend/apis/medical_document_parser.py)、[table_integrity.py](../legacy_retrieval/dr_agent/mcp_backend/apis/table_integrity.py) |
| 返回窗口与 provenance 身份 | [evidence_handoff_v3.py](../legacy_retrieval/dr_agent/mcp_backend/apis/evidence_handoff_v3.py)、[returned_window_identity.py](../legacy_retrieval/dr_agent/mcp_backend/apis/returned_window_identity.py) |
| SFT 样本构建与训练 | [sft/build_alignment.py](../../docs/../sft/build_alignment.py)、[train_tc2.py](../../docs/../sft/train_tc2.py) |
| 所有阶段的 Judge 计划与验证 | [judge/tiered_allstages_v1/pipeline.py](../../docs/../judge/tiered_allstages_v1/pipeline.py) |
| State 及阶段评分核心 | [judge/tiered_allstages_v1/engine.py](../../docs/../judge/tiered_allstages_v1/engine.py) |
| Final 分维度评判 | [judge/tiered_judge_v2/isolated.py](../../docs/../judge/tiered_judge_v2/isolated.py) |
| 增量覆盖与不可变回执 | [shared/gain_contract.py](../../docs/../shared/gain_contract.py) |
| 奖励编译与组内优势 | [training/core.py](../../docs/../training/core.py) |
| 裁剪策略损失 | [training/policy_loss.py](../../docs/../training/policy_loss.py) |
| Qwen3 专用行为概率 replay | [training/layer_replay.py](../../docs/../training/layer_replay.py) |
| 实际 LoRA 更新 | [training/train.py](../../docs/../training/train.py) |

## Judge 的版本目录如何理解

`tiered_allstages_v1/pipeline.py` 是统一计划入口；`tiered_judge_v2/isolated.py` 提供 Final 的维度隔离；`tiered_judge_v1` 与 `general_final_v6` 中仍有被当前入口使用的策略、聚合与合同模块。它们是依赖关系，不是三套同时推荐的替代服务。

`pipeline.profile()` 明确绑定活动 prompt、effective config 和依赖源码 hash。当前导入从 `active_policy_v11.py` 开始，后者会继承前序策略。删除或重命名这些文件需要同步修改导入、冻结清单与回归测试；不能只为外观将其移入 `docs/history/`。

后续如整理稳定 API，应先提供经过测试的功能入口，再迁移内部文件。本轮只增加阅读导航，不改变算法、评分合同或导入结构。
