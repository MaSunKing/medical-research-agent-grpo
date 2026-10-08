# 当前 Runtime 组件

本目录是最新检索工程的组件源码，不是携带私有运行环境的一键服务器包。集成时按下列依赖顺序安装，不能只复制一个补丁文件：

1. 工具结果与候选结构 → `candidate_window.CandidateHistory` → `candidate_latest_window_v13.install`。
2. Checklist exact anchors → `checklist_feedback_v55` / `checklist_examples_v56` → `sft_interface_v20` 展示。
3. Browse 状态 → `environment_failure_budget_v1` 回执结算与黑名单扩展。
4. Evidence 精确合同 → `evidence_freshness_v57` → 共享阶段输入构建与 Pre-Final 导出。

候选窗口最新组沿后端排序，历史组使用 MiniLM；没有第三次合并重排。`candidate_semantic_window_v11.local_scores` 是 v13 使用的历史评分依赖，不表示启用 v11 的全池重排。

`build_environment_failure_budget_v1.py` 是已有 runner 的补丁构建器，须与对应 runner 模板集成；公开目录不附采集目录、私有绝对路径配置或完整实验产物。当前单元回归可运行：

```bash
python -B -m unittest discover -s retrieval/runtime -p 'test_public_runtime.py'
```

保留英文模型 Prompt 与原工具接口，文档中的中文是说明，不是替换模型输入。修改源码后必须重新冻结部署清单，并做真实 Runtime 输入对齐测试；离线回归不证明医学语义或联网可用性。
