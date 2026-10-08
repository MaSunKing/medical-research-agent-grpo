# 医学研究智能体 · Medical Research Agent

面向医学文献与网页资料的证据检索智能体。系统把问题拆成可独立核验的需求，围绕当前证据缺口动态搜索、读取来源、更新状态，再把可追溯的证据交给答案生成环节。

项目关注的不只是“搜到一个相关标题”，而是 **读到了什么、解决了哪个需求、还有什么没有证实**。用于研究，不替代临床判断。

## 整体流程

```mermaid
flowchart LR
    Q[原始问题] --> C[Checklist：独立需求与原问题约束]
    C --> D[Decision：结合 State 与历史选择动作]
    D --> S[Search：动态生成针对缺口的 query]
    S --> P[累计候选池与可见窗口]
    P --> B[Browse：选择来源与阅读重点]
    D --> B
    B --> E[清洗正文、选择片段、保留证据坐标]
    E --> U[State：评估证据覆盖与未解决缺口]
    U --> D
    D --> T[Stop：收益、缺口与预算]
    T --> F[Pre-Final：当前状态、证据与历史]
    F --> A[Final：基于可见证据回答并引用]
```

## 两条检索链路

论文和网页共用工具协议、候选池与证据交接，但使用适合各自内容的检索和正文处理路径。

| 环节 | 论文 | 网页 |
|---|---|---|
| 来源发现 | PubMed 与 Semantic Scholar | Serper 网页搜索 |
| Search 返回内容 | 标题、摘要及来源元数据；摘要不等于全文 | 标题、snippet 与 URL；预览不等于已读正文 |
| Search 语义排序 | MedCPT Cross-Encoder | MiniLM Cross-Encoder |
| 排序关注点 | 原问题 0.7 + 当前 query 0.3 | 原问题 0.7 + 当前 query 0.3；权威性不覆盖相关性 |
| Browse | 按可用来源读取摘要、正文或 PDF；访问失败有明确回执 | 访问探针、网页正文提取与结构清洗 |
| 片段选择 | 医学片段检索、语义重排、表格结构保留 | 正文清洗后混合召回与语义重排 |

语义模型不可用时，Search 模块保留原有可用排序并记录降级，不把模型加载失败伪装成“没有相关证据”。PDF 解析属于可选正文路径，不承诺每篇论文都能取得全文。

```mermaid
flowchart TB
    Query[原问题 + 当前缺口 query] --> Paper[PubMed / Semantic Scholar]
    Query --> Web[Serper]
    Paper --> PR[标题与摘要：MedCPT 重排]
    Web --> WR[标题与 snippet：MiniLM 重排]
    PR --> Pool[累计候选池]
    WR --> Pool
    Pool --> Window[最新候选优先 + 历史候选补充]
    Window --> Read[Browse 已列出的来源 ID]
    Read --> Clean[正文清洗 / 结构解析 / PDF 可选]
    Clean --> Chunk[片段召回与重排]
    Chunk --> Evidence[精确文本、chunk ID、来源与坐标]
```

## 候选池：保留历史，不覆盖最新结果

当前可见窗口最多 **8 个候选**。首次搜索沿 Search 后端的返回顺序填充；之后最多先放 4 个最新候选，剩余位置从历史池按原问题的 MiniLM 相关性选择。历史不足时，用更多最新候选补齐。

**最新候选在前，历史候选在后，不再进行合并后的全局重排。** 因此 Search 已完成的语义排序不会被第二次混合排序打乱。候选池保存累计来源，可见窗口只是当前给模型看的投影；记录包含分组、显示顺序和来源 ID。

## Checklist、State 与动态 Query

- Checklist 从原问题提取可独立回答的需求，保留人群、疾病、干预、时间等共享约束；不会机械地把所有关键词拆成任务。
- Decision 看原问题、当前 State、历史和候选，针对具体缺口生成 query。论文允许医学关键词或 Boolean 检索式，网页允许聚焦短语；不强制英文完整句子。
- Prompt 要求不引入无关疾病、药物、年份或研究类型；这些是模型指导，不是后端已经证明的语义硬约束。
- Browse 只允许使用当前列出的来源 ID。标题相关不等于证据支持；导航、登录、广告和访问错误不能用于提升覆盖状态。
- State 依据已读证据更新 `unknown / partial / direct` 与证据 ID。`direct` 是模型评估，仍可因证据不足或冲突而修订。

工具接口保持不变：没有为了 query 聚焦新增必填的 requirement-ID 字段。英文 Prompt 源码见 [Decision](retrieval/runtime/sft_interface_v20.py)、[Checklist](retrieval/runtime/checklist_feedback_v55.py) 和 [State](retrieval/runtime/state_compact_v38.py)。

## 历史、预算和失败处理

历史是固定预算下的工作记忆，保存已发生的动作、工具结果摘要和状态变化；证据正文单独交接。历史不能引用未来阶段，也不能把失败尝试标成成功。最终状态以最新有效 State 为准。

当前阶段合同：总上下文 **10,240 tokens**；Checklist、State 各预留 1,200，Decision/Stop 各 240，Final 2,400。不会为了塞入历史单独放宽某一阶段。

工具预算按执行回执结算：**6 次计费动作 + 最多 3 次环境失败豁免，最多 9 次真实工具执行**。空搜索、无相关内容和非法参数不是环境失败豁免。3 次豁免用尽不直接强制 Stop，后续动作继续按正常预算结算。Browse 同时遵守来源黑名单、冷却与失败重试限制。

## 证据交接与引用

Pre-Final 不是另写一份自由摘要，而是按实际 Final 输入合同导出：问题、最新 State、Checklist freshness、已打开证据、来源头信息，以及共享输入构建器实际保留的历史。原始轨迹与输入包分别保存，便于复查。

证据 freshness 用 **精确 chunk ID 与文本** 计算，避免仅因展示元数据变化误判状态过期。Evidence Card 是长证据的预算内交接机制，不替代原始出处。

当前独立 Final 接口使用短引用别名，并保留到原始 chunk ID 的确定性映射；输出为单个 `<answer>...</answer>`。引用存在、格式正确与“该句确被证据支持”分别检查，不能互相替代。见 [引用与 Final 合同](docs/final_contract.md)。

## 源码与阅读入口

| 目录 / 文档 | 内容 |
|---|---|
| [retrieval/](retrieval/README.md) | 当前论文、网页、PDF、清洗与语义排序源码 |
| [retrieval/runtime/](retrieval/runtime/README.md) | 当前候选窗口、英文 Prompt、失败预算与证据交接组件 |
| [系统架构](docs/architecture.md) | 模块边界与证据生命周期 |
| [检索设计](docs/retrieval.md) | Search / Browse 与候选池细节 |
| [代码导航](docs/code_navigation.md) | 按功能定位实际源码 |
| [API 配置](docs/apis.md) | 服务用途与变量名；不含真实密钥 |

## 项目迭代

首页描述当前检索与工程机制；实验方法、参数和结果按版本分开维护，不混用旧检索与新检索的实验结论。

| 版本 | 独立说明 | 检索代码 |
|---|---|---|
| 单 LoRA | [保留原项目说明与历史实验](versions/single_lora/README.md) | [冻结的旧检索](versions/legacy_retrieval/README.md) |
| 双 LoRA | [972 题复用、角色拆分与 50 题验证框架](versions/dual_lora/README.md) | 与单 LoRA 共用冻结旧检索 |
| Process-SFT + Final-SFT | [700 题 Process、Pre-Final、独立 Final 与消融框架](versions/process_final_sft/README.md) | 当前检索 |

版本目录明确区分已经执行的训练与待执行的研究计划。公开内容仅包含代码、合同、配置和方法说明，不包含私有训练题、完整采集记录、金标答案、模型权重或凭据。

Apache-2.0；必要署名见 [第三方声明](THIRD_PARTY_NOTICES.md)。模型与外部服务另受各自条款约束。
