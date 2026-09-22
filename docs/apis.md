# 外部 API、用途与配置

公开仓库不包含任何凭证。先复制 `.env.example` 为本地 `.env`，只填写实际启用的服务；`.env`、`*.env`、`secrets*` 和本地凭证目录均被 `.gitignore` 排除。代码和日志只应检查“是否配置”，不得输出密钥值。

## 按运行模式区分

| 运行模式 | 是否需要外部 API |
|---|---|
| `python -B run_pipeline.py demo/check` | 不需要；离线 fixture 不加载模型、不联网、不调用 Judge |
| 本地 Qwen3-8B / LoRA 推理 | 不需要云端 LLM API；需要本地模型与显卡环境 |
| PubMed/PMC 检索 | NCBI Key 可选；无 Key 仍可访问公开 E-utilities，但速率更低 |
| Semantic Scholar 检索 | S2 Key 可选；无 Key 时受匿名限流约束 |
| 通用网页检索 | 当前 Serper 路由需要 Serper Key |
| PDF 正文解析 | 仅启用 PDF 路由时需要 MinerU 云端 Token 或自托管 MinerU 地址 |
| 分阶段语义评分 | 需要一个 OpenAI-compatible HTTPS Judge endpoint 与 Key |
| ChatGPT Pro 独立复核 | 是采集后的人工/产品侧审计，不是代码运行依赖，也不读取本仓库密钥 |

## 服务清单

### 1. Serper：通用网页候选发现

- 环境变量：`SERPER_API_KEY`；可选轮换键 `SERPER_API_KEY_FALLBACK`、`SERPER_API_KEY_FALLBACK_2` … `SERPER_API_KEY_FALLBACK_20`。
- 用途：`medical_web_search` 的通用网页候选发现；返回候选 URL、标题和摘要，后续仍需 Browse 抓正文并建立 evidence provenance。
- 使用位置：`retrieval/dr_agent/mcp_backend/apis/serper_apis.py`。
- 必需性：只有启用 Serper-backed web search 时必需；PubMed-only 或离线回放不需要。

### 2. PubMed / PMC / Europe PMC：医学论文检索与正文

- 环境变量：`NCBI_API_KEY`（可选）、`NCBI_EMAIL`、`NCBI_TOOL`。
- 用途：调用 NCBI E-utilities 完成 PubMed 搜索、摘要/元数据获取、关联记录查询，并通过 PMC/Europe PMC 获取可用全文 XML。
- 使用位置：`retrieval/dr_agent/mcp_backend/apis/pubmed_apis.py`。
- 必需性：公开接口可无 Key 使用；批量采集建议配置 NCBI Key、真实联系邮箱和工具名并遵守速率限制。
- 注意：代码实际读取的是 `NCBI_API_KEY`，不是 `PUBMED_API_KEY`。

### 3. Semantic Scholar：论文元数据与候选补充

- 环境变量：`S2_API_KEY`（可选）。
- 用途：论文搜索、元数据/引用关系补充，以及可选的 PubMed 候选 enrichment。
- 使用位置：`retrieval/dr_agent/mcp_backend/apis/semantic_scholar_apis.py`；跨进程限速与重试位于 `s2_transport.py`。
- 必需性：官方 API 可匿名访问，但 Key 能提供更稳定的额度；是否启用 enrichment 由检索配置决定。
- 注意：代码实际读取的是 `S2_API_KEY`，不是 `SEMANTIC_SCHOLAR_API_KEY`。

### 4. MinerU：PDF 正文结构化解析

- 云端方式：`MINERU_API_TOKEN`。
- 自托管方式：`MEDGAP_MINERU_BASE_URL`，服务需提供 `/file_parse` 及任务结果接口。
- 开关：`MEDGAP_PAPER_PDF_ENABLED=1`；不开启时不会因为存在 Token 就自动把所有来源送去 PDF 解析。
- 用途：当 HTML/XML 正文不可得且来源为 PDF 时，将 PDF 转成 Markdown/结构化正文，再进入统一的清洗、切分和证据选择流程。
- 使用位置：`mineru_cloud.py` / `mineru_cloud_legacy_v54.py`（云端）与 `mineru_client.py`（自托管），策略入口为 `pdf_policy_v54.py`。
- 必需性：可选。HTML、PubMed 摘要和 PMC XML 路径不依赖 MinerU。

### 5. 分阶段 Judge API：轨迹评分与 Evidence Gain

- 环境变量约定：`JUDGE_BASE_URL`、`JUDGE_MODEL`、`JUDGE_API_KEY`。
- 接口要求：OpenAI-compatible `POST /chat/completions`、HTTPS、JSON object 输出。
- 用途：轨迹采集完成后，对 Checklist、Search、Browse、State、Stop，以及 Final completeness / fidelity / citation 分阶段评分；Evidence Gain 使用冻结 before receipt 与新增 evidence 产生增量回执。
- 使用位置：Judge schema/prompt 在 `judge/tiered_allstages_v1/`；通用 HTTPS transport 与严格响应校验在 `training/judge.py::openai_transport`。实际批处理启动层把上述变量传入 transport。
- 必需性：真实语义评分需要；轨迹采集、离线合同测试、本地模型推理和已经缓存且通过身份校验的回放不需要再次调用。
- 边界：Judge 原始输出不能直接等同训练奖励，仍需通过 schema/binding、protocol eligibility、Evidence Gain、cost/duplicate 和 Stop authority 等 reward compiler 门控。

### 6. MedGap semantic verifier：证据与医学语义校验

- 环境变量：`MEDGAP_VERIFIER_BASE_URL`、`MEDGAP_VERIFIER_MODEL`、`MEDGAP_VERIFIER_API_KEY_ENV`。
- 默认 Key 变量：`DASHSCOPE_API_KEY`；可通过 `MEDGAP_VERIFIER_API_KEY_ENV` 改成其他本地变量名。
- 用途：对 evidence chunk / Final 做严格结构化医学语义核验和缓存；它与分阶段 Judge 是不同合同，不能混用回执。
- 使用位置：`retrieval/dr_agent/medgap/verifier.py` 与 `service.py`。
- 必需性：可选；只在启用 MedGap verifier 的运行配置中调用。

### 7. 可选网页与证据选择后端

| 能力 | 环境变量 | 无配置时行为 | 代码位置 |
|---|---|---|---|
| Jina Reader | `JINA_API_KEY` | 不使用该抓取后端 | `retrieval/.../apis/jina_apis.py` |
| Crawl4AI 服务 | `CRAWL4AI_API_URL`、`CRAWL4AI_API_KEY`、`CRAWL4AI_BLOCKLIST_PATH` | 不使用 Docker crawler 路径 | `retrieval/.../apis/crawl4ai_docker_api.py` |
| Semantic evidence reader | `MEDGAP_EVIDENCE_READER_BASE_URL`、`MEDGAP_EVIDENCE_READER_MODEL`、`MEDGAP_EVIDENCE_READER_API_KEY` | 确定性回退到上游排序 | `retrieval/.../apis/evidence_reader_v28.py` |

## 不属于外部 API 的组件

- Qwen3-8B 与 LoRA：本地加载，不调用云端生成 API。
- BGE embedding、MiniLM / MedCPT reranker：本地模型权重；下载方式与授权由部署者自行处理。
- BM25、正文清洗、chunk 切分、表格完整性和 citation ID 校验：本地确定性代码。
- ChatGPT Pro 50 题复核：作为独立语义审计结果公开，不是训练 Runtime 内的 API 调用。

## 安全规则

1. 不把真实 `.env`、Key、Authorization header、完整请求头或 Secret 扫描结果提交到 Git。
2. 日志只记录服务名、请求身份摘要、状态码、重试次数和缓存命中，不记录 Key。
3. Judge、Verifier 和 Evidence Reader 使用独立变量与独立缓存身份，避免不同评分合同相互污染。
4. 外部 API 不可用时必须显式失败、进入可验证 fallback 或标记 pending；不得把“未评分”偷偷改写成 0 分。
