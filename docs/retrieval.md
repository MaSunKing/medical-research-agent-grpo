# 当前检索设计

[工程架构](architecture.md) · [逐步工程设计](../versions/process_final_sft/engineering.md) · [首页](../README.md)

## Search发现来源，Browse建立证据

Search参数为channel、query、1–2个requirement_ids；标题/摘要/snippet是相关性线索，不是已打开证据。Browse选择当前可见来源字母与reading_focus，后者用于片段排序，不是来源ID。模型仍负责围绕原题缺口选动作，不强制读第一条。

论文并行查询PubMed与Semantic Scholar、合并去重，再按标题/摘要语义排序；网页使用Serper及标题/snippet排序。两路均为**原问题60% + 当前query40%**。论文路径采用MedCPT；网页本地Cross-Encoder以部署配置和实际加载记录为准。模型不可用时记录实际fallback，不伪装为语义重排或“医学无证据”。

Search解决“哪些来源值得打开”，Browse解决“该来源实际提供了什么”。先控制候选窗口，再按明确阅读重点取片段，可以减少无关正文反复进入上下文，也为模型保留换来源与换方向的选择。默认embedding为`BAAI/bge-small-en-v1.5`，本地重排采用配置的`ncbi/MedCPT-Cross-Encoder`；实际部署路径与加载身份仍须随运行登记。

## query、reading_focus与正文使用独立预算

`reading_focus`是本次Browse希望找到的内容，例如疗效、严重不良反应或具体人群结果。每次Browse独立传入并计量，可以沿用或更换重点，不会与上次拼接累计。模型负责把本次缺口写得简洁并保留关键限定。

| 限制层级 | 当前预算 | 目的 |
|---|---|---|
| 动作中的query / reading_focus字段 | 通用Text最多1200字符 | 约束动作结构及描述规模 |
| 实际检索focus投影 | 320个UTF-8字节，单次独立 | 保持当步检索意图集中，控制排序输入规模 |
| 原题检索锚点 | 640个UTF-8字节，独立于focus | 为当步检索提供原问题方向 |
| 模型阶段输出 | Checklist1200、Decision400、State1200、Final2400 tokens | 限制各阶段生成规模 |
| Browse返回内容 | 单次最多3片段，并受正文处理和完整请求token预算约束 | 把相关原文交付给后续节点 |

320/640字节只作用于检索投影，不是原题在模型输入中的总长度，也不是证据正文长度。完整原题继续由节点上下文携带。超过检索投影预算的末尾限定不会进入该投影，因此应优先写出实体、比较对象和本次缺口；字符数、UTF-8字节数与token数不能互换。

源码：[动作合同](../versions/process_final_sft/current/runtime/src/trainable_agent/contracts.py) · [检索锚点](../versions/process_final_sft/current/runtime/retrieval_snapshot_v2/backend/agent/dr_agent/anchored_query.py)

## 新查询窗口与重复查询缓存

```mermaid
flowchart TB
    Q[渠道和规范化query] --> H{本题同工具身份已有正常回执?}
    H -->|没有| B[真实搜索与60/40排序]
    B --> W[最新最多先4个，历史按原题补位]
    W --> P[可见最多8个，初次或历史不足由新来源补足]
    H -->|已有| C[原Search前8候选重显，不混其他query]
    P --> V[作用域过滤，未读优先、部分已读后置]
    C --> V
    V --> D[候选和cache/network反馈交给Decision]
    D --> R[模型自行选源和reading_focus]
```

最新/历史组组装后不做第三次语义全池重排。阅读状态稳定分组可能改变显示次序，但不重命名字母或删除来源历史。

相同channel及大小写/空白规范化后的相同query，不再因重复在Decision直接拒绝。Execute读取本题同工具身份的正常或空回执缓存，标记cache_hit=true、network_executed=false；计一次正常动作和actual动作，不产生新Serper搜索请求。只重显原Search前8候选，不混其他query、不联网补满8条。失败/未知结果不冒充正常缓存，缺可核验回执则失败。

模型可以打开其他候选、换query/渠道或停止；持续原样重复由正常工具预算自然结束。近似query提醒不是语义相同判定，不硬禁真正不同的查询。Serper凭据按主键、FALLBACK和已配置编号fallback的唯一键顺序尝试，密钥不入轨迹；与查询缓存机制分开。

缓存减少重复外部请求，预算约束重复动作的总成本；两者没有替策略选择下一步。在四组同配置评测中，Base、SFT与GRPO使用相同保护，是否仍提出完全相同query单独作为策略行为统计。结果与计数分母见[四组重复Search及问题记录](../versions/process_final_sft/experiments/newbench_20261028/README.md)。

## 同来源续读与可用性

同篇论文不等于相同chunk；Browse按焦点排序、排除已读窗口，可继续取得同来源新片段。摘要路线耗尽不表示全文已读完。

| 回执/条件 | 处理边界 |
|---|---|
| 明确阻断、不可重试来源故障 | 按可用性规则隐藏，原始记录不删 |
| 临时环境故障 | 最多一次显式重试机会，仍失败后不能原样无限重试 |
| 可核验PDF同任务续取 | 保留受限机会，不以本地超时断言远端停止 |
| no_new_passages | 按query、Browse需求、解析路线和内容身份作用域隐藏，不永久封整篇 |
| 不可重试source_unreadable | 相同搜索签名、需求和读取条件下隐藏；新条件可再评估 |
| 普通ok/empty但无新E | 不仅凭无新E断言耗尽，可提醒无进展；实际内容重读计正常预算 |

Search绑定R1，不意味着随后Browse绑定R2也按R1的耗尽作用域隐藏。可用性必须匹配实际动作与工具身份，不能连带误移除，也不能改别名或重启绕过原读取条件。

## 正文、排序与证据范围

HTML/XML/可用全文或PDF进入清洗、混合召回与片段排序。一次最多3片段，保护完整表格行、表头与已解析脚注；overlap可保留上下文，不能把相邻重叠全部删除。导航/目录与正文混合时只处理可确认边界，保护临床否定、数字、不确定性和来源范围。

清洗保留`template_heading_and_short_list`、`guideline_heading_and_topic_list`两条弱标题规则，用于减少目录和页面模板占用上下文。弱规则前先检查正文特征：事实陈述、否定或比较、带单位的数值/阈值、中文指令及可识别表格会得到保护，相邻指南导航分支使用同一保护，避免被后续分支再次排除。裸数字及逗号、分号、冒号不单独构成通行条件，明确版权、技术支持UI和订阅控制仍按原顺序过滤。

这种保守保护优先减少有内容短段落的误判，可能保留部分具有事实句式的混合目录；相关性仍由后续排序和内容评价判断。没有句末句号不等于没有有效信息，识别到正文形态也不证明其中结论正确。实现与回归见[正文保护](../versions/process_final_sft/engineering.md#6-正文解析与清洗减少模板噪声保护有内容的短段落)。

原始片段、来源范围、解析路线、chunk ID、坐标和哈希归档。精确片段去重与来源续读分开；抽取成功不保证适用，空搜不证明研究不存在。失败/无新E反馈回Decision，只有新增E触发State。

## 来源映射与worker恢复

来源显示字母、canonical `source_id`与已打开的E ID各有职责。Search注册来源映射，Browse引用已有来源，后续节点引用实际证据；显示顺序变化不重新命名来源。使用ID引用减少每次请求重复携带整份来源元数据。

worker启动时只读取自身output下已完成的Search操作回执，核验操作身份、结果哈希与非链接路径边界，再按完成顺序恢复WEB和论文映射。已完成操作与paired Search缓存重放也补注册；恢复不重新Search。WEB仍调用原URL/hash验证，论文恢复核对原生身份、别名和URL，保留canonical ID。

新鲜Search补注册时合并已有WEB完整metadata，保留公共返回投影没有携带的检索来源信息；论文保留原生有效字段，同时更新最近一次发现的provenance。重启只能恢复回执实际保存的字段，缺失映射不触发隐式联网补搜，损坏或不匹配回执拒绝恢复。

源码：[worker](../versions/process_final_sft/current/runtime/retrieval_snapshot_v2/worker.py) · [恢复组件](../versions/process_final_sft/current/runtime/retrieval_snapshot_v2/source_registry.py) · [论文Registry](../versions/process_final_sft/current/runtime/retrieval_snapshot_v2/backend/agent/dr_agent/mcp_backend/apis/dual_paper.py)

## 预算与交接

正常动作6、已识别环境失败豁免最多5、actual上限11（含缓存操作）；空搜、无关或普通无新增不自动豁免。动作不可用的可恢复拒绝另限6次；schema/ID/State错误不能全当环境问题吞掉。

完整请求软线12,288、硬线15,000，输出/修正预留均计入。先确定性去冗余，再用隔离模型原文抽取兜底，Search预览与Browse正文不混合压缩。原记录不变；无法安全容纳则明确失败。

生产模块与公开快照的区别见[代码导航](code_navigation.md)。历史版本及[后续评测计划](../versions/process_final_sft/evaluation.md)保持各自身份。
