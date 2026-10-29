# 1. 数据与实验设置

[版本三地图](README.md) · [完整架构](engineering.md) · [评测地图](evaluation.md) · [术语](glossary.md)

本页集中记录模型、量化、题目来源、运行配置及数据隔离。单题节点与工具循环由[系统架构与执行机制](engineering.md)说明，训练目标和结果分别由训练章节与各评测报告承接。

## 研究目标与模型角色

目标是让8B模型的检索与证据整理策略（Process）围绕问题缺口自主选择搜索、打开来源并整理可追溯证据，再交给答案生成模块（Final）。答案模块读取原问题与实际交付材料，综合回答、引用来源并说明证据不足之处。工程合同保证数据和动作可核验；医学支持程度仍由模型判断，并在评测中检查。

基座为`Qwen/Qwen3-8B`，通过bitsandbytes以4-bit NF4双重量化加载，BF16计算。Base指同一基座禁用本项目adapter，不是另一份名为Qwen3-8B-Base的模型或未量化对照。

| 角色 | 输入与职责 | 参数状态 |
|---|---|---|
| Base Process | 原题与当步上下文，生成需求、决策、State | 禁用项目adapter |
| Process-SFT / GRPO | 同一Process协议，策略不同 | 分别加载原972或GRPO adapter |
| 基座答案生成器（Base Final） | 原题、实际交付的证据和State，综合答案 | 同一Qwen3-8B禁用Process adapter，答案模块参数不更新 |
| 答案模块SFT（Final-SFT） | 相同答案输入合同，学习gold答案 | 另建答案生成LoRA；本轮GRPO不采用 |
| 压缩兜底 | 预算压力下处理已给定材料 | 同基座禁用Process adapter，不自主检索 |

检索策略与答案生成可在同一基座上分别控制参数；“独立答案模块”不表示另部署一个更大的模型。Process-GRPO采集及跨基准四组验证使用冻结FP32 Process线性输出头，生成答案时恢复原输出头并禁用Process adapter。开发对照中Base/SFT与GRPO的数值路径差异另见[运行配置对照](experiments/process_grpo_20261026/runtime.md)。

“答案模块参数固定”表示本轮不更新这些权重；它仍会根据各组交付的不同材料生成不同回答。“固定输入对照”则让各答案候选读取同一份请求快照，用于隔离答案模块本身的差异。

## 模型加载配置

| 模型/量化项 | 当前具体配置 |
|---|---|
| 基座模型 | `Qwen/Qwen3-8B` |
| 量化实现 | Transformers `BitsAndBytesConfig` / bitsandbytes |
| 基座权重 | `load_in_4bit=True`，`bnb_4bit_quant_type="nf4"` |
| 双重量化 | `bnb_4bit_use_double_quant=True` |
| 量化计算类型 | `bnb_4bit_compute_dtype=torch.bfloat16` |
| Process训练 | NF4 QLoRA；量化基座冻结，FP32 LoRA参与更新 |

Process训练采用NF4 QLoRA，量化基座冻结、LoRA参数保留FP32。部署的非thinking、采样、数值路径及源码身份分别登记，不能只凭模型名称视为完全相同配置；历史配置中的raw键名保留原身份。

## 题集分别用于什么

| 题集 | 来源与规模 | 用途 |
|---|---|---|
| Process972 | 原始Teacher真实阶段轨迹，972题 | Process-SFT；10575个合法阶段目标 |
| Final600 | 论文/健康科普问答来源题池，固定500 train＋100 dev | 固定答案输入快照、gold构建与答案模块SFT |
| GRPO100 | Final train500中的50道短题＋LiveQA历史Train-1的50道长题 | 每题4条，采集400条；99组用于本轮更新 |
| Process开发对照题集（HealthBench50） | 原始HealthBench题池，25论文需求＋25网页/指南需求 | Base/SFT/GRPO开发对照，不参与本轮GRPO更新 |
| 跨基准验证题集 | HealthBench50＋MedBrowseComp50，区别于开发对照的50题子集 | 四种Process策略同配置重新检索，各100条；不参与本轮GRPO更新 |

Final600的来源标签、原始问题和split保存在私有身份清单；gold采用指定版本，其他老师候选只留作记录。不能把Gold标记解释成独立临床认证。

### Process开发对照题集怎样构造

从HealthBench原始5000题筛选可独立回答的单轮英文医学检索问题，保留原题和rubric。对原972、已有题库及已知使用记录检查ID和规范化文本；英文数字词集合的Jaccard近似筛查阈值对已知池0.70、候选之间0.50。固定候选种子2026101850，再人工复核检索适配性，冻结25道论文需求和25道网页/指南需求题。

这种分层不是HealthBench官方标签，也不强制Agent只能用一条渠道。题集经过工程pilot与开发诊断，不是原5000题的等概率样本；评分rubric不输入模型。选题身份与更细历史记录见[原选题说明](latest50_evaluation.md#评测集如何构造)。

### 数据隔离的结论与边界

Final500与dev100题目ID不交叉；GRPO100不使用dev100或HealthBench题。GRPO100与Process开发对照50题ID、规范化文本交集均为0，因此没有直接用Process开发对照50题进行本轮GRPO更新。

Process开发对照50题仍已用于开发、规则讨论和checkpoint选择，不能称为独立未见测试。ID/文本排重不证明语义完全无关，也不能排除基座预训练污染。跨基准验证评测已完成，见[跨基准验证设置](experiments/newbench_20261028/setup.md)；下一轮训练未启动。

## 运行配置

以下为当前节点式运行的配置合同；原972训练仍使用其10,240-token合同，具体见[Process-SFT](process_sft.md)。实验复现以对应冻结执行配置为准。

| 项目 | 运行预算 |
|---|---|
| 完整请求 | soft12288、hard15000 tokens；输入、输出预留和修正余量共同计入 |
| 预留 | repair384、safety128 |
| 阶段输出 | Checklist1200、Decision400、State1200、Final2400 tokens |
| 工具 | 正常动作6次；已识别环境失败最多5次豁免；actual最多11次 |
| 可恢复决策拒绝 | 最多6次，与工具预算分开 |
| 可见候选 / Browse | 最多8个候选；单次最多3个片段 |
| 压缩 | 确定性去冗余优先；必要时T=0.1，独立请求预算10240，最多12次调用 |
| 生成 | 非thinking；T=1、top_p=1、top_k=0；按题与调用记录种子 |

保守JSON修正只处理可确定的语法，如代码围栏、尾逗号或字符串内控制空白；不猜医学内容、不重写字段含义、不放宽ID/范围校验。原始输出与修正记录都保留。

## 跨基准验证评测的实现身份

HealthBench50与MedBrowseComp50的四组均重新运行，统一FP32 Process头、同预算与同检索，答案模块共用参数不更新的基座答案生成器。部署源代码按单独快照公开在[跨基准验证评测代码](experiments/newbench_20261028/code/README.md)，不以旧600题源码冒充本轮运行。工程缓存防止重复网络调用，动作预算限制空转；是否提出重复查询仍是策略行为。

<a id="实际交付与完整存档"></a>

## 实验身份与公开范围

PreFinal完整材料归档、实际答案请求与输入快照分别登记身份。交付覆盖只依据真正进入答案请求的材料；固定输入对照复用同一份请求快照，不用更大存档补分。[交接边界](engineering.md#handoff)

程序结束、格式合格与哈希通过属于工程完整性；证据适用、State忠实与答案正确属于内容质量。完整批量数据、gold、权重与凭据留私有，公开源码与真实案例范围见[源码入口](current/README.md)及[案例说明](../../examples/version3_cases/README.md)。

<a id="单题如何运行"></a>
<a id="search与browse的区别"></a>
<a id="检索文本与证据正文使用不同预算"></a>
<a id="有界清洗与来源连续性"></a>
<a id="上下文工具与语法合同"></a>

运行流程、Search/Browse、阅读重点、正文保护与来源恢复统一见[完整架构](engineering.md)。
