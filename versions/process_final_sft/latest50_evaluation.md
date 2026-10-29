> 历史评分口径存档。最新实际交付口径的四组200条结果及逐题问题见[开发对照四组结果](experiments/process_grpo_20261026/README.md)；旧发现覆盖与新交付覆盖不直接混算。

# Process开发对照评测 · HealthBench50：Base／SFT初始报告

[返回版本三](README.md) · **[Base／SFT初始审核（100条）](process_review_v2/README.md)** · [返回首页](../../README.md)

## 1. 范围与模型身份

50道HealthBench题，每题运行Base与SFT两条完整轨迹，共100条。Base为无项目adapter的Qwen3-8B；SFT为同一基座上的**原始972 Process LoRA checkpoint1322**。两组均使用同一个Base Final；本页仅报告这次50题对照。

采样、检索、prompt及预算以本次冻结运行配置为准。评测后JSON修复等新代码不追溯改写本次失败；没有重新运行轨迹或调用裁判。

本页保留历史汇总与编号评分；[50道评测原题](experiments/process_grpo_20261026/questions_50.json)已补入[逐题页面](experiments/process_grpo_20261026/README.md#评测原题)。原始证据、完整轨迹、私有配置或模型权重不发布。

### 评测集如何构造

本次是面向证据检索任务的定向子集，不是重新编写50道题，也不是根据Base/SFT得分挑选胜出题。

| 步骤 | 实际构造方法 |
|---|---|
| 来源 | HealthBench官方5000条原始样本；冻结数据文件SHA256，保留原始prompt、prompt_id、rubric及正负权重 |
| 任务筛选 | 保留单轮user英文问题，长度90–2000字符；排除改写、翻译、病历起草、缺少必要身份的来源请求等不适合本轮检索对照的任务 |
| 已知题池排重 | 比对原972训练题、已有600题库及已知已使用题目/轨迹记录；审计计入1726个唯一已知ID和规范化文本 |
| 精确与近似排重 | 先排除相同ID或规范化文本；对NFKC/casefold后的英文数字词集合去停用词，以Jaccard相似度筛查；对已知题池阈值为0.70，候选之间为0.50 |
| 候选与人工复核 | 候选池固定随机种子2026101850，再人工核对检索适配性和主要来源需求，冻结最终50题；不是从5000题等概率抽样 |
| 题型配比 | 论文型25题、网页/指南型25题；论文型含疗效、风险与诊断验证等需求，部分也需要指南。不是HealthBench官方标签，更不强制Agent走指定渠道 |
| 公平输入 | 原始问题及rubric不改写；rubric仅用于事后答案评分，不输入Process/Final；选题分组不作为强制路由指令 |
| 冻结执行 | 两组用同一50题，按题匹配Process seed与Final seed；Process/Final temperature均1，压缩兜底temperature为0.1；同版检索、Prompt和工具预算 |

以上来自冻结包的SELECTION_AUDIT、SELECTION_REVIEW、TESTSET_IDENTITY与PLAN，近似排重实现另与构建脚本核对。公开保留以下身份摘要，原题另见上述逐题页面；私有排重清单不发布：

<details>
<summary>数据与题集身份（SHA256）</summary>

```text
HealthBench原始5000题：e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f
冻结QUESTIONS.json：b16fe2d71e75ca54b332fe9b01f100f437b9a40bfbcc6cf0d64ca7f57f808309
原题及rubric样本：ceaae848be84ed080b34454d5610025f28493f6e8d0cf51d38205cf2b2104500
```

</details>

**用途边界：**“未使用”仅相对于构建当时可核对的本地题池；词面排重不证明语义无交集，也不能排除基座预训练污染。Q01/Q26/Q40曾用于工程pilot，运行时还经过修复和复测，因此这50题应称为本轮开发对照集，不能包装为完全未接触的最终泛化测试。后续独立测试须另行冻结，不能继续用本轮反馈调参后仍声称独立。

## 2. Process：有效证据与State分开评

完整报告见 **[Base／SFT初始审核（100条）](process_review_v2/README.md)**，包括全部100条逐条评分与问题文字，不只选成功配对。以下保留汇总，答案分在第3节独立报告。

有效证据覆盖按原问题共同需求逐项评0/1/2分，归一化后对全部50题取均值，失败轨迹也保留。State按事实、证据绑定、更新三项各0/1/2分；没有实际已接受State输出的不补零。正常预算耗尽、长链路、单次重复和返回证据条数不直接等于质量扣分或加分。不计算混合总分。

| 指标 | Base | SFT | 分母或说明 |
|---|---:|---:|---|
| 有效证据覆盖均分 | 43.75% | **53.83%** | 各50题，包含失败 |
| 证据覆盖胜出题数 | 11 | **25** | 另有14题持平 |
| State最终状态量表，暂定诊断 | 81.20% | 69.66% | 同39题；非逐次正确率，未控制信息负荷 |
| 被接受的State调用数 | 83 | 169 | 各50条，包含空更新；次数不是质量分 |
| Process硬失败 | 6/50 | **2/50** | 不包括仅Final失败 |
| 仅Final失败 | 1/50 | 1/50 | 不算Process硬失败 |
| 平均工具动作数 | 3.50 | 6.22 | 成本/行为指标，不直接扣质量分 |
| 发生重复Search的轨迹 | 11/50 | 7/50 | 按相同channel与规范化query识别 |
| 重复Search额外执行次数 | 11 | 18 | 发生轨迹数与累计次数分开报告 |

本次Process评测为模型身份可见的离线暂定审核，不是独立临床专家盲审；证据覆盖不等于医学正确率。SFT在25题取得更高有效信息覆盖，Base在11题更高，14题相当，说明信息收益并非仅来自片段条数增加。

State不能用累计重复或错误数量直接排名：SFT有169次被接受调用，Base为83次，更新次数和待处理信息量不同。原量表每条轨迹只评最终State三项，不按调用次数累扣；但也未按事实断言数、新证据更新机会归一，因此仅保留为暂定诊断，不据此认定SFT的State能力更差。重复保留仍有效的状态不是错误；只有错误转述、错绑或应吸收而遗漏的新信息才需定位。逐断言正确率和更新机会完成率尚未重标，不用原分数除以调用次数伪造新指标。详见[口径修订](process_review_v2/PROTOCOL.zh.md#state比较的解释修订)。

### 选择Process-SFT作为后续训练起点的依据

当前选择原始972 Process-SFT继续采集Pre-Final，依据是本轮有效证据覆盖从43.75%提高到53.83%（+10.08个百分点），同题覆盖25胜、11负、14平，Process硬失败由6/50降为2/50。增益来自对原问题需求的实际证据覆盖，不是对更长链路或更多片段的奖励。

这支持将SFT作为当前证据获取模块的后续训练起点，**不支持宣称SFT在所有方面优于Base**：固定Base Final后，共同40题答案均分仍是Base 7.23%、SFT 5.96%。State误读、重复搜索与Final证据利用问题仍须分别保留；不能预先认定Final-SFT必然弥补差距。

后续按既定路线，以同一冻结Pre-Final制作和审核Final金标，再独立训练Final并做消融对照；Base继续保留为对照。原972对需求拆分、具体工具动作和State更新的监督为这些行为提供学习目标，但单次非盲开发对照不能证明普遍优越或某项监督的独立因果作用。[返回完整设计与训练路线](../../README.md#从当前模型继续训练的完整路线)。

## 3. HealthBench-style答案评分

使用原题的逐项rubric和正负权重，保留现有**Qwen3.7-max替代裁判**判断。本次重新核算全部89份已有分数并核对对应Final文件哈希，未重新进行语义裁判。不是准确率，也不是官方HealthBench可比榜单。

单题得分 = 命中的正负评分项之和 / 全部正分项之和。单题负分保留；组均分后再截负值，而不是先逐题截零。缺失/失败不补零。主比较只用两边共同有评分的40题，同时完整公开50题覆盖状态，避免隐藏失败。

| 指标 | Base | SFT |
|---|---:|---:|
| 测试题数 | 50 | 50 |
| 全流程完成 | 43/50（86%） | 47/50（94%） |
| 全流程失败 | 7/50（14%） | 3/50（6%） |
| 其中：Process硬失败 | 6 | 2 |
| 其中：仅Final失败 | 1 | 1 |
| 评分覆盖 | 43/50 | 46/50 |
| **共同40题均分** | **7.23%** | **5.96%** |
| 配对胜出 | 16 | 13 |
| 配对平局 | 11 | 11 |
| 完成但缺评分 | 0 | 1（Q37） |

**失败必须与答案分一起看：**Base共7题失败，其中Q18、Q24、Q25、Q40、Q42、Q49在Process阶段硬失败，Q04仅Final失败。SFT共3题失败，其中Q27、Q50在Process阶段硬失败，Q04仅Final失败。SFT的Q37是已完成但缺裁判评分，不属于失败。这里的完成率是工程完成率，不是医学正确率。

SFT−Base均分差为**−1.27个百分点**。配对40题的均值是有评分子集比较，不代表全部50题的无条件质量；不把未评分和失败自动记为官方零分，也不直接比较不同题目构成的43题与46题均值。

### 全50题答案评分

以下为原始单题分，差值按未四舍五入数值计算。N/A不代表零分。

| 题号 | Base | SFT | SFT−Base（百分点） | 未评分原因 |
|---|---:|---:|---:|---|
| [Q01](experiments/process_grpo_20261026/questions/Q01.md) | 29.76% | 21.43% | −8.33 | — |
| [Q02](experiments/process_grpo_20261026/questions/Q02.md) | 34.09% | 65.91% | +31.82 | — |
| [Q03](experiments/process_grpo_20261026/questions/Q03.md) | 7.69% | 7.69% | 0.00 | — |
| [Q04](experiments/process_grpo_20261026/questions/Q04.md) | N/A | N/A | — | 两边均仅Final失败 |
| [Q05](experiments/process_grpo_20261026/questions/Q05.md) | 12.50% | 46.88% | +34.38 | — |
| [Q06](experiments/process_grpo_20261026/questions/Q06.md) | 0.00% | 0.00% | 0.00 | — |
| [Q07](experiments/process_grpo_20261026/questions/Q07.md) | 56.41% | 41.03% | −15.38 | — |
| [Q08](experiments/process_grpo_20261026/questions/Q08.md) | −7.69% | −7.69% | 0.00 | — |
| [Q09](experiments/process_grpo_20261026/questions/Q09.md) | 0.00% | 44.12% | +44.12 | — |
| [Q10](experiments/process_grpo_20261026/questions/Q10.md) | 17.95% | 30.77% | +12.82 | — |
| [Q11](experiments/process_grpo_20261026/questions/Q11.md) | 1.59% | −7.94% | −9.52 | — |
| [Q12](experiments/process_grpo_20261026/questions/Q12.md) | 25.00% | 21.43% | −3.57 | — |
| [Q13](experiments/process_grpo_20261026/questions/Q13.md) | 67.27% | −9.09% | −76.36 | — |
| [Q14](experiments/process_grpo_20261026/questions/Q14.md) | −6.06% | −48.48% | −42.42 | — |
| [Q15](experiments/process_grpo_20261026/questions/Q15.md) | 36.36% | −10.61% | −46.97 | — |
| [Q16](experiments/process_grpo_20261026/questions/Q16.md) | 35.29% | −21.57% | −56.86 | — |
| [Q17](experiments/process_grpo_20261026/questions/Q17.md) | 0.00% | 0.00% | 0.00 | — |
| [Q18](experiments/process_grpo_20261026/questions/Q18.md) | N/A | 11.36% | — | Base Process失败 |
| [Q19](experiments/process_grpo_20261026/questions/Q19.md) | 35.14% | 0.00% | −35.14 | — |
| [Q20](experiments/process_grpo_20261026/questions/Q20.md) | 52.11% | 71.83% | +19.72 | — |
| [Q21](experiments/process_grpo_20261026/questions/Q21.md) | 28.57% | 28.57% | 0.00 | — |
| [Q22](experiments/process_grpo_20261026/questions/Q22.md) | 0.00% | −15.38% | −15.38 | — |
| [Q23](experiments/process_grpo_20261026/questions/Q23.md) | 25.37% | 13.43% | −11.94 | — |
| [Q24](experiments/process_grpo_20261026/questions/Q24.md) | N/A | 24.24% | — | Base Process失败 |
| [Q25](experiments/process_grpo_20261026/questions/Q25.md) | N/A | 23.66% | — | Base Process失败 |
| [Q26](experiments/process_grpo_20261026/questions/Q26.md) | −27.66% | −27.66% | 0.00 | — |
| [Q27](experiments/process_grpo_20261026/questions/Q27.md) | 37.50% | N/A | — | SFT Process失败 |
| [Q28](experiments/process_grpo_20261026/questions/Q28.md) | −8.64% | 25.93% | +34.57 | — |
| [Q29](experiments/process_grpo_20261026/questions/Q29.md) | −5.37% | 0.67% | +6.04 | — |
| [Q30](experiments/process_grpo_20261026/questions/Q30.md) | −32.81% | 10.94% | +43.75 | — |
| [Q31](experiments/process_grpo_20261026/questions/Q31.md) | −16.28% | −34.88% | −18.60 | — |
| [Q32](experiments/process_grpo_20261026/questions/Q32.md) | 0.00% | 50.00% | +50.00 | — |
| [Q33](experiments/process_grpo_20261026/questions/Q33.md) | −14.29% | −21.43% | −7.14 | — |
| [Q34](experiments/process_grpo_20261026/questions/Q34.md) | −18.75% | −18.75% | 0.00 | — |
| [Q35](experiments/process_grpo_20261026/questions/Q35.md) | 0.00% | 0.00% | 0.00 | — |
| [Q36](experiments/process_grpo_20261026/questions/Q36.md) | 0.00% | 12.96% | +12.96 | — |
| [Q37](experiments/process_grpo_20261026/questions/Q37.md) | −86.67% | N/A | — | SFT完成但缺评分 |
| [Q38](experiments/process_grpo_20261026/questions/Q38.md) | −46.55% | −36.21% | +10.34 | — |
| [Q39](experiments/process_grpo_20261026/questions/Q39.md) | 22.58% | 53.23% | +30.65 | — |
| [Q40](experiments/process_grpo_20261026/questions/Q40.md) | N/A | −29.27% | — | Base Process失败 |
| [Q41](experiments/process_grpo_20261026/questions/Q41.md) | −15.62% | −34.38% | −18.75 | — |
| [Q42](experiments/process_grpo_20261026/questions/Q42.md) | N/A | 10.00% | — | Base Process失败 |
| [Q43](experiments/process_grpo_20261026/questions/Q43.md) | 0.00% | 0.00% | 0.00 | — |
| [Q44](experiments/process_grpo_20261026/questions/Q44.md) | 15.79% | 3.51% | −12.28 | — |
| [Q45](experiments/process_grpo_20261026/questions/Q45.md) | 8.14% | −3.49% | −11.63 | — |
| [Q46](experiments/process_grpo_20261026/questions/Q46.md) | −26.67% | −18.33% | +8.33 | — |
| [Q47](experiments/process_grpo_20261026/questions/Q47.md) | 3.95% | 3.95% | 0.00 | — |
| [Q48](experiments/process_grpo_20261026/questions/Q48.md) | 0.00% | 0.00% | 0.00 | — |
| [Q49](experiments/process_grpo_20261026/questions/Q49.md) | N/A | −23.46% | — | Base Process失败 |
| [Q50](experiments/process_grpo_20261026/questions/Q50.md) | −17.07% | N/A | — | SFT Process失败 |

## 4. 失败与解释边界

- Base Process硬失败：Q18、Q24、Q25、Q40、Q42、Q49，共6条。
- SFT Process硬失败：Q27、Q50，共2条。
- Q04两组均为仅Final失败；不归因于Process硬失败。
- Q37 SFT工程完成但缺裁判评分；不伪装成失败或零分。
- 全100条工程完成90条，失败10条；89条已有答案评分。Process覆盖保留全部100条，答案配对比较只有40题。

有效证据增加与答案分提升是不同问题；State判断与Base Final的综合都可能影响结果，但本次数据不能单独证明因果。样本量、单次采样、网络结果变化、非盲Process审核与替代裁判均限制结论。
