# 跨基准验证评测 · HealthBench50＋MedBrowseComp50

**同一工程框架下，GRPO保持0次完全重复Search，并减少严重Process问题；GRPO25交付更多适用证据，GRPO13在HealthBench答案加权分上最高。** 这些优势分别由行为、内容和答案指标支持，不能合并为“所有指标都优于SFT”。

[评测方法与结果](../../evaluation.md) · [逐题索引](../../records.md#cross-benchmark) · [100道评测原题](questions_100.json) · [400条记录](records_400.json) · [复算代码](verify_results.py) · [名词与指标解释](../../glossary.md)

**阅读入口：** [怎样区分三类结果](accuracy.md#score-guide) · [100题原题、标准与四组判定](../../records.md#cross-benchmark)。主表先看答案正确率与信息齐全率；Process分项、原生Judge及配对统计在下方折叠区，不混成一个“总准确率”。

四组分别更换Process检索策略；**答案生成模块（Final）**统一使用基座答案生成器（Base Final），即同一Qwen3-8B禁用本项目adapter。答案模块参数与生成配置相同，但各组重新检索、交付的材料可以不同，最终答案也可以不同；这里固定的是参数和配置，不是答案文本。

## 核心答案准确率与必要信息

|指标|Base|Process-SFT|GRPO13|GRPO25|
|---|---:|---:|---:|---:|
|HealthBench核心答案正确／50|26（52%）|27（54%）|**30（60%）**|26（52%）|
|MedBrowseComp核心答案正确／50|14（28%）|**19（38%）**|16（32%）|**19（38%）**|
|HealthBench必要信息齐全／50|15（30%）|22（44%）|23（46%）|**25（50%）**|

**HB上，Base→SFT→GRPO的改进同时获得最终核心回答与信息交付支持：** GRPO13的答案正确率最高，GRPO25交付完整必要信息的题数最多。MB上GRPO25与SFT持平；不同checkpoint的优势分别报告。

核心答案按原题必须回答的结论／字段记0或1，失败保留为0，分母固定50；必要信息齐全只看实际Final输入。它们与下方原生rubric、参考匹配及Process分项的含义、分母不同。[评分标准、MB多跳要求与6次工具预算](accuracy.md) · [400条判断](accuracy_records_400.json) · [评分表CSV](accuracy_scores_400.csv) · [100份共同标准与来源](answer_requirements_100.json)。每题页均有四组具体判定和理由。

MB要求精确试验绑定，并常需跨FDA、专利或公司资料核实；四组均采用6次正常工具动作预算，访问故障最多豁免5次，Search／Browse共享预算。低分应结合这种有界多跳任务解释，预算贡献仍需对照验证，试验错绑和Final误读等实际错误完整保留。

<a id="diagnostic-scores"></a>

<details>
<summary>展开辅助诊断：Process质量、失败、原生Judge分与配对统计</summary>

## 一眼看到的变化

|跨基准验证指标|Base|Process-SFT|GRPO13|GRPO25|
|---|---:|---:|---:|---:|
|完全相同Search额外动作 / 全部Search|31 / 171|1 / 300|0 / 291|0 / 294|
|重大或Process失败，去重 / 100|21|17|14|14|
|HealthBench实际交付覆盖%|63.08|78.00|76.08|81.92|
|MedBrowseComp实际交付覆盖%|4.50|10.00|7.17|13.17|

GRPO13与GRPO25都比Base减少了严重Process问题；相对SFT从17题降至14题，是本批观察，不能据此宣称已经证明稳定下降。缓存/预算/候选可用性保护四组相同；0次完全重复反映策略行为，不能归因于只给GRPO开启工程保护。

## 重大错误与轨迹失败

|组别|HealthBench重大|MedBrowseComp重大|重大合计|Process失败|Final失败|轨迹失败合计 / 100|
|---|---:|---:|---:|---:|---:|---:|
|Base|1|14|15|6|0|6|
|Process-SFT|11|4|15|2|0|2|
|GRPO13|8|5|13|1|1|2|
|GRPO25|10|3|13|1|2|3|

轨迹失败按每条终态计一次，已修复JSON尝试不另算失败。本批工程失败0、缺失0。重大内容错误可出现在已完成轨迹，也可与Final失败重叠，不能把各列盲目相加。HealthBench的重大分布不支持GRPO/SFT每项都更低；GRPO25也有3条轨迹失败，多于SFT的2条。

## 逐题胜负，不用平均分替代

每格均为前者胜 / 前者负 / 平；本地按既定R比较全部100题，R已含重大折扣和Process失败0，Final-only失败不将Process置0。第二列只比较P1实际证据覆盖，不能当成最终答案正确率。

|前者 vs 后者|本地Process R：胜/负/平|实际交付P1：胜/负/平|
|---|---|---|
|Process-SFT vs Base|62 / 35 / 3|31 / 8 / 61|
|GRPO13 vs Base|58 / 36 / 6|30 / 11 / 59|
|GRPO25 vs Base|61 / 33 / 6|38 / 10 / 52|
|GRPO13 vs Process-SFT|32 / 45 / 23|16 / 20 / 64|
|GRPO25 vs Process-SFT|41 / 38 / 21|22 / 11 / 67|
|GRPO25 vs GRPO13|49 / 28 / 23|22 / 11 / 67|

**GRPO25对SFT的交付覆盖为22胜、11负、67平；Process R为41胜、38负、21平。** 前者显示内容交付方向改善，后者仅小幅胜多。GRPO13对SFT的R为32胜45负23平，因此不宣称GRPO13全面优于SFT。全部胜负题号见[可追溯计数](errors_and_wins.json)。

<a id="benchmark答案结果"></a>

## 原生rubric与参考匹配指标

Qwen替代Judge；HealthBench加权均分×100不是准确率。MedBrowseComp此表为参考答案匹配准确率。失败未评分，不补零，不把两个benchmark合成均分。上方核心答案准确率按问题与来源核验、固定全量分母，不能用本表的有效评分分母反推。

|组别|HealthBench加权分×100（有效/50）|MedBrowseComp正确/有效|
|---|---:|---:|
|Base|4.74（45/50）|9/49（18.37%）|
|Process-SFT|14.24（49/50）|12/49（24.49%）|
|GRPO13|17.76（49/50）|12/49（24.49%）|
|GRPO25|15.23（49/50）|10/48（20.83%）|

GRPO13在HealthBench答案加权均分最高，但对SFT的逐题答案比较为15胜18负15平；MedBrowseComp为1胜1负46平。不能由均分最高推导逐题全面占优。全部答案配对和区间见[统计表](summary.json)。

## 分题集胜负与答案配对

前者胜 / 前者负 / 平；本地R包含每题全部四组，答案配对只比较双方已评分，失败计数仍单列。

|前者 vs 后者|HealthBench Process R|MedBrowseComp Process R|HealthBench答案|MedBrowseComp答案|
|---|---|---|---|---|
|Process-SFT vs Base|27 / 21 / 2|35 / 14 / 1|23 / 13 / 8|5 / 2 / 41|
|GRPO13 vs Base|28 / 19 / 3|30 / 17 / 3|22 / 10 / 12|5 / 1 / 42|
|GRPO25 vs Base|26 / 22 / 2|35 / 11 / 4|21 / 9 / 14|5 / 3 / 39|
|GRPO13 vs Process-SFT|22 / 21 / 7|10 / 24 / 16|15 / 18 / 15|1 / 1 / 46|
|GRPO25 vs Process-SFT|21 / 23 / 6|20 / 15 / 15|21 / 15 / 12|2 / 2 / 43|
|GRPO25 vs GRPO13|20 / 22 / 8|29 / 6 / 15|17 / 20 / 11|2 / 3 / 42|

[污染排除敏感性](contamination_sensitivity.json) · [参考答案与题面边界核查](reference_boundary_audit.json)；这些分析不改原分。

</details>

## 为什么保留GRPO路线

本轮训练只更新Process，答案生成模块继续使用Base Final且参数不更新。跨基准四组同配置验证表明，改进不只体现为答案话术：完全重复搜索消失、实际交付覆盖提高，并且重大或Process硬失败题数少于SFT与Base。GRPO25因此作为当前Process交付/可靠性主候选；GRPO13保留为HealthBench答案指标的对照。继续研究GRPO有数据依据，尚不代表已执行下一轮训练或证明一个checkpoint在所有题型稳定最优。

## 数据与边界核查

- 两个benchmark各50题，四组全部重新检索，共400条；四组均使用本轮重新生成的轨迹，未复用开发对照轨迹。选题、排重与运行约束见[实验设置](setup.md)。
- 本地P1只看实际Final输入，P2/P3/P4看State、拆解与决策。没有因链路长自动扣分；信息收益保留，实质错误单独处理。
- 368条由三子代理隐藏模型身份、Final答案和Judge分初审；根补32条时接触过汇总/部分映射，再由身份盲读者逐条第二读。主审统一复核重大和重点高分，不宣称全程双盲或独立医生评审。
- 每条判定保留私有溯源，公开内容点与理由。过强标签、普通漏更新和已恢复的中间错误不自动判重大；核心方向或任务被实质破坏才取消高奖励，四组规则一致。
- Q58一条GRPO13轨迹打开MedBrowseComp数据页，标记污染，未把gold页当研究证据。整题四组排除后的准确率排序不变；其余未知预训练污染不能据此排除。
- 6题空独占期参考按作者NA惯例处理；22条有效NA答案均判对，但答对NA不证明完成检索事实链。10道年份题未发现答对年份仅因缺月日被判错。
- HealthBench Q22/Q27/Q42有题面/判据范围或正负冲突；Q50的润色规则与原题相符。统一排除前三题的敏感性排序保持，原始主分不改。
- Q64多个P成分、Q65/Q78/Q99胜方与字母约束存在题面歧义。公开逐题摘要保留它们，不挑题重排胜负。

## 逐题记录

<details>
<summary>展开100道题，每题含Base、SFT、GRPO13、GRPO25</summary>

|题号|题集|四组评分与具体问题|
|---|---|---|
|Q01|HealthBench|[查看四组完整审核摘要](questions/Q01.md)|
|Q02|HealthBench|[查看四组完整审核摘要](questions/Q02.md)|
|Q03|HealthBench|[查看四组完整审核摘要](questions/Q03.md)|
|Q04|HealthBench|[查看四组完整审核摘要](questions/Q04.md)|
|Q05|HealthBench|[查看四组完整审核摘要](questions/Q05.md)|
|Q06|HealthBench|[查看四组完整审核摘要](questions/Q06.md)|
|Q07|HealthBench|[查看四组完整审核摘要](questions/Q07.md)|
|Q08|HealthBench|[查看四组完整审核摘要](questions/Q08.md)|
|Q09|HealthBench|[查看四组完整审核摘要](questions/Q09.md)|
|Q10|HealthBench|[查看四组完整审核摘要](questions/Q10.md)|
|Q11|HealthBench|[查看四组完整审核摘要](questions/Q11.md)|
|Q12|HealthBench|[查看四组完整审核摘要](questions/Q12.md)|
|Q13|HealthBench|[查看四组完整审核摘要](questions/Q13.md)|
|Q14|HealthBench|[查看四组完整审核摘要](questions/Q14.md)|
|Q15|HealthBench|[查看四组完整审核摘要](questions/Q15.md)|
|Q16|HealthBench|[查看四组完整审核摘要](questions/Q16.md)|
|Q17|HealthBench|[查看四组完整审核摘要](questions/Q17.md)|
|Q18|HealthBench|[查看四组完整审核摘要](questions/Q18.md)|
|Q19|HealthBench|[查看四组完整审核摘要](questions/Q19.md)|
|Q20|HealthBench|[查看四组完整审核摘要](questions/Q20.md)|
|Q21|HealthBench|[查看四组完整审核摘要](questions/Q21.md)|
|Q22|HealthBench|[查看四组完整审核摘要](questions/Q22.md)|
|Q23|HealthBench|[查看四组完整审核摘要](questions/Q23.md)|
|Q24|HealthBench|[查看四组完整审核摘要](questions/Q24.md)|
|Q25|HealthBench|[查看四组完整审核摘要](questions/Q25.md)|
|Q26|HealthBench|[查看四组完整审核摘要](questions/Q26.md)|
|Q27|HealthBench|[查看四组完整审核摘要](questions/Q27.md)|
|Q28|HealthBench|[查看四组完整审核摘要](questions/Q28.md)|
|Q29|HealthBench|[查看四组完整审核摘要](questions/Q29.md)|
|Q30|HealthBench|[查看四组完整审核摘要](questions/Q30.md)|
|Q31|HealthBench|[查看四组完整审核摘要](questions/Q31.md)|
|Q32|HealthBench|[查看四组完整审核摘要](questions/Q32.md)|
|Q33|HealthBench|[查看四组完整审核摘要](questions/Q33.md)|
|Q34|HealthBench|[查看四组完整审核摘要](questions/Q34.md)|
|Q35|HealthBench|[查看四组完整审核摘要](questions/Q35.md)|
|Q36|HealthBench|[查看四组完整审核摘要](questions/Q36.md)|
|Q37|HealthBench|[查看四组完整审核摘要](questions/Q37.md)|
|Q38|HealthBench|[查看四组完整审核摘要](questions/Q38.md)|
|Q39|HealthBench|[查看四组完整审核摘要](questions/Q39.md)|
|Q40|HealthBench|[查看四组完整审核摘要](questions/Q40.md)|
|Q41|HealthBench|[查看四组完整审核摘要](questions/Q41.md)|
|Q42|HealthBench|[查看四组完整审核摘要](questions/Q42.md)|
|Q43|HealthBench|[查看四组完整审核摘要](questions/Q43.md)|
|Q44|HealthBench|[查看四组完整审核摘要](questions/Q44.md)|
|Q45|HealthBench|[查看四组完整审核摘要](questions/Q45.md)|
|Q46|HealthBench|[查看四组完整审核摘要](questions/Q46.md)|
|Q47|HealthBench|[查看四组完整审核摘要](questions/Q47.md)|
|Q48|HealthBench|[查看四组完整审核摘要](questions/Q48.md)|
|Q49|HealthBench|[查看四组完整审核摘要](questions/Q49.md)|
|Q50|HealthBench|[查看四组完整审核摘要](questions/Q50.md)|
|Q51|MedBrowseComp|[查看四组完整审核摘要](questions/Q51.md)|
|Q52|MedBrowseComp|[查看四组完整审核摘要](questions/Q52.md)|
|Q53|MedBrowseComp|[查看四组完整审核摘要](questions/Q53.md)|
|Q54|MedBrowseComp|[查看四组完整审核摘要](questions/Q54.md)|
|Q55|MedBrowseComp|[查看四组完整审核摘要](questions/Q55.md)|
|Q56|MedBrowseComp|[查看四组完整审核摘要](questions/Q56.md)|
|Q57|MedBrowseComp|[查看四组完整审核摘要](questions/Q57.md)|
|Q58|MedBrowseComp|[查看四组完整审核摘要](questions/Q58.md)|
|Q59|MedBrowseComp|[查看四组完整审核摘要](questions/Q59.md)|
|Q60|MedBrowseComp|[查看四组完整审核摘要](questions/Q60.md)|
|Q61|MedBrowseComp|[查看四组完整审核摘要](questions/Q61.md)|
|Q62|MedBrowseComp|[查看四组完整审核摘要](questions/Q62.md)|
|Q63|MedBrowseComp|[查看四组完整审核摘要](questions/Q63.md)|
|Q64|MedBrowseComp|[查看四组完整审核摘要](questions/Q64.md)|
|Q65|MedBrowseComp|[查看四组完整审核摘要](questions/Q65.md)|
|Q66|MedBrowseComp|[查看四组完整审核摘要](questions/Q66.md)|
|Q67|MedBrowseComp|[查看四组完整审核摘要](questions/Q67.md)|
|Q68|MedBrowseComp|[查看四组完整审核摘要](questions/Q68.md)|
|Q69|MedBrowseComp|[查看四组完整审核摘要](questions/Q69.md)|
|Q70|MedBrowseComp|[查看四组完整审核摘要](questions/Q70.md)|
|Q71|MedBrowseComp|[查看四组完整审核摘要](questions/Q71.md)|
|Q72|MedBrowseComp|[查看四组完整审核摘要](questions/Q72.md)|
|Q73|MedBrowseComp|[查看四组完整审核摘要](questions/Q73.md)|
|Q74|MedBrowseComp|[查看四组完整审核摘要](questions/Q74.md)|
|Q75|MedBrowseComp|[查看四组完整审核摘要](questions/Q75.md)|
|Q76|MedBrowseComp|[查看四组完整审核摘要](questions/Q76.md)|
|Q77|MedBrowseComp|[查看四组完整审核摘要](questions/Q77.md)|
|Q78|MedBrowseComp|[查看四组完整审核摘要](questions/Q78.md)|
|Q79|MedBrowseComp|[查看四组完整审核摘要](questions/Q79.md)|
|Q80|MedBrowseComp|[查看四组完整审核摘要](questions/Q80.md)|
|Q81|MedBrowseComp|[查看四组完整审核摘要](questions/Q81.md)|
|Q82|MedBrowseComp|[查看四组完整审核摘要](questions/Q82.md)|
|Q83|MedBrowseComp|[查看四组完整审核摘要](questions/Q83.md)|
|Q84|MedBrowseComp|[查看四组完整审核摘要](questions/Q84.md)|
|Q85|MedBrowseComp|[查看四组完整审核摘要](questions/Q85.md)|
|Q86|MedBrowseComp|[查看四组完整审核摘要](questions/Q86.md)|
|Q87|MedBrowseComp|[查看四组完整审核摘要](questions/Q87.md)|
|Q88|MedBrowseComp|[查看四组完整审核摘要](questions/Q88.md)|
|Q89|MedBrowseComp|[查看四组完整审核摘要](questions/Q89.md)|
|Q90|MedBrowseComp|[查看四组完整审核摘要](questions/Q90.md)|
|Q91|MedBrowseComp|[查看四组完整审核摘要](questions/Q91.md)|
|Q92|MedBrowseComp|[查看四组完整审核摘要](questions/Q92.md)|
|Q93|MedBrowseComp|[查看四组完整审核摘要](questions/Q93.md)|
|Q94|MedBrowseComp|[查看四组完整审核摘要](questions/Q94.md)|
|Q95|MedBrowseComp|[查看四组完整审核摘要](questions/Q95.md)|
|Q96|MedBrowseComp|[查看四组完整审核摘要](questions/Q96.md)|
|Q97|MedBrowseComp|[查看四组完整审核摘要](questions/Q97.md)|
|Q98|MedBrowseComp|[查看四组完整审核摘要](questions/Q98.md)|
|Q99|MedBrowseComp|[查看四组完整审核摘要](questions/Q99.md)|
|Q100|MedBrowseComp|[查看四组完整审核摘要](questions/Q100.md)|

</details>

## 离线复核

```bash
python -B verify_results.py
python -B -m unittest test_public_results -v
python -B code/verify_public.py --tests
```

第一项复算全部400条及逐题胜负；第二项验证失败、重大折扣及不利结果保留；第三项检查152个冻结源文件及合成评测合同。均不加载模型、不请求搜索/Judge、不训练。100道实际评测原题已公开；内容点和问题记录用于溯源，不包含gold、原始答卷或私有训练题，也不表示临床认证。
