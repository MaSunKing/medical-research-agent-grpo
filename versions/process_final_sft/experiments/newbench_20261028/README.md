# 新100题四组评测｜GRPO的改进与边界

**同一工程框架下，GRPO保持0次完全重复Search，并减少严重Process问题；GRPO25交付更多适用证据，GRPO13在HealthBench答案加权分上最高。** 这些优势分别由行为、内容和答案指标支持，不能合并为“所有指标都优于SFT”。

[版本三评测](../../evaluation.md) · [逐题索引](../../records.md#新增100题400条) · [100道评测原题](questions_100.json) · [400条记录](records_400.json) · [复算代码](verify_results.py)

## 一眼看到的变化

|同配置新题指标|Base|Process-SFT|GRPO13|GRPO25|
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

## benchmark答案结果

Qwen替代Judge；HealthBench加权均分×100不是准确率。MedBrowseComp为参考答案匹配准确率。失败未评分，不补零，不把两个benchmark合成均分。

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

## 为什么保留GRPO路线

本轮训练只更新Process，Final保持Base。新的四组同配置检验表明，改进不只体现为答案话术：完全重复搜索消失、实际交付覆盖提高，并且重大或Process硬失败题数少于SFT与Base。GRPO25因此作为当前Process交付/可靠性主候选；GRPO13保留为HealthBench答案指标的对照。继续研究GRPO有数据依据，尚不代表已执行下一轮训练或证明一个checkpoint在所有题型稳定最优。

## 数据与边界核查

- 两个benchmark各50道新题，四组全部重新检索，共400条；没有拿旧Base/SFT轨迹充作新对照。选题、排重与运行约束见[实验设置](setup.md)。
- 本地P1只看实际Final输入，P2/P3/P4看State、拆解与决策。没有因链路长自动扣分；信息收益保留，实质错误单独处理。
- 368条由三子代理隐藏模型身份、Final答案和Judge分初审；根补32条时接触过汇总/部分映射，再由身份盲读者逐条第二读。主审统一复核重大和重点高分，不宣称全程双盲或独立医生评审。
- 原稿与最终裁决均保留私有溯源。10条过强标签/漏更新由重大降普通，3条恢复期护理的内容覆盖统一调整为部分支持，2条证据理由澄清。不是为了让某组获胜而改权重。
- Q58一条GRPO13轨迹打开MedBrowseComp数据页，标记污染，未把gold页当研究证据。整题四组排除后的准确率排序不变；其余未知预训练污染不能据此排除。
- 6题空独占期参考按作者NA惯例处理；22条有效NA答案均判对，但答对NA不证明完成检索事实链。10道年份题未发现答对年份仅因缺月日被判错。
- HealthBench Q22/Q27/Q42有题面/判据范围或正负冲突；Q50的润色规则与原题相符。统一排除前三题的敏感性排序保持，原始主分不改。
- Q64多个P成分、Q65/Q78/Q99胜方与字母约束存在题面歧义。公开逐题摘要保留它们，不挑题重排胜负。

## 逐题记录

<details>
<summary>展开100道题，每题含Base、SFT、GRPO13、GRPO25</summary>

|题号|题集|四组评分与具体问题|
|---|---|---|
|新Q01|HealthBench|[查看四组完整审核摘要](questions/Q01.md)|
|新Q02|HealthBench|[查看四组完整审核摘要](questions/Q02.md)|
|新Q03|HealthBench|[查看四组完整审核摘要](questions/Q03.md)|
|新Q04|HealthBench|[查看四组完整审核摘要](questions/Q04.md)|
|新Q05|HealthBench|[查看四组完整审核摘要](questions/Q05.md)|
|新Q06|HealthBench|[查看四组完整审核摘要](questions/Q06.md)|
|新Q07|HealthBench|[查看四组完整审核摘要](questions/Q07.md)|
|新Q08|HealthBench|[查看四组完整审核摘要](questions/Q08.md)|
|新Q09|HealthBench|[查看四组完整审核摘要](questions/Q09.md)|
|新Q10|HealthBench|[查看四组完整审核摘要](questions/Q10.md)|
|新Q11|HealthBench|[查看四组完整审核摘要](questions/Q11.md)|
|新Q12|HealthBench|[查看四组完整审核摘要](questions/Q12.md)|
|新Q13|HealthBench|[查看四组完整审核摘要](questions/Q13.md)|
|新Q14|HealthBench|[查看四组完整审核摘要](questions/Q14.md)|
|新Q15|HealthBench|[查看四组完整审核摘要](questions/Q15.md)|
|新Q16|HealthBench|[查看四组完整审核摘要](questions/Q16.md)|
|新Q17|HealthBench|[查看四组完整审核摘要](questions/Q17.md)|
|新Q18|HealthBench|[查看四组完整审核摘要](questions/Q18.md)|
|新Q19|HealthBench|[查看四组完整审核摘要](questions/Q19.md)|
|新Q20|HealthBench|[查看四组完整审核摘要](questions/Q20.md)|
|新Q21|HealthBench|[查看四组完整审核摘要](questions/Q21.md)|
|新Q22|HealthBench|[查看四组完整审核摘要](questions/Q22.md)|
|新Q23|HealthBench|[查看四组完整审核摘要](questions/Q23.md)|
|新Q24|HealthBench|[查看四组完整审核摘要](questions/Q24.md)|
|新Q25|HealthBench|[查看四组完整审核摘要](questions/Q25.md)|
|新Q26|HealthBench|[查看四组完整审核摘要](questions/Q26.md)|
|新Q27|HealthBench|[查看四组完整审核摘要](questions/Q27.md)|
|新Q28|HealthBench|[查看四组完整审核摘要](questions/Q28.md)|
|新Q29|HealthBench|[查看四组完整审核摘要](questions/Q29.md)|
|新Q30|HealthBench|[查看四组完整审核摘要](questions/Q30.md)|
|新Q31|HealthBench|[查看四组完整审核摘要](questions/Q31.md)|
|新Q32|HealthBench|[查看四组完整审核摘要](questions/Q32.md)|
|新Q33|HealthBench|[查看四组完整审核摘要](questions/Q33.md)|
|新Q34|HealthBench|[查看四组完整审核摘要](questions/Q34.md)|
|新Q35|HealthBench|[查看四组完整审核摘要](questions/Q35.md)|
|新Q36|HealthBench|[查看四组完整审核摘要](questions/Q36.md)|
|新Q37|HealthBench|[查看四组完整审核摘要](questions/Q37.md)|
|新Q38|HealthBench|[查看四组完整审核摘要](questions/Q38.md)|
|新Q39|HealthBench|[查看四组完整审核摘要](questions/Q39.md)|
|新Q40|HealthBench|[查看四组完整审核摘要](questions/Q40.md)|
|新Q41|HealthBench|[查看四组完整审核摘要](questions/Q41.md)|
|新Q42|HealthBench|[查看四组完整审核摘要](questions/Q42.md)|
|新Q43|HealthBench|[查看四组完整审核摘要](questions/Q43.md)|
|新Q44|HealthBench|[查看四组完整审核摘要](questions/Q44.md)|
|新Q45|HealthBench|[查看四组完整审核摘要](questions/Q45.md)|
|新Q46|HealthBench|[查看四组完整审核摘要](questions/Q46.md)|
|新Q47|HealthBench|[查看四组完整审核摘要](questions/Q47.md)|
|新Q48|HealthBench|[查看四组完整审核摘要](questions/Q48.md)|
|新Q49|HealthBench|[查看四组完整审核摘要](questions/Q49.md)|
|新Q50|HealthBench|[查看四组完整审核摘要](questions/Q50.md)|
|新Q51|MedBrowseComp|[查看四组完整审核摘要](questions/Q51.md)|
|新Q52|MedBrowseComp|[查看四组完整审核摘要](questions/Q52.md)|
|新Q53|MedBrowseComp|[查看四组完整审核摘要](questions/Q53.md)|
|新Q54|MedBrowseComp|[查看四组完整审核摘要](questions/Q54.md)|
|新Q55|MedBrowseComp|[查看四组完整审核摘要](questions/Q55.md)|
|新Q56|MedBrowseComp|[查看四组完整审核摘要](questions/Q56.md)|
|新Q57|MedBrowseComp|[查看四组完整审核摘要](questions/Q57.md)|
|新Q58|MedBrowseComp|[查看四组完整审核摘要](questions/Q58.md)|
|新Q59|MedBrowseComp|[查看四组完整审核摘要](questions/Q59.md)|
|新Q60|MedBrowseComp|[查看四组完整审核摘要](questions/Q60.md)|
|新Q61|MedBrowseComp|[查看四组完整审核摘要](questions/Q61.md)|
|新Q62|MedBrowseComp|[查看四组完整审核摘要](questions/Q62.md)|
|新Q63|MedBrowseComp|[查看四组完整审核摘要](questions/Q63.md)|
|新Q64|MedBrowseComp|[查看四组完整审核摘要](questions/Q64.md)|
|新Q65|MedBrowseComp|[查看四组完整审核摘要](questions/Q65.md)|
|新Q66|MedBrowseComp|[查看四组完整审核摘要](questions/Q66.md)|
|新Q67|MedBrowseComp|[查看四组完整审核摘要](questions/Q67.md)|
|新Q68|MedBrowseComp|[查看四组完整审核摘要](questions/Q68.md)|
|新Q69|MedBrowseComp|[查看四组完整审核摘要](questions/Q69.md)|
|新Q70|MedBrowseComp|[查看四组完整审核摘要](questions/Q70.md)|
|新Q71|MedBrowseComp|[查看四组完整审核摘要](questions/Q71.md)|
|新Q72|MedBrowseComp|[查看四组完整审核摘要](questions/Q72.md)|
|新Q73|MedBrowseComp|[查看四组完整审核摘要](questions/Q73.md)|
|新Q74|MedBrowseComp|[查看四组完整审核摘要](questions/Q74.md)|
|新Q75|MedBrowseComp|[查看四组完整审核摘要](questions/Q75.md)|
|新Q76|MedBrowseComp|[查看四组完整审核摘要](questions/Q76.md)|
|新Q77|MedBrowseComp|[查看四组完整审核摘要](questions/Q77.md)|
|新Q78|MedBrowseComp|[查看四组完整审核摘要](questions/Q78.md)|
|新Q79|MedBrowseComp|[查看四组完整审核摘要](questions/Q79.md)|
|新Q80|MedBrowseComp|[查看四组完整审核摘要](questions/Q80.md)|
|新Q81|MedBrowseComp|[查看四组完整审核摘要](questions/Q81.md)|
|新Q82|MedBrowseComp|[查看四组完整审核摘要](questions/Q82.md)|
|新Q83|MedBrowseComp|[查看四组完整审核摘要](questions/Q83.md)|
|新Q84|MedBrowseComp|[查看四组完整审核摘要](questions/Q84.md)|
|新Q85|MedBrowseComp|[查看四组完整审核摘要](questions/Q85.md)|
|新Q86|MedBrowseComp|[查看四组完整审核摘要](questions/Q86.md)|
|新Q87|MedBrowseComp|[查看四组完整审核摘要](questions/Q87.md)|
|新Q88|MedBrowseComp|[查看四组完整审核摘要](questions/Q88.md)|
|新Q89|MedBrowseComp|[查看四组完整审核摘要](questions/Q89.md)|
|新Q90|MedBrowseComp|[查看四组完整审核摘要](questions/Q90.md)|
|新Q91|MedBrowseComp|[查看四组完整审核摘要](questions/Q91.md)|
|新Q92|MedBrowseComp|[查看四组完整审核摘要](questions/Q92.md)|
|新Q93|MedBrowseComp|[查看四组完整审核摘要](questions/Q93.md)|
|新Q94|MedBrowseComp|[查看四组完整审核摘要](questions/Q94.md)|
|新Q95|MedBrowseComp|[查看四组完整审核摘要](questions/Q95.md)|
|新Q96|MedBrowseComp|[查看四组完整审核摘要](questions/Q96.md)|
|新Q97|MedBrowseComp|[查看四组完整审核摘要](questions/Q97.md)|
|新Q98|MedBrowseComp|[查看四组完整审核摘要](questions/Q98.md)|
|新Q99|MedBrowseComp|[查看四组完整审核摘要](questions/Q99.md)|
|新Q100|MedBrowseComp|[查看四组完整审核摘要](questions/Q100.md)|

</details>

## 离线复核

```bash
python -B verify_results.py
python -B -m unittest test_public_results -v
python -B code/verify_public.py --tests
```

第一项复算全部400条及逐题胜负；第二项验证失败、重大折扣及不利结果保留；第三项检查152个冻结源文件及合成评测合同。均不加载模型、不请求搜索/Judge、不训练。100道实际评测原题已公开；内容点和问题记录用于溯源，不包含gold、原始答卷或私有训练题，也不表示临床认证。
