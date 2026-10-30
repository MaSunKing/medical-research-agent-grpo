# 6. 逐题评分与问题索引

[概览](README.md) · [数据与实验设置](design.md) · [Process-SFT](process_sft.md) · [Final-SFT](final_sft.md) · [GRPO](grpo.md) · [评测](evaluation.md) · **逐题记录** · [名词与指标解释](glossary.md)


所有已有分数和审核结论保持原样；这里补充主题、字段解释和稳定入口。三项评测的逐题页均包含实际原题。gold、完整批量答卷、逐字证据和回执留在私有归档；少量输出与短证据按[真实案例页](../../examples/version3_cases/README.md)所述范围公开。

<a id="新增100题400条"></a>
<a id="cross-benchmark"></a>

## 跨基准验证评测 · HealthBench50＋MedBrowseComp50

[跨基准验证评测 · HealthBench50＋MedBrowseComp50](experiments/newbench_20261028/README.md) · [400条机器可读记录](experiments/newbench_20261028/records_400.json) · [400行分数](experiments/newbench_20261028/scores_400.csv)

Q01–50来自HealthBench，Q51–100来自MedBrowseComp。每题四组均列出分项、内容点、具体错误、失败及恢复，含全部400条。题号仅在本评测内有效；开发对照的Q01–Q50属于另一题集，不能按同号配对。

每题按“原题→本题怎样算答对→PreFinal内容点→四组判定与理由”排列；辅助分项放在页末折叠区。下表可直接打开本题标准，不需搜索JSON。每题同时列出四组核心答案正确／错误、必要信息齐全判定及理由。[准确率标准与MB任务边界](experiments/newbench_20261028/accuracy.md) · [400条答案判断](experiments/newbench_20261028/accuracy_records_400.json) · [准确率CSV](experiments/newbench_20261028/accuracy_scores_400.csv)。

|题号|benchmark|原题与四组结果|本题答对标准|
|---|---|---|---|
|Q01|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q01.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q01.md#answer-criteria)|
|Q02|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q02.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q02.md#answer-criteria)|
|Q03|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q03.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q03.md#answer-criteria)|
|Q04|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q04.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q04.md#answer-criteria)|
|Q05|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q05.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q05.md#answer-criteria)|
|Q06|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q06.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q06.md#answer-criteria)|
|Q07|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q07.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q07.md#answer-criteria)|
|Q08|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q08.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q08.md#answer-criteria)|
|Q09|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q09.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q09.md#answer-criteria)|
|Q10|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q10.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q10.md#answer-criteria)|
|Q11|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q11.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q11.md#answer-criteria)|
|Q12|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q12.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q12.md#answer-criteria)|
|Q13|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q13.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q13.md#answer-criteria)|
|Q14|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q14.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q14.md#answer-criteria)|
|Q15|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q15.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q15.md#answer-criteria)|
|Q16|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q16.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q16.md#answer-criteria)|
|Q17|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q17.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q17.md#answer-criteria)|
|Q18|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q18.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q18.md#answer-criteria)|
|Q19|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q19.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q19.md#answer-criteria)|
|Q20|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q20.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q20.md#answer-criteria)|
|Q21|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q21.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q21.md#answer-criteria)|
|Q22|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q22.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q22.md#answer-criteria)|
|Q23|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q23.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q23.md#answer-criteria)|
|Q24|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q24.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q24.md#answer-criteria)|
|Q25|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q25.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q25.md#answer-criteria)|
|Q26|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q26.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q26.md#answer-criteria)|
|Q27|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q27.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q27.md#answer-criteria)|
|Q28|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q28.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q28.md#answer-criteria)|
|Q29|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q29.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q29.md#answer-criteria)|
|Q30|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q30.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q30.md#answer-criteria)|
|Q31|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q31.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q31.md#answer-criteria)|
|Q32|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q32.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q32.md#answer-criteria)|
|Q33|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q33.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q33.md#answer-criteria)|
|Q34|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q34.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q34.md#answer-criteria)|
|Q35|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q35.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q35.md#answer-criteria)|
|Q36|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q36.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q36.md#answer-criteria)|
|Q37|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q37.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q37.md#answer-criteria)|
|Q38|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q38.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q38.md#answer-criteria)|
|Q39|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q39.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q39.md#answer-criteria)|
|Q40|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q40.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q40.md#answer-criteria)|
|Q41|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q41.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q41.md#answer-criteria)|
|Q42|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q42.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q42.md#answer-criteria)|
|Q43|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q43.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q43.md#answer-criteria)|
|Q44|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q44.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q44.md#answer-criteria)|
|Q45|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q45.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q45.md#answer-criteria)|
|Q46|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q46.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q46.md#answer-criteria)|
|Q47|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q47.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q47.md#answer-criteria)|
|Q48|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q48.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q48.md#answer-criteria)|
|Q49|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q49.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q49.md#answer-criteria)|
|Q50|HealthBench|[原题与判定](experiments/newbench_20261028/questions/Q50.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q50.md#answer-criteria)|
|Q51|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q51.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q51.md#answer-criteria)|
|Q52|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q52.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q52.md#answer-criteria)|
|Q53|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q53.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q53.md#answer-criteria)|
|Q54|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q54.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q54.md#answer-criteria)|
|Q55|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q55.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q55.md#answer-criteria)|
|Q56|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q56.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q56.md#answer-criteria)|
|Q57|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q57.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q57.md#answer-criteria)|
|Q58|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q58.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q58.md#answer-criteria)|
|Q59|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q59.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q59.md#answer-criteria)|
|Q60|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q60.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q60.md#answer-criteria)|
|Q61|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q61.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q61.md#answer-criteria)|
|Q62|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q62.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q62.md#answer-criteria)|
|Q63|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q63.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q63.md#answer-criteria)|
|Q64|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q64.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q64.md#answer-criteria)|
|Q65|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q65.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q65.md#answer-criteria)|
|Q66|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q66.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q66.md#answer-criteria)|
|Q67|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q67.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q67.md#answer-criteria)|
|Q68|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q68.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q68.md#answer-criteria)|
|Q69|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q69.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q69.md#answer-criteria)|
|Q70|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q70.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q70.md#answer-criteria)|
|Q71|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q71.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q71.md#answer-criteria)|
|Q72|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q72.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q72.md#answer-criteria)|
|Q73|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q73.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q73.md#answer-criteria)|
|Q74|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q74.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q74.md#answer-criteria)|
|Q75|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q75.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q75.md#answer-criteria)|
|Q76|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q76.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q76.md#answer-criteria)|
|Q77|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q77.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q77.md#answer-criteria)|
|Q78|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q78.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q78.md#answer-criteria)|
|Q79|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q79.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q79.md#answer-criteria)|
|Q80|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q80.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q80.md#answer-criteria)|
|Q81|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q81.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q81.md#answer-criteria)|
|Q82|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q82.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q82.md#answer-criteria)|
|Q83|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q83.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q83.md#answer-criteria)|
|Q84|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q84.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q84.md#answer-criteria)|
|Q85|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q85.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q85.md#answer-criteria)|
|Q86|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q86.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q86.md#answer-criteria)|
|Q87|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q87.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q87.md#answer-criteria)|
|Q88|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q88.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q88.md#answer-criteria)|
|Q89|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q89.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q89.md#answer-criteria)|
|Q90|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q90.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q90.md#answer-criteria)|
|Q91|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q91.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q91.md#answer-criteria)|
|Q92|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q92.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q92.md#answer-criteria)|
|Q93|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q93.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q93.md#answer-criteria)|
|Q94|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q94.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q94.md#answer-criteria)|
|Q95|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q95.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q95.md#answer-criteria)|
|Q96|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q96.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q96.md#answer-criteria)|
|Q97|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q97.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q97.md#answer-criteria)|
|Q98|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q98.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q98.md#answer-criteria)|
|Q99|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q99.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q99.md#answer-criteria)|
|Q100|MedBrowseComp|[原题与判定](experiments/newbench_20261028/questions/Q100.md#four-results)|[怎样算答对](experiments/newbench_20261028/questions/Q100.md#answer-criteria)|

<a id="process四组50题"></a>

<a id="旧50题process四组"></a>
<a id="process-development"></a>

## Process开发对照评测 · HealthBench50

每题包含同一组内容点、四组覆盖/P/R/HB、State/Checklist/决策理由及具体错误。失败也列出。**先看[字段速查](evaluation.md#表格字段速查)**，不要把质量R与答案HB直接比较。

| 题号 | 主题与逐题记录 |
|---|---|
| Q01 | [难治性慢性丛集头痛的氯胺酮疗效与证据强度](experiments/process_grpo_20261026/questions/Q01.md) |
| Q02 | [猪心异种移植长期免疫排斥风险及数据是否稳健](experiments/process_grpo_20261026/questions/Q02.md) |
| Q03 | [机械通气ICU患者预防性PPI的Meta分析与消化道出血效应](experiments/process_grpo_20261026/questions/Q03.md) |
| Q04 | [高超加工食物摄入与儿童白血病幸存者复发的研究及确定性](experiments/process_grpo_20261026/questions/Q04.md) |
| Q05 | [适用于已患产后抑郁青少年母亲的心理社会项目](experiments/process_grpo_20261026/questions/Q05.md) |
| Q06 | [标准治疗无效慢性荨麻疹中二代抗组胺药off-label使用（含加量）的疗效数据](experiments/process_grpo_20261026/questions/Q06.md) |
| Q07 | [孕期微剂量大麻暴露后的十年神经发育数据](experiments/process_grpo_20261026/questions/Q07.md) |
| Q08 | [顺势疗法滴剂用于轻度季节过敏的安全性](experiments/process_grpo_20261026/questions/Q08.md) |
| Q09 | [核实所述2020 AJHPM/组织声明身份](experiments/process_grpo_20261026/questions/Q09.md) |
| Q10 | [高流行区蜱叮咬后何时适合抗生素预防](experiments/process_grpo_20261026/questions/Q10.md) |
| Q11 | [PHASES如何用于5mm前交通动脉瘤的计分](experiments/process_grpo_20261026/questions/Q11.md) |
| Q12 | [该训练足球运动员的蛋白摄入建议](experiments/process_grpo_20261026/questions/Q12.md) |
| Q13 | [FVL阳性者COC血栓风险及是否可接受/禁忌的指南依据](experiments/process_grpo_20261026/questions/Q13.md) |
| Q14 | [当前青少年重复轻型头伤的检查选择与影像适应证](experiments/process_grpo_20261026/questions/Q14.md) |
| Q15 | [所述恶化化验、黄疸和意识变化的急性失代偿风险评估](experiments/process_grpo_20261026/questions/Q15.md) |
| Q16 | [72岁女性高血压房颤的CHA2DS2-VASc风险评估](experiments/process_grpo_20261026/questions/Q16.md) |
| Q17 | [JNC高血压急症静脉药物选择及降压目标](experiments/process_grpo_20261026/questions/Q17.md) |
| Q18 | [6岁儿童过敏反应的肾上腺素给药剂量与途径](experiments/process_grpo_20261026/questions/Q18.md) |
| Q19 | [婴儿VSD术后所给超声和稳定血流动力学的随访考虑](experiments/process_grpo_20261026/questions/Q19.md) |
| Q20 | [所述少尿、嗜睡且口服补液无改善婴儿是否需立即医院评估/补液](experiments/process_grpo_20261026/questions/Q20.md) |
| Q21 | [成人推荐睡眠小时数/范围](experiments/process_grpo_20261026/questions/Q21.md) |
| Q22 | [跳过疫苗的疾病风险及保护收益](experiments/process_grpo_20261026/questions/Q22.md) |
| Q23 | [长途飞行后单侧腿肿的血栓风险及不能凭疼痛缺失排除的边界](experiments/process_grpo_20261026/questions/Q23.md) |
| Q24 | [手卫生降低感冒流感传播的原因及可提炼的主要收益](experiments/process_grpo_20261026/questions/Q24.md) |
| Q25 | [ABCD皮肤镜规则的组成、评分和适用限制](experiments/process_grpo_20261026/questions/Q25.md) |
| Q26 | [STOPP/START最新版本及更新范围](experiments/process_grpo_20261026/questions/Q26.md) |
| Q27 | [DES后DAPT的正式共识/指南及6月、12月、延长治疗差异](experiments/process_grpo_20261026/questions/Q27.md) |
| Q28 | [两周龄正常分娩新生儿哺乳频率及喂养充分性](experiments/process_grpo_20261026/questions/Q28.md) |
| Q29 | [新版GINA指南身份/版本及成人适用范围](experiments/process_grpo_20261026/questions/Q29.md) |
| Q30 | [儿童ADHD正式诊断框架与持续时间、场景、损害及排除条件](experiments/process_grpo_20261026/questions/Q30.md) |
| Q31 | [克拉霉素三联失败且高耐药后的救援药物选择](experiments/process_grpo_20261026/questions/Q31.md) |
| Q32 | [EMS纳洛酮剂量及当前协议/地区适用性](experiments/process_grpo_20261026/questions/Q32.md) |
| Q33 | [健康成人轻度腹泻脱水的ORS/液体用量](experiments/process_grpo_20261026/questions/Q33.md) |
| Q34 | [ACOG泌乳支持正式指导及适用内容](experiments/process_grpo_20261026/questions/Q34.md) |
| Q35 | [核实当前正式屈光手术指南及用户所称2022版本的身份](experiments/process_grpo_20261026/questions/Q35.md) |
| Q36 | [黑色素瘤切除后皮肤检查频率及分期/风险调整](experiments/process_grpo_20261026/questions/Q36.md) |
| Q37 | [Nunavut本地试纸/血糖检测的日频率建议](experiments/process_grpo_20261026/questions/Q37.md) |
| Q38 | [BRCA1遗传咨询教学的核心内容](experiments/process_grpo_20261026/questions/Q38.md) |
| Q39 | [何时在乳果糖基础上加利福昔明及复发预防条件](experiments/process_grpo_20261026/questions/Q39.md) |
| Q40 | [房颤CHA2DS2-VASc5停华法林是否需桥接的ACC适用指征](experiments/process_grpo_20261026/questions/Q40.md) |
| Q41 | [ESRD腹腔镜胆囊切除单次头孢唑林预防剂量是否调整](experiments/process_grpo_20261026/questions/Q41.md) |
| Q42 | [核实APA所称2023指南及末页出处](experiments/process_grpo_20261026/questions/Q42.md) |
| Q43 | [48岁乳腺筛查年度与隔年间隔的指南依据](experiments/process_grpo_20261026/questions/Q43.md) |
| Q44 | [美国启动他汀的适用依据](experiments/process_grpo_20261026/questions/Q44.md) |
| Q45 | [妊娠11周TSH4.5/低边缘FT4的解释及LT4调整监测依据](experiments/process_grpo_20261026/questions/Q45.md) |
| Q46 | [银屑病生物制剂使用后的LFT监测间隔](experiments/process_grpo_20261026/questions/Q46.md) |
| Q47 | [终末期癌症谵妄镇静是否有统一标准及不同指南共识/差异](experiments/process_grpo_20261026/questions/Q47.md) |
| Q48 | [黄疸、肾衰、出血与Weil病关系](experiments/process_grpo_20261026/questions/Q48.md) |
| Q49 | [无警示症状幼儿100.4°F低热的家庭处理和就医界限](experiments/process_grpo_20261026/questions/Q49.md) |
| Q50 | [两岁儿童MRI水合氯醛与丙泊酚的选择及安全条件](experiments/process_grpo_20261026/questions/Q50.md) |

[下载200条Process评分](experiments/process_grpo_20261026/scores_200.csv) · [汇总与配对区间](experiments/process_grpo_20261026/summary.json)

<a id="final固定输入100题"></a>
<a id="final-comparison"></a>

## Final模块对照评测 · dev100

每题包含Base与SFT 1/1.5/2轮的四组分数、优点及具体问题。此处为项目四维评分，不是HealthBench。

| 序号 | 主题与逐题记录 |
|---|---|
| Dev001 | [植皮供皮区的水凝胶敷料](experiments/final_sft_dev100/questions/001.md) |
| Dev002 | [老年外科患者的专项评估](experiments/final_sft_dev100/questions/002.md) |
| Dev003 | [牙科气溶胶控制与传染预防](experiments/final_sft_dev100/questions/003.md) |
| Dev004 | [慢性肾病高钾治疗](experiments/final_sft_dev100/questions/004.md) |
| Dev005 | [胃癌手术联合腹腔化疗](experiments/final_sft_dev100/questions/005.md) |
| Dev006 | [血小板减少及血液病的抗纤溶治疗](experiments/final_sft_dev100/questions/006.md) |
| Dev007 | [右雷佐生与蒽环类心脏保护](experiments/final_sft_dev100/questions/007.md) |
| Dev008 | [HDL、LDL与血脂检测](experiments/final_sft_dev100/questions/008.md) |
| Dev009 | [血管手术中的远隔缺血预适应](experiments/final_sft_dev100/questions/009.md) |
| Dev010 | [抗氧化补充剂与黄斑变性进展](experiments/final_sft_dev100/questions/010.md) |
| Dev011 | [PC6刺激预防术后恶心呕吐](experiments/final_sft_dev100/questions/011.md) |
| Dev012 | [数字技术与囊性纤维化吸入治疗依从性](experiments/final_sft_dev100/questions/012.md) |
| Dev013 | [癌症厌食及消瘦患者的运动](experiments/final_sft_dev100/questions/013.md) |
| Dev014 | [咖啡因的作用与过量影响](experiments/final_sft_dev100/questions/014.md) |
| Dev015 | [COPD急性加重的短程激素](experiments/final_sft_dev100/questions/015.md) |
| Dev016 | [IVF促排卵激素比较](experiments/final_sft_dev100/questions/016.md) |
| Dev017 | [术前焦虑的音乐与镇静药](experiments/final_sft_dev100/questions/017.md) |
| Dev018 | [非特异性腰痛的非药物治疗](experiments/final_sft_dev100/questions/018.md) |
| Dev019 | [COVID-19抗体检测准确性](experiments/final_sft_dev100/questions/019.md) |
| Dev020 | [癌症患者重返工作的干预](experiments/final_sft_dev100/questions/020.md) |
| Dev021 | [囊性纤维化肺部非结核分枝杆菌治疗](experiments/final_sft_dev100/questions/021.md) |
| Dev022 | [多发性硬化结局预测模型](experiments/final_sft_dev100/questions/022.md) |
| Dev023 | [肌强直的药物治疗](experiments/final_sft_dev100/questions/023.md) |
| Dev024 | [抗生素预防风湿热复发](experiments/final_sft_dev100/questions/024.md) |
| Dev025 | [肩袖撕裂修复](experiments/final_sft_dev100/questions/025.md) |
| Dev026 | [孕期戒烟的心理社会支持](experiments/final_sft_dev100/questions/026.md) |
| Dev027 | [外科胶与术后胰瘘](experiments/final_sft_dev100/questions/027.md) |
| Dev028 | [原发性进行性失语的刺激或训练](experiments/final_sft_dev100/questions/028.md) |
| Dev029 | [儿童结核Xpert Ultra检测](experiments/final_sft_dev100/questions/029.md) |
| Dev030 | [女性HIV传播与治疗](experiments/final_sft_dev100/questions/030.md) |
| Dev031 | [妊娠期糖尿病的肌醇补充](experiments/final_sft_dev100/questions/031.md) |
| Dev032 | [间歇性外斜视治疗](experiments/final_sft_dev100/questions/032.md) |
| Dev033 | [剖宫产定义与风险](experiments/final_sft_dev100/questions/033.md) |
| Dev034 | [婴儿反流与喂养调整](experiments/final_sft_dev100/questions/034.md) |
| Dev035 | [口服减充血剂与血压](experiments/final_sft_dev100/questions/035.md) |
| Dev036 | [辅助生殖中的富血小板血浆](experiments/final_sft_dev100/questions/036.md) |
| Dev037 | [减少抗胆碱药与老年认知](experiments/final_sft_dev100/questions/037.md) |
| Dev038 | [疫苗前运动与流感预防](experiments/final_sft_dev100/questions/038.md) |
| Dev039 | [儿童青少年慢性肌肉骨骼痛](experiments/final_sft_dev100/questions/039.md) |
| Dev040 | [大麻使用方式与影响](experiments/final_sft_dev100/questions/040.md) |
| Dev041 | [COVID-19住院患者抗凝](experiments/final_sft_dev100/questions/041.md) |
| Dev042 | [辅助生殖第3或第5天胚胎移植](experiments/final_sft_dev100/questions/042.md) |
| Dev043 | [医护戒烟支持培训](experiments/final_sft_dev100/questions/043.md) |
| Dev044 | [高风险细菌感染后HUS预防](experiments/final_sft_dev100/questions/044.md) |
| Dev045 | [儿童牙列拥挤治疗](experiments/final_sft_dev100/questions/045.md) |
| Dev046 | [卵巢癌紫杉醇每周与三周给药](experiments/final_sft_dev100/questions/046.md) |
| Dev047 | [儿童肘部骨折治疗](experiments/final_sft_dev100/questions/047.md) |
| Dev048 | [学前机构健康饮食项目](experiments/final_sft_dev100/questions/048.md) |
| Dev049 | [胶质母细胞瘤MGMT甲基化检测](experiments/final_sft_dev100/questions/049.md) |
| Dev050 | [父母酒精或药物使用的心理干预](experiments/final_sft_dev100/questions/050.md) |
| Dev051 | [稳定哮喘停用LABA](experiments/final_sft_dev100/questions/051.md) |
| Dev052 | [高血压初始利尿剂与其他药物](experiments/final_sft_dev100/questions/052.md) |
| Dev053 | [首次非诱发癫痫后的复发因素](experiments/final_sft_dev100/questions/053.md) |
| Dev054 | [社区与门诊验光配镜](experiments/final_sft_dev100/questions/054.md) |
| Dev055 | [乳腺癌患者骨丢失预防](experiments/final_sft_dev100/questions/055.md) |
| Dev056 | [军人与退伍军人的健康服务](experiments/final_sft_dev100/questions/056.md) |
| Dev057 | [吸烟身体影响反馈与戒烟](experiments/final_sft_dev100/questions/057.md) |
| Dev058 | [GAD-7和GAD-2焦虑筛查](experiments/final_sft_dev100/questions/058.md) |
| Dev059 | [二氢可待因与非法阿片使用](experiments/final_sft_dev100/questions/059.md) |
| Dev060 | [手卫生、口罩与呼吸道病毒](experiments/final_sft_dev100/questions/060.md) |
| Dev061 | [梅尼埃病鼓室内激素](experiments/final_sft_dev100/questions/061.md) |
| Dev062 | [腹部手术切口关闭](experiments/final_sft_dev100/questions/062.md) |
| Dev063 | [学校简短干预与青少年物质使用](experiments/final_sft_dev100/questions/063.md) |
| Dev064 | [医护抗菌漱口及鼻喷防护](experiments/final_sft_dev100/questions/064.md) |
| Dev065 | [糖尿病合并慢性肾病的教育](experiments/final_sft_dev100/questions/065.md) |
| Dev066 | [精神疾病住院患者戒烟](experiments/final_sft_dev100/questions/066.md) |
| Dev067 | [清洁、消毒与物体表面病菌](experiments/final_sft_dev100/questions/067.md) |
| Dev068 | [衣原体快速即时检测](experiments/final_sft_dev100/questions/068.md) |
| Dev069 | [成人踝部骨折康复](experiments/final_sft_dev100/questions/069.md) |
| Dev070 | [两性霉素B制剂与肾功能](experiments/final_sft_dev100/questions/070.md) |
| Dev071 | [稳定COPD阿地溴铵联合LABA](experiments/final_sft_dev100/questions/071.md) |
| Dev072 | [耳真菌病的唑类治疗](experiments/final_sft_dev100/questions/072.md) |
| Dev073 | [钙补充与体重下降](experiments/final_sft_dev100/questions/073.md) |
| Dev074 | [晚期宫颈癌阴道出血控制](experiments/final_sft_dev100/questions/074.md) |
| Dev075 | [1型糖尿病基础胰岛素比较](experiments/final_sft_dev100/questions/075.md) |
| Dev076 | [儿童先心病术后强心药预防](experiments/final_sft_dev100/questions/076.md) |
| Dev077 | [癌症基因检测指导靶向治疗](experiments/final_sft_dev100/questions/077.md) |
| Dev078 | [难治性心绞痛体外反搏](experiments/final_sft_dev100/questions/078.md) |
| Dev079 | [胚胎移植中的缩宫素拮抗](experiments/final_sft_dev100/questions/079.md) |
| Dev080 | [无家可归与健康](experiments/final_sft_dev100/questions/080.md) |
| Dev081 | [固定矫治器初始弓丝](experiments/final_sft_dev100/questions/081.md) |
| Dev082 | [碳水化合物种类与食物来源](experiments/final_sft_dev100/questions/082.md) |
| Dev083 | [个体咨询与戒烟](experiments/final_sft_dev100/questions/083.md) |
| Dev084 | [婴儿腰椎穿刺体位](experiments/final_sft_dev100/questions/084.md) |
| Dev085 | [急性肝衰竭的肝支持系统](experiments/final_sft_dev100/questions/085.md) |
| Dev086 | [牙科专业人员提供戒烟帮助](experiments/final_sft_dev100/questions/086.md) |
| Dev087 | [透析患者疲劳干预](experiments/final_sft_dev100/questions/087.md) |
| Dev088 | [恒定压力充气床面与压疮](experiments/final_sft_dev100/questions/088.md) |
| Dev089 | [非诱发静脉血栓后的癌症筛查](experiments/final_sft_dev100/questions/089.md) |
| Dev090 | [髌腱病力量训练](experiments/final_sft_dev100/questions/090.md) |
| Dev091 | [囊性纤维化肺部MRSA清除](experiments/final_sft_dev100/questions/091.md) |
| Dev092 | [患者家属获取急诊救治的策略](experiments/final_sft_dev100/questions/092.md) |
| Dev093 | [捐献肝脏的机器灌注](experiments/final_sft_dev100/questions/093.md) |
| Dev094 | [女性尿失禁单切口吊带](experiments/final_sft_dev100/questions/094.md) |
| Dev095 | [家长对儿童常规疫苗的看法](experiments/final_sft_dev100/questions/095.md) |
| Dev096 | [阿司匹林或NSAID预防肠癌](experiments/final_sft_dev100/questions/096.md) |
| Dev097 | [阴道念珠菌病口服或局部抗真菌药](experiments/final_sft_dev100/questions/097.md) |
| Dev098 | [抗炎眼药与睑缘炎](experiments/final_sft_dev100/questions/098.md) |
| Dev099 | [痴呆躁动及精神症状的抗精神病药](experiments/final_sft_dev100/questions/099.md) |
| Dev100 | [酒类服务场所干预与伤害预防](experiments/final_sft_dev100/questions/100.md) |

[下载400条Final评分](experiments/final_sft_dev100/scores_400.csv) · [四组汇总](experiments/final_sft_dev100/summary.json)

## 审核与复算

Process开发对照评测中的100条GRPO轨迹隐藏checkpoint、Final答案和Judge分进行分段审阅；主审非完全盲，89条交叉复核，174个E锚点核验。同评测的Base/SFT共100条沿用已有同规则审核。Final模块对照的300条SFT答案隐藏标签，Base100条后补，不是四组同步完全盲评。跨基准验证的审核方式见[对应报告](experiments/newbench_20261028/README.md#数据与边界核查)。

来源/数学一致性检查不等于独立临床认证。可运行[复算脚本](experiments/process_grpo_20261026/verify_results.py)，核对600条公开分数及发布文件哈希。
