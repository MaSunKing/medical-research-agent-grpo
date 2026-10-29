# Q98：找到了目标试验，仍因动作绑定超限而失败

[案例索引](../README.md) · [四组完整问题记录](../../../versions/process_final_sft/experiments/newbench_20261028/questions/Q98.md) · [原始阶段摘录及错误](CASE.json)

这是跨基准验证评测中MedBrowseComp子集Q98的GRPO25真实失败记录。问题要求串联指定试验、G开头成分、截至2024年12月的整体FDA批准、公司上市及当日开盘价；原题的范围、日期和`NOT LISTED`条件均保留，完整英文见逐题页及`CASE.json`。

## 实际发生的过程

模型Checklist拆成4项：试验方案、整体批准公司、上市/价格、未上市分支。随后完成5次工具动作：

| 轮次 | 实际执行 | 观察结果 |
|---|---|---|
| 1 | web Search，R1 | 查找NCT00022191试验方案与比较 |
| 2 | Browse ClinicalTrials.gov，R1 | E1实际为注册管理提示，不能仅因网页访问成功就认定得到疗效证据；State保持R1 missing |
| 3 | paper Search，R1 | 转向试验方案和疗效论文 |
| 4 | Browse另一已发现网页，R1 | E2/E3定位正确试验及入排条件；尚未读到哪组疗效更好的结论 |
| 5 | web Search，R2 | 转向gemcitabine的整体FDA批准公司 |

注意第3步虽然使用paper渠道，随后Browse选到的来源仍是候选池内的网页；模型可跨已发现来源选择，不把渠道名称当作已读论文保证。

Loyola Medicine的[试验记录](https://www.loyolamedicine.org/clinical-research/clinical-trials/nct00022191)在E2中实际出现：

> Cisplatin Plus Gemcitabine With or Without Paclitaxel

同一片段还写道：

> It is not yet known which combination chemotherapy regimen is more effective

因此，定位试验方案有信息价值，但试验简介不能替代已完成的疗效结果，后续批准与股票链也尚未完成。原State标签为direct，gap却明确保留疗效与完成日期未证实；案例保留这一原始状态，不润色成已全部解决。

## 两次未执行的Browse提案

接下来模型连续两次生成Browse，均绑定`["R2","R3","R4"]`。当前合同每次Search/Browse最多绑定2项需求，因此两个提案都在动作校验处失败；完整原文与错误在`CASE.json`。

```json
{"action":"browse","requirement_ids":["R2","R3","R4"],"source_id":"T","reading_focus":"FDA approval date and manufacturer of gemcitabine intravesical system: Inlexzo versus overall FDA drug approval, comparative market value and stock listing"}
```

第二次改用来源字母U，但仍绑定相同3项，错误没有恢复。这两次是被拒的模型输出，不记成已经执行的Browse，也不补造工具响应。

## 如何定位失败

| 检查 | 真实记录 |
|---|---|
| Process终态 | `model_contract_failure` |
| 直接错误 | `browse.requirement_ids`最多2项，实际3项 |
| 正常/实际工具动作 | 5/5；不是工具预算用尽 |
| 已登记证据 | E1–E3，供过程诊断 |
| 实际Final输入 | 未生成 |
| Final答案 | `null`，未开始 |
| 本地P1 / R | 0 / 0：发现的材料尚未交付，保留其他分项作诊断 |
| benchmark答案分 | 未评分；不伪造为0分 |

完整对照仍见[本题四组记录](../../../versions/process_final_sft/experiments/newbench_20261028/questions/Q98.md)。本例说明“找到部分有效内容”和“可靠完成整条Agent任务”各有独立条件：原始信息收益保留用于分析，失败终态也完整保留。

这一错误可定位到动作需求绑定，不能直接归为API故障、上下文过长或Final拒答。案例不重跑、不修改原始题目，也不把缺失的最终答案补写出来。
