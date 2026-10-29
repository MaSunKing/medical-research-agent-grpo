# Process开发对照评测 · HealthBench50

**[完整评测规则与结果](../../evaluation.md)** · **[50题逐题索引](../../records.md#process-development)** · [版本三概览](../../README.md) · [名词与指标解释](../../glossary.md)

本实验包含Base/SFT各50条，以及随后加入的GRPO13/25各50条。所有分数与文字问题保留；完整叙述已统一到版本三章节，避免训练、结果和旧计划重复堆叠。

| 需要查看 | 入口 |
|---|---|
| 评分含义、四组总表、HB配对与失败 | [评测](../../evaluation.md) |
| 训练、奖励与概率校正 | [GRPO](../../grpo.md) |
| 50题内容点与具体错误 | [逐题记录](../../records.md#process-development) |
| Base/SFT与GRPO运行配置 | [配置核对](runtime.md) |
| 200条机器可读分数 | [CSV](scores_200.csv) · [JSON](summary.json) |

历史对照配置差异、State重大问题和统计区间保留在评测章节。这里不将“没有完全相同Search”解释成不存在所有语义冗余。

## 评测原题

每个逐题页面已在评分之前加入实际评测原题、来源编号和SHA256；另有[50道原题](questions_50.json)供核对。保留原评分及具体问题，不公开gold、原始模型答案或证据正文。
