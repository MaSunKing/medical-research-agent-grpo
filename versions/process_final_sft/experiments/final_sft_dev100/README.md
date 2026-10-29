# Final模块对照评测 · dev100

**[完整训练、评分与暂停加训依据](../../final_sft.md)** · **[100题逐题索引](../../records.md#final-comparison)** · [版本三概览](../../README.md) · [名词与指标解释](../../glossary.md)

本实验单独比较**答案生成模块（Final）**怎样使用同一份材料回答，不比较重新检索能力。基座答案生成器（Base Final）使用同一Qwen3-8B禁用本项目adapter；Final-SFT则是为答案生成任务单独训练的adapter。

Base及Final-SFT 1/1.5/2轮各100条，400条评分与问题全部保留。每题的四个版本使用同一份冻结输入，不同题各有自己的输入；失败不补零，四组共同有效97题。它是项目证据审核，不是HealthBench分。

这里固定的是每题实际输入，比较的是答案模块参数；四组Process评测则固定答案模块参数、允许检索材料变化，两种对照分别回答不同问题。

- [400条评分CSV](scores_400.csv)
- [汇总与同题比较JSON](summary.json)
- [全部100题诊断](../../records.md#final-comparison)

## 评测原题

每个逐题页面已在评分之前加入实际评测原题、来源编号和SHA256；另有[100道原题](questions_100.json)供核对。保留原评分及具体问题，不公开gold、原始模型答案或证据正文。
