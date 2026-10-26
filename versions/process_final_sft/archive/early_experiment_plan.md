# 早期实验计划（历史存档）

> 历史方案存档：记录此前候选设计，不代表本轮已执行设置。[返回版本三](../README.md)

## 后续计划（原文保留，待讨论）

以下保留既有开发/评测/GRPO计划，本次不改样本数、组合和算法；其中数量不代表当前600题采集或972训练的执行记录，也不构成对新方案的确认。[独立评测计划](../evaluation.md)与[GRPO计划](../grpo.md)保持不变。

### 38题六checkpoint开发计划

冻结同一38题Pre-Final，每题生成Base及六个Final-SFT checkpoint答案：38×7=266个答案。共用Prompt、citation合同、matched seed和采样配置。该实验不重新检索，比较综合、覆盖、概念/数字绑定与引用支持，据此选择Frozen Final；不提前填入效果结果。

## 正式评测与GRPO

Process38与Final38是开发定位；之后冻结独立HealthBench20题，执行[1/2/3/4四组双轨评测](../evaluation.md)。最后按[GRPO公式与算法](../grpo.md)只更新Process、固定Final。开发题若用于选择Prompt或checkpoint，不再声称是独立最终测试。
