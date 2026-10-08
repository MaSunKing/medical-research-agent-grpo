# 50 题 Base / SFT 配对评测

## 评测结论

在同一 Qwen3-8B backbone、同一评测 Runtime 与同一冻结题集下，SFT 在 50 题配对评测中取得 **36 胜、10 负、2 平、2 个双方失败平局**。Base 严格端到端通过 41/50，SFT 通过 43/50。

在双方端到端均有效的 36 对样本中，冻结综合均分为 Base **64.36**、SFT **81.42**，差值 **+17.06**。这说明当前冻结样本上，SFT 更稳定地完成了工具协议、证据选择、状态更新、停止与引用式回答；该结果不等同于临床正确率，也不是统计显著性结论。

## 评测方法

- 评测工具：**ChatGPT Pro**，使用项目内冻结规则逐题进行证据约束审核，并使用 Codex 文件审计流程读取评测包。
- 评测者可见 Base / SFT 标签，因此不声称盲评。
- 输入仅来自冻结包中的问题、协议记录、工具回执、已打开证据、State 与正式 Final；不使用外部医学知识替模型补答案。
- 维度：Checklist、Search、Browse、Evidence、State、Stop、Final Completeness、Fidelity、Citation、Clarity。
- 原始维度采用 0–4 的离散评分；null 表示不可观察，不按 0 分处理。
- 协议有效性优先：没有被冻结审计接受的 Final 时，不生成正式综合分。
- 逐题分数直接按同一冻结 rubric 加权计算，不另设展示上限；100.00 只表示在本次可观察维度的离散档位中达到最高档，不等同于绝对医学正确。

## 汇总指标

| 指标 | Base | SFT |
|---|---:|---:|
| 严格端到端通过 | 41/50 | 43/50 |
| strict-final success | 49/50 | 44/50 |
| first-attempt success | 39/50 | 35/50 |
| 经一次格式纠正后恢复的 Final | 10/50 | 9/50 |
| 严格胜场 | 10 | 36 |
| 条件综合均分 | 64.36 | 81.42 |

有效平局 2 题；双方失败平局 2 题。strict-final 与 first-attempt 反映协议行为，不应单独解释为答案质量。

## 逐题结果

| 题号 | 冻结问题 ID | Base | SFT | Base/SFT E2E | 结果 |
|---|---|---:|---:|---|---|
| q01 | `medgrpo_v1_04e088f6f52347b0` | 55.00 | 68.75 | true/true | SFT |
| q02 | `medgrpo_v1_2be04f0a3852fb54` | 50.00 | 71.25 | true/true | SFT |
| q03 | `medgrpo_v1_894f314fcdf79ee7` | 45.00 | 66.25 | true/true | SFT |
| q04 | `medgrpo_v1_2f322b435997e76a` | 67.86 | 31.25 | true/true | Base |
| q05 | `medgrpo_v1_8cb9fecbba10cbce` | 57.50 | 80.00 | true/true | SFT |
| q06 | `medgrpo_v1_099cf204ec96625c` | 50.00 | 75.00 | true/true | SFT |
| q07 | `medgrpo_v1_4d99d4174bb3500f` | 50.00 | 35.00 | true/true | Base |
| q08 | `medgrpo_v1_60497329178f53db` | — | 66.25 | false/true | SFT |
| q09 | `medgrpo_v1_e9171e78ea2a08b3` | 53.75 | — | true/false | Base |
| q10 | `medgrpo_v1_f21527f45469c7bb` | 60.00 | 68.75 | true/true | SFT |
| q11 | `medgrpo_v1_6163c074ccb78f7a` | — | 66.07 | false/true | SFT |
| q12 | `medgrpo_v1_888d7b38e86f8a3a` | 60.42 | 58.75 | true/true | 平局 |
| q13 | `medgrpo_v1_bee00f262b29f211` | — | 92.50 | false/true | SFT |
| q14 | `medgrpo_v1_5d3365f17cac4ef5` | 32.50 | 95.00 | true/true | SFT |
| q15 | `medgrpo_v1_90c6ec05d46f5f39` | 81.25 | 96.25 | true/true | SFT |
| q16 | `medgrpo_v1_de6eddebe535dd31` | — | 81.25 | false/true | SFT |
| q17 | `medgrpo_v1_30d72a5f177af814` | 67.86 | 95.83 | true/true | SFT |
| q18 | `medgrpo_v1_7b0af9da4047c976` | 52.50 | 97.50 | true/true | SFT |
| q19 | `medgrpo_v1_a662cdee1dd2c37e` | 58.75 | 100.00 | true/true | SFT |
| q20 | `medgrpo_v1_d2bf2d7b91040431` | — | 85.00 | false/true | SFT |
| q21 | `medgrpo_v1_dc4a772fdcb8ebbc` | — | — | false/false | 失败平局 |
| q22 | `medgrpo_v1_0e84758cb6f2c775` | 90.00 | 98.75 | true/true | SFT |
| q23 | `medgrpo_v1_77cc4d28b33619ff` | 61.25 | 98.75 | true/true | SFT |
| q24 | `medgrpo_v1_72169019a1402648` | 60.00 | 55.36 | true/true | Base |
| q25 | `medgrpo_v1_7370d4ef8bea75d6` | 81.25 | 70.00 | true/true | Base |
| q26 | `medgrpo_v1_d64242d3c2d36b10` | 87.50 | 95.00 | true/true | SFT |
| q27 | `medgrpo_v1_0b0631fc90e40a21` | 71.25 | 92.50 | true/true | SFT |
| q28 | `medgrpo_v1_72de682ab70216e2` | 63.75 | 98.75 | true/true | SFT |
| q29 | `medgrpo_v1_c5cc6850c399a5df` | 81.25 | 96.25 | true/true | SFT |
| q30 | `medgrpo_v1_ceb0ac46d6b397c5` | — | — | false/false | 失败平局 |
| q31 | `medgrpo_v1_fc301022e16c7bc5` | 71.25 | 98.75 | true/true | SFT |
| q32 | `medgrpo_v1_369513962bf4b437` | — | 80.00 | false/true | SFT |
| q33 | `medgrpo_v1_760fc3d67be8b7ba` | 56.94 | 92.50 | true/true | SFT |
| q34 | `medgrpo_v1_e129ac576a953b07` | 62.50 | — | true/false | Base |
| q35 | `medgrpo_v1_837ff9edadf2e750` | 65.00 | 82.50 | true/true | SFT |
| q36 | `medgrpo_v1_d03ead64ccf8d4eb` | — | 98.75 | false/true | SFT |
| q37 | `medgrpo_v1_d77afaad25a51c7d` | 78.75 | 67.50 | true/true | Base |
| q38 | `medgrpo_v1_20bc1dd642be1906` | 78.75 | 98.75 | true/true | SFT |
| q39 | `medgrpo_v1_6f4acbd99466f7de` | 64.58 | — | true/false | Base |
| q40 | `medgrpo_v1_354e24393be206a8` | 78.75 | 91.25 | true/true | SFT |
| q41 | `medgrpo_v1_f8985d53ae55046f` | 60.00 | 69.64 | true/true | SFT |
| q42 | `medgrpo_v1_467afd5694005296` | 65.00 | 97.50 | true/true | SFT |
| q43 | `medgrpo_v1_d06c873112f79f31` | 55.00 | 53.75 | true/true | 平局 |
| q44 | `medgrpo_v1_2216033aa966cc4e` | 44.44 | — | true/false | Base |
| q45 | `medgrpo_v1_e295d10c281d8707` | 83.75 | — | true/false | Base |
| q46 | `medgrpo_v1_182ec1058442b620` | 56.25 | 80.00 | true/true | SFT |
| q47 | `medgrpo_v1_3f9e8b3728bdfbcd` | 70.83 | 91.67 | true/true | SFT |
| q48 | `medgrpo_v1_5d02dd6ee49182ae` | 44.44 | 68.75 | true/true | SFT |
| q49 | `medgrpo_v1_5d607db025cca2d8` | 58.75 | 97.50 | true/true | SFT |
| q50 | `medgrpo_v1_990fb647220fd306` | 92.50 | 96.25 | true/true | SFT |

## 结果版本与边界

本页、公开 CSV 与 rubric JSON 共同组成 `evaluation_v1_1_frozen`，逐题分数、胜负和汇总均从同一版本读取，不混用其他复核草稿。q36 的正式结果采用审计接受的 SFT `attempt-002`；`attempt-001` 的结局与恢复原因在冻结包中不可观察，因此不据此声称已完全排除样本选择风险。

## 可复核文件

- [公开逐题 CSV](local50_scores_public.csv)
- [冻结评分规则](local50_rubric_public.json)

公开仓库不包含私有题目全文、完整模型 capture、API 响应、密钥或集群路径。完整私有评测包继续保存在本地，用于追溯每题 evidence、Final 与审核理由。
