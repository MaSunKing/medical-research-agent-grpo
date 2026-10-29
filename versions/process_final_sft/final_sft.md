# 3. Final-SFT：训练、对照与决策

[概览](README.md) · [数据与实验设置](design.md) · [Process-SFT](process_sft.md) · **Final-SFT** · [GRPO](grpo.md) · [评测](evaluation.md) · [逐题记录](records.md)

评测入口：**[Final模块对照评测 · dev100](experiments/final_sft_dev100/README.md)** · [名词与指标解释](glossary.md)。

**在dev100固定输入对照中，Final-SFT的三个checkpoint均未超过Base Final；当前固定Base Final，独立Final-SFT未接入本轮GRPO。** 这是相应题集和评分条件下的模块选型结论。

| 训练边界 | 本页对象 |
|---|---|
| 输入 | 500训练题的固定Final请求与指定gold；另100题用于固定输入对照 |
| 初始化 | 同一量化基座上新建独立Final LoRA，不从Process adapter继续训练 |
| 更新对象 | 仅Final LoRA；loss只计算答案与EOS，不更新Process策略 |
| 产物 | 1／1.5／2轮Final checkpoint及其与Base Final的dev100对照记录 |

这条支线检验答案模块是否需要替换；Process-GRPO继续使用原972 Process初始化，不依赖本页的训练产物。

## 训练与对照设置

本实验回答“在相同Pre-final输入下，训练Final是否比原Base Final更好”。输入来自Process-SFT采集并冻结的实际Final请求，训练时不重新检索，不用Final输出倒改证据。gold采用用户指定gold_final600_v2，其他老师候选留作记录；gold的训练授权不等于独立人类临床认证。

| 项目 | 已执行设置 |
|---|---|
| 数据 | 600题预先固定500 train / 100 dev；两者题目ID不交叉 |
| 输入 | 同一冻结MESSAGES、证据与模板；dev四组实际输入、seed和采样参数一致 |
| 基座 | Qwen/Qwen3-8B，bitsandbytes NF4双重量化，BF16 compute |
| 初始化 | 新建独立Final LoRA，不从Process972 adapter继续训练 |
| LoRA | r32、alpha64、dropout0.05；q/k/v/o及gate/up/down投影 |
| optimizer / LR | AdamW，1e-4；accum8，warmup比例0.05，weight_decay=0，grad clip=1 |
| 训练范围 | 只算Final答案和EOS的target-token loss，输入/证据全部mask；context=15000 |
| checkpoint | 1轮63步、1.5轮95步、2轮127步 |
| Base Final | 同一基座禁用adapter；没有一份独立“Base训练” |
| 验证 | 三个SFT checkpoint各100条，再补Base100，共400条；不做本实验的HealthBench评分 |

## 答案如何评分

对原问题和实际可见证据，证据利用40%、问题相关性30%、忠实性20%、不确定性校准10%；各维0–4，按适用权重归一为百分制。没有已打开证据时证据利用记不适用，其他维仍评价是否切题、诚实回答。长度、引用数量和接近gold字面不直接计分。

原300条SFT答案按隐藏标签审阅；Base为后补评，旧300条未改分，因此不是四组同时完全盲评。失败保留原始候选及诊断分，有效答案分留空；不是按官方零分处理。分数为助手证据审阅，不是医学准确率或官方HealthBench分。

## Base与三个checkpoint的结果

| Final模型 | 格式有效/100 | 各自有效答案均分 | 四组共同有效97题均分 |
|---|---:|---:|---:|
| Base | **100** | **76.69** | **77.06** |
| SFT 1轮 | 98 | 58.21 | 58.40 |
| SFT 1.5轮 | 98 | 61.84 | 61.57 |
| SFT 2轮 | 99 | 66.06 | 65.98 |

| 与Base同题有效配对 | 题数 | SFT−Base均分 | SFT胜 | Base胜 | 平 |
|---|---:|---:|---:|---:|---:|
| 1轮 | 98 | −18.47 | 13 | 81 | 4 |
| 1.5轮 | 98 | −15.20 | 20 | 70 | 8 |
| 2轮 | 99 | −10.61 | 26 | 64 | 9 |

共同97题的证据利用均值（0–4）为Base2.8969、1轮2.2268、1.5轮2.3711、2轮2.5567；忠实性为2.6804、1.4433、1.5670、1.6598。差异不只来自格式失败，也涉及证据利用与事实忠实性。各自有效答案中重大问题条数26、63、51、53，分母不同，作诊断而非直接临床错误率。

## 为什么没有继续增加轮次

本轮三个SFT checkpoint均低于Base，尚无证据证明继续增加epoch就能超过Base。1→1.5→2轮评分实际上升，因此也不能据此宣布“已经证实过拟合”或“小模型必然被微调破坏”。更合适的决策是保留三个checkpoint和逐题材料，暂停盲目加训，以表现更好的Base Final作为固定生成器，先研究Process交付的信息是否改善。

已审计全部500条token边界：输入labels=-100，只监督真实答案和EOS；原生非thinking模板逐token一致，没有把Process adapter误挂为Final。合成微型模型的实际loss/梯度与标准因果CE一致；checkpoint哈希彼此不同。随后服务器配置/哈希回执无异常。这些排除所检查的明显工程错配，不证明数值路径和训练内容完全无问题。

600个问题均无中文；训练gold496英文、4中文，dev99英文、1中文。少量语言不一致值得整理，但不能用0.8%的训练中文解释整体下降。额外6条训练gold证据抽核未发现共同明显标签错误，不是全500题的独立医学认证。

目前不能确认下降由学习率、样本分布、答案风格、gold质量或其他因素中的哪项主导。assistant-only loss仍可学习利用输入，不能把不监督输入token解释成“没有学习理解输入”；不因此改成预测工具全文。

## 当前选择与后续边界

后续本轮GRPO统一使用冻结Base Final，更新对象仅Process LoRA；没有使用Final-SFT结果来宣称Final训练成功。若重启Final-SFT，应先处理已确认的数据/错误模式，再用小范围受控设置与独立验证比较，而不是预设多训两轮有效。该重训尚未启动。

dev100用于开发和checkpoint诊断，不是独立最终测试；也不与Process开发对照题集混为同一题集。原始400条均有完整本地证据、答案与文字审核；以下仅公开逐题诊断摘要。


## Final-only loss

固定输入为x，审核gold答案为y，只监督答案与EOS。每个effective batch按有效目标token总数归一：

```math
\mathcal L_F(B)=\frac{\sum_{i\in B}\sum_{t\in T_i}-\log\pi_{\theta_F}(y_{it}\mid x_i,y_{i,\lt t})}{\sum_{i\in B}|T_i|}.
```

输入、证据、State、工具Observation全部mask。mask决定哪些位置产生监督，不等于模型无法读取输入；因果移位、模板和adapter路径的审计结论见上文，不能把loss mask当作已证实的退化原因。

## 逐题材料

全部100题×四组的分数、优点和具体问题可从[逐题索引](records.md#final-comparison)进入；机器可读[400条评分](experiments/final_sft_dev100/scores_400.csv)及[汇总JSON](experiments/final_sft_dev100/summary.json)保留原值。缺失/失败答案没有补零，候选诊断分与有效答案分分列。

**下一章：[冻结Base Final后的Process-GRPO](grpo.md)。**
