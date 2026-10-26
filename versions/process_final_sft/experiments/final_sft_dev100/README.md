# Final-SFT与Base Final：训练、dev100评分与暂不加训的依据

[版本三](../../README.md) · [后续Process-GRPO](../process_grpo_20261026/README.md) · [400条分数](scores_400.csv) · [汇总JSON](summary.json)

## 1. 训练与对照设置

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

## 2. 评分规则

对原问题和实际可见证据，证据利用40%、问题相关性30%、忠实性20%、不确定性校准10%；各维0–4，按适用权重归一为百分制。没有已打开证据时证据利用记不适用，其他维仍评价是否切题、诚实回答。长度、引用数量和接近gold字面不直接计分。

原300条SFT答案按隐藏标签审阅；Base为后补评，旧300条未改分，因此不是四组同时完全盲评。失败保留原始候选及诊断分，有效答案分留空；不是按官方零分处理。分数为助手证据审阅，不是医学准确率或官方HealthBench分。

## 3. 全量结果

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

## 4. 为什么没有直接再加两轮

本轮三个SFT checkpoint均低于Base，尚无证据证明继续增加epoch就能超过Base。1→1.5→2轮评分实际上升，因此也不能据此宣布“已经证实过拟合”或“小模型必然被微调破坏”。更合适的决策是保留三个checkpoint和逐题材料，暂停盲目加训，以表现更好的Base Final作为固定生成器，先研究Process交付的信息是否改善。

已审计全部500条token边界：输入labels=-100，只监督真实答案和EOS；原生非thinking模板逐token一致，没有把Process adapter误挂为Final。合成微型模型的实际loss/梯度与标准因果CE一致；checkpoint哈希彼此不同。随后服务器配置/哈希回执无异常。这些排除所检查的明显工程错配，不证明数值路径和训练内容完全无问题。

600个问题均无中文；训练gold496英文、4中文，dev99英文、1中文。少量语言不一致值得整理，但不能用0.8%的训练中文解释整体下降。额外6条训练gold证据抽核未发现共同明显标签错误，不是全500题的独立医学认证。

目前不能确认下降由学习率、样本分布、答案风格、gold质量或其他因素中的哪项主导。assistant-only loss仍可学习利用输入，不能把不监督输入token解释成“没有学习理解输入”；不因此改成预测工具全文。

## 5. 当前选择与后续边界

后续本轮GRPO统一使用冻结Base Final，更新对象仅Process LoRA；没有使用Final-SFT结果来宣称Final训练成功。若重启Final-SFT，应先处理已确认的数据/错误模式，再用小范围受控设置与独立验证比较，而不是预设多训两轮有效。该重训尚未启动。

dev100用于开发和checkpoint诊断，不是独立最终测试；也不与旧HealthBench50混为同一题集。原始400条均有完整本地证据、答案与文字审核；以下仅公开逐题诊断摘要。

## 6. 全100题逐题评分与问题

- [Dev001：四组分数、优点与具体问题](questions/001.md)
- [Dev002：四组分数、优点与具体问题](questions/002.md)
- [Dev003：四组分数、优点与具体问题](questions/003.md)
- [Dev004：四组分数、优点与具体问题](questions/004.md)
- [Dev005：四组分数、优点与具体问题](questions/005.md)
- [Dev006：四组分数、优点与具体问题](questions/006.md)
- [Dev007：四组分数、优点与具体问题](questions/007.md)
- [Dev008：四组分数、优点与具体问题](questions/008.md)
- [Dev009：四组分数、优点与具体问题](questions/009.md)
- [Dev010：四组分数、优点与具体问题](questions/010.md)
- [Dev011：四组分数、优点与具体问题](questions/011.md)
- [Dev012：四组分数、优点与具体问题](questions/012.md)
- [Dev013：四组分数、优点与具体问题](questions/013.md)
- [Dev014：四组分数、优点与具体问题](questions/014.md)
- [Dev015：四组分数、优点与具体问题](questions/015.md)
- [Dev016：四组分数、优点与具体问题](questions/016.md)
- [Dev017：四组分数、优点与具体问题](questions/017.md)
- [Dev018：四组分数、优点与具体问题](questions/018.md)
- [Dev019：四组分数、优点与具体问题](questions/019.md)
- [Dev020：四组分数、优点与具体问题](questions/020.md)
- [Dev021：四组分数、优点与具体问题](questions/021.md)
- [Dev022：四组分数、优点与具体问题](questions/022.md)
- [Dev023：四组分数、优点与具体问题](questions/023.md)
- [Dev024：四组分数、优点与具体问题](questions/024.md)
- [Dev025：四组分数、优点与具体问题](questions/025.md)
- [Dev026：四组分数、优点与具体问题](questions/026.md)
- [Dev027：四组分数、优点与具体问题](questions/027.md)
- [Dev028：四组分数、优点与具体问题](questions/028.md)
- [Dev029：四组分数、优点与具体问题](questions/029.md)
- [Dev030：四组分数、优点与具体问题](questions/030.md)
- [Dev031：四组分数、优点与具体问题](questions/031.md)
- [Dev032：四组分数、优点与具体问题](questions/032.md)
- [Dev033：四组分数、优点与具体问题](questions/033.md)
- [Dev034：四组分数、优点与具体问题](questions/034.md)
- [Dev035：四组分数、优点与具体问题](questions/035.md)
- [Dev036：四组分数、优点与具体问题](questions/036.md)
- [Dev037：四组分数、优点与具体问题](questions/037.md)
- [Dev038：四组分数、优点与具体问题](questions/038.md)
- [Dev039：四组分数、优点与具体问题](questions/039.md)
- [Dev040：四组分数、优点与具体问题](questions/040.md)
- [Dev041：四组分数、优点与具体问题](questions/041.md)
- [Dev042：四组分数、优点与具体问题](questions/042.md)
- [Dev043：四组分数、优点与具体问题](questions/043.md)
- [Dev044：四组分数、优点与具体问题](questions/044.md)
- [Dev045：四组分数、优点与具体问题](questions/045.md)
- [Dev046：四组分数、优点与具体问题](questions/046.md)
- [Dev047：四组分数、优点与具体问题](questions/047.md)
- [Dev048：四组分数、优点与具体问题](questions/048.md)
- [Dev049：四组分数、优点与具体问题](questions/049.md)
- [Dev050：四组分数、优点与具体问题](questions/050.md)
- [Dev051：四组分数、优点与具体问题](questions/051.md)
- [Dev052：四组分数、优点与具体问题](questions/052.md)
- [Dev053：四组分数、优点与具体问题](questions/053.md)
- [Dev054：四组分数、优点与具体问题](questions/054.md)
- [Dev055：四组分数、优点与具体问题](questions/055.md)
- [Dev056：四组分数、优点与具体问题](questions/056.md)
- [Dev057：四组分数、优点与具体问题](questions/057.md)
- [Dev058：四组分数、优点与具体问题](questions/058.md)
- [Dev059：四组分数、优点与具体问题](questions/059.md)
- [Dev060：四组分数、优点与具体问题](questions/060.md)
- [Dev061：四组分数、优点与具体问题](questions/061.md)
- [Dev062：四组分数、优点与具体问题](questions/062.md)
- [Dev063：四组分数、优点与具体问题](questions/063.md)
- [Dev064：四组分数、优点与具体问题](questions/064.md)
- [Dev065：四组分数、优点与具体问题](questions/065.md)
- [Dev066：四组分数、优点与具体问题](questions/066.md)
- [Dev067：四组分数、优点与具体问题](questions/067.md)
- [Dev068：四组分数、优点与具体问题](questions/068.md)
- [Dev069：四组分数、优点与具体问题](questions/069.md)
- [Dev070：四组分数、优点与具体问题](questions/070.md)
- [Dev071：四组分数、优点与具体问题](questions/071.md)
- [Dev072：四组分数、优点与具体问题](questions/072.md)
- [Dev073：四组分数、优点与具体问题](questions/073.md)
- [Dev074：四组分数、优点与具体问题](questions/074.md)
- [Dev075：四组分数、优点与具体问题](questions/075.md)
- [Dev076：四组分数、优点与具体问题](questions/076.md)
- [Dev077：四组分数、优点与具体问题](questions/077.md)
- [Dev078：四组分数、优点与具体问题](questions/078.md)
- [Dev079：四组分数、优点与具体问题](questions/079.md)
- [Dev080：四组分数、优点与具体问题](questions/080.md)
- [Dev081：四组分数、优点与具体问题](questions/081.md)
- [Dev082：四组分数、优点与具体问题](questions/082.md)
- [Dev083：四组分数、优点与具体问题](questions/083.md)
- [Dev084：四组分数、优点与具体问题](questions/084.md)
- [Dev085：四组分数、优点与具体问题](questions/085.md)
- [Dev086：四组分数、优点与具体问题](questions/086.md)
- [Dev087：四组分数、优点与具体问题](questions/087.md)
- [Dev088：四组分数、优点与具体问题](questions/088.md)
- [Dev089：四组分数、优点与具体问题](questions/089.md)
- [Dev090：四组分数、优点与具体问题](questions/090.md)
- [Dev091：四组分数、优点与具体问题](questions/091.md)
- [Dev092：四组分数、优点与具体问题](questions/092.md)
- [Dev093：四组分数、优点与具体问题](questions/093.md)
- [Dev094：四组分数、优点与具体问题](questions/094.md)
- [Dev095：四组分数、优点与具体问题](questions/095.md)
- [Dev096：四组分数、优点与具体问题](questions/096.md)
- [Dev097：四组分数、优点与具体问题](questions/097.md)
- [Dev098：四组分数、优点与具体问题](questions/098.md)
- [Dev099：四组分数、优点与具体问题](questions/099.md)
- [Dev100：四组分数、优点与具体问题](questions/100.md)
