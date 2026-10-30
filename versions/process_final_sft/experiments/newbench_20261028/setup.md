# 跨基准验证评测 · HealthBench50＋MedBrowseComp50：设置与题目来源

[跨基准验证评测 · HealthBench50＋MedBrowseComp50](README.md) · [名词与指标解释](../../glossary.md) · [源码](code/README.md)

|项目|冻结设置|
|---|---|
|HealthBench|从固定官方5000题源文件中按固定哈希取50道可独立回答的单用户英文题，保留原题及医生rubric；自定义子集|
|MedBrowseComp|从固定605题文件选50题，五类任务各10题、每个NCT仅一次；不是官方MedBrowseComp_50切分|
|已知池排重|原972、600题、GRPO100和Process开发对照题池；ID、规范化文本和词面近似筛查；不证明无预训练/语义污染|
|四组|Base关闭adapter；SFT为原972 checkpoint1322；GRPO为第13/25批；四组都重新运行|
|底座|Qwen/Qwen3-8B；4-bit NF4 double quant，BF16计算；同一冻结FP32 Process输出头|
|Final/压缩|关闭Process adapter，恢复原输出头，四组同配置；Final不训练|
|Process采样|temperature1、top_p1、top_k0、单beam、repetition_penalty1；保存effective参数|
|约束|相同Prompt、检索、6次正常计费工具动作；访问受阻/环境故障最多5次豁免，实际调用最多11次；Search/Browse共享，不是6轮搜索加阅读。上下文软12288/硬15000；不为某组加预算|
|输出预算|Checklist1200、Decision400、State1200、Final2400|
|顺序|每5题为块轮换四组与先后顺序，减小时间漂移；实时网页仍可能变化|
|资源|8逻辑CPU、检索4线程，1GPU；评测不做训练概率重放、backward或optimizer|
|裁判|qwen3.7-max替代Judge；不向Agent传rubric/reference/source_row；Judge隐藏arm|

本评测题目在本轮生成前选定，不按生成成功与否筛题。400条中387条正常完成、10条Process失败、3条Final失败，缺失/中断/工程失败为0；完成的387条均有答案评分。此次结果现已用于分析，今后不能再称未接触测试。下一轮采集或训练未在此启动。

来源：[HealthBench官方代码](https://github.com/openai/simple-evals/blob/main/healthbench_eval.py)、[MedBrowseComp数据卡](https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp)、[作者处理脚本](https://github.com/shan23chen/MedBrowseComp/blob/main/process_NCT_predictions.py)。固定源文件、runtime和私有归档SHA在summary/SOURCE_MANIFEST中保留。对原题的字母、日期截止及整体批准限定不作有利于某组的改写。

## 可复核的选题合同

源文件地址、revision、SHA256和全部100个原始题号见[选题身份清单](selection_manifest.json)，选择实现见[prepare_data.py](code/eval/prepare_data.py)。身份清单不含gold；实际评测原题另见[100题原文](questions_100.json)，并置于每个逐题页评分表之前。

排序键为固定seed `newbench-20261026-v1` 与来源ID/行身份的SHA256；它不是按本次模型输出或分数排序。HealthBench仅保留一条user消息、至少12个英文词、ASCII比例至少97%、常见英文词比例至少12%的候选，再排重取50题；此启发式筛选不等于代表全部HealthBench。MedBrowseComp按五类各10题、每个NCT只保留一题，保留原题的日期和字母限制。

已知池比较来源ID与NFKC/casefold后词序规范化文本；近似检查同时要求文本长度至少80、长度比至少0.8、词集合Jaccard至少0.65以及SequenceMatcher至少0.85。相似语义或模型预训练污染仍可能遗漏。私有排除池保留哈希/规模供核验，不公开训练题全文。
