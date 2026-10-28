> 构建时部署说明的源码快照；本轮评测现已完成，结果见[实验汇总](../../README.md)。公开目录不附私有数据、模型和完整封印；以下执行步骤需用户自行绑定私有部署包。

# 两个新测试集 × 四组 Process 评测

HealthBench 50 + MedBrowseComp 50；Base、原972 Process-SFT、GRPO13、GRPO25 各自重新运行，合计400条轨迹。四组都使用冻结 Base Final。旧50题的历史轨迹和分数不作为本次结果。

## 题集与比较边界

- HealthBench：固定官方5000题源文件，从新的单用户轮次英文候选中按固定哈希顺序取50题；保留原文、prompt_id、官方医生rubrics与正负权重。这是自定义单轮子集，不代表完整HealthBench或多轮表现。
- MedBrowseComp：固定官方HF revision，从605题中每类任务10题、每个NCT试验只选1题，共50个试验；不是官方MedBrowseComp_50切分。原题中的时间截止（含Dec 2024）必须保留，不能用2026最新状态替换金标语境。CSV单元格原文保留，不隐式解码字面反斜杠换行。
- 排重记录在 `data/KNOWN_EXCLUSIONS.json` 与 `SELECTION.json`：核对本地incoming/outgoing明确题目注册表，涵盖原972、600题、GRPO100及旧评测池；ID、NFKC/大小写/标点/空白规范化、词面近似排重。它不证明预训练无污染或所有语义改写绝不重叠。
- 无生成结果参与选题。所有参考答案、评分条目、来源记录只供离线评分；引擎只接收 `question`，不传gold/rubrics/reference/source_row。

## 模型与运行合同

模型底座为Qwen3-8B，服务器按原NF4 double-quant / BF16计算加载；LoRA为FP32。Base关闭adapter，SFT加载checkpoint1322，GRPO分别加载checkpoint0013与0025。四组均使用同一冻结且未合并的FP32 Process输出头；Final与阅读压缩调用关闭adapter、使用原BF16输出头。权重不放入本包，服务器核对底座清单、原972哈希、训练谱系、训练源码/参数和各checkpoint封印。

运行代码来自已核验冻结快照，不吸收本地脏工作树的检索/prompt变化。四组使用相同的检索、完整原题、Checklist/Decision/State/Final提示、软上下文12288/硬15000、普通工具预算6与环境豁免5、无效决策6、压缩调用12；Checklist1200/Decision400/State1200/Final2400。paired seed按题目和真实调用身份派生。Process使用T=1/top_p=1/top_k=0/单beam/重复惩罚1；冻结Final和压缩使用冻结native.py规定的相同参数，真实effective参数归档。不因题目难而扩大某一组预算。

按5题一块轮换四组，块间轮换模型先后顺序，缩短同题组间的运行时间差；不先跑完100题Base再跑其余三组。实时网页仍可能变化，同配置不代表每组获得完全相同的网页结果；实际回执和运行时间均保留。四组adapter哈希另外与上一轮已验证导出中的精确checkpoint绑定，不能仅凭目录名或step号替换。

GPU任务8逻辑CPU、内存28G、1GPU、5:45，检索torch4线程、冻结历史rank2线程；限定原验收A40节点类型所在TC2N01/02，节点由Slurm选择。无概率重放或训练backward；本次是评测而非采集GRPO训练样本。

## 服务器评分与本地评分

服务器使用用户现有DashScope `qwen3.7-max`，不随包携带key。HealthBench沿用已固定的官方grader模板和加权计算；每个rubric逐项判定并保留说明、请求、响应、usage。每题可负分，先求均值再截负值。MedBrowseComp用单独的语义等价判据对照参考答案，正确1/错误0；不重新求解问题、不因引用/大小写等外观差异否定明确正确答案。其提示和JSON接口是本项目适配，非逐字官方提示。两者都标记替代judge，非官方榜单。

同一问题四组共用rubrics或reference；Judge看不到arm名称。失败、缺失、未完成评分保留null，不伪装成官方零分。汇总各benchmark的完成率、有效评分数、有效答案均分/准确率、全计划正确交付率（MedBrowseComp工程指标）、同题配对差与配对bootstrap区间，不将两种分数平均。单次运行的bootstrap不代表重复运行波动。

导出所有400格（含Process失败/Final失败/工程失败/缺失）、SQLite、真实请求/工具回执、原始Pre-Final、真正发给Final的首个输入、Final答案与评分回执。`PRE_FINAL_PACKAGE.json`不能替代实际Final输入。导出生成400条待审核模板；本地继续按既有Process规则逐条评内容交付、证据可靠性、检索与控制，保留具体问题和扣分理由。每题内容点先依据原题及来源统一批准，再比较四组；模板没有自动医学评分，也没有冒充已批准内容点。

## 服务器执行顺序

所有计算均在Slurm计算节点执行，下面的submit命令从登录节点提交。CPU和GPU任务不在计算节点内部再次sbatch。原始构建时未执行；实际本轮已完成生成和评分，未训练。

1. 安全解压及SHA256检查后，把包路径记为 `PKG`，包外专用目录记为 `WORK`；复制并核对 `SERVER_CONFIG.example.json` 到包外 `CONFIG`。不得修改已封印包。
2. CPU验证及绑定（读取外部现有模型文件，不加载模型）：

```bash
bash "$PKG/eval/submit.sh" cpu "$PKG" "$WORK" "$PY" "$CONFIG"
```

3. 工程试跑：两个benchmark的首题，四组共8条，随后核验封印并写 `PILOT_ACCEPTANCE.json`。已有试跑结果留给正式评测，不重复生成。

```bash
bash "$PKG/eval/submit.sh" smoke "$PKG" "$WORK" "$PY"
```

4. 试跑通过后，从同一个WORK开始/续跑全量。仅跳过已封印记录（含模型失败）；未封印目录必须先审查，不覆盖、不默认重抽。每个分配接近时限时保留断点后退出；400条通常需要多个5:45分配，不保证两个分配结束。

```bash
bash "$PKG/eval/submit.sh" run "$PKG" "$WORK" "$PY"
```

5. 全量完成后，在CPU任务中开启潜在付费评分。外部secrets路径应为用户已有文件，权限不变，不复制到结果目录。Judge默认一次失败重试；未知结果的旧请求要求显式核对，不自动重复扣费。API验收可先用下面命令对8条试跑评分，然后正式评分从缓存接着跑。

```bash
export EVAL_JUDGE_SECRETS=/secure/path/secrets.env
bash "$PKG/eval/submit.sh" grade-smoke "$PKG" "$WORK" "$PY" --allow-judge-api
bash "$PKG/eval/submit.sh" grade "$PKG" "$WORK" "$PY" --allow-judge-api
```

6. 结果导出（新目录）：

```bash
bash "$PKG/eval/submit.sh" export "$PKG" "$WORK" "$PY" "$EXPORT_DIR"
```

submit.sh每次输出该任务的独立 `watch -n 60` 指令，含精确JobID、最后保存的阶段/计数、最近4行进展和2行错误。Ctrl+C仅退出watch，不取消任务。不要在登录节点运行递归统计、模型、评分或打包。不得把本包数据或私密轨迹推到公开GitHub。

## 来源

- HealthBench: https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
- 固定原题源地址及SHA：`data/SOURCES.json`
- MedBrowseComp: https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp
- 论文: https://arxiv.org/abs/2505.14963
