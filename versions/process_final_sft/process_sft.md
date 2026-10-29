# 2. Process-SFT

[概览](README.md) · [数据与实验设置](design.md) · **Process-SFT** · [Final-SFT](final_sft.md) · [GRPO](grpo.md) · [评测](evaluation.md) · [逐题记录](records.md)

[名词与指标解释](glossary.md)

Process-SFT是参数训练主线的起点，产物继续用于Process-GRPO；Final-SFT另作独立答案模块对照。

| 训练边界 | 本页对象 |
|---|---|
| 输入 | 972题Teacher真实轨迹拆成的10575条合法阶段样本 |
| 初始化 | 冻结`Qwen/Qwen3-8B`基座，新建Process LoRA |
| 更新对象 | 仅Process LoRA；监督Checklist、Decision、State的当前completion，Final不作为训练目标 |
| 产物 | 原972 Process checkpoint1322，作为后续Process-GRPO初始化 |

### 轨迹拆分

一条轨迹按当前阶段拆分为：原题 + Compact History + 当前Checklist/State + 候选或已读证据 → 当前阶段completion。

- Checklist：监督独立需求与共享范围。
- Decision：监督具体Search query、Browse来源ID及阅读重点，不只监督动作类别。
- State：监督有证据来源的状态更新与未解决缺口。
- Stop：属于Decision的合法停止动作；正常工具预算耗尽时由程序结束，不伪造模型Stop目标。
- Final：不作为Process target，不进入更早阶段历史。

输入、历史和工具Observation全部mask，仅当前合法completion计算CE。拒绝尝试保留反馈与因果记录，不自动作为合格正监督。原始训练使用当时真实可见输入；当前推理与导出共用阶段构建器，但不能宣称已与旧Teacher完整源码、预算和候选处理逐字一致。

### 原始972已完成训练参数

本表描述生成checkpoint1322的那次Process训练，不是当前在线上下文配置，也不是后续Final训练参数。训练时从新初始化LoRA开始；当前运行加载已经训练完成的adapter，不重新初始化。

| 项目 | 配置 |
|---|---|
| Base / adapter初始化 | `Qwen/Qwen3-8B`；训练开始时新初始化Process LoRA，`initial_adapter=null` |
| 题目与阶段数 | 972题；Checklist972、Decision6143、State3460，共10575条 |
| Epoch / batch | 1轮；microbatch1；accum8 |
| LoRA | r32、alpha64、dropout0.05 |
| LoRA注入层 | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj |
| LR / optimizer | 1e-4；AdamW；warmup67更新后cosine；weight decay0.01，grad clip1.0 |
| 优化器细项 / 随机种子 | AdamW betas=(0.9, 0.999)、eps=1e-8；seed=42 |
| 量化 / 计算精度 | bitsandbytes 4-bit NF4，double quant=True，BF16 compute；基座冻结，FP32 LoRA更新 |
| 训练attention / 显存策略 | Transformers SDPA，强制PyTorch Flash SDP后端并启用确定性算法；梯度检查点，训练use_cache=False；不是另行配置FlashAttention-2 |
| 原训练预算 | 10,240；最长编码样本9,762 tokens，不用当前推理预算追改训练身份 |
| Loss | effective batch内target-token mean；Checklist/Decision/State权重均1 |
| 更新步数 | ceil(10575/8)=1322；最后一批7条 |
| 验证集合 | 原训练包972题全部用于训练，没有包内dev；不声称已有独立验证损失 |
| Final / Pre-Final目标 | Final目标0；Pre-Final是程序输出，不计算loss |

原始972训练按冻结训练包的数据、模型和源码哈希核验。[实际原始972训练器](current/original972_training/train.py)、[token loss](current/original972_training/training_core.py)及[配置](current/original972_training/TRAIN_CONFIG.json)已公开；不附原题、编码数据或权重。本目录[既有Process脚本](process/train.py)保持原身份，不再作为972训练入口。当前推理改动与训练样本保持独立身份。

### 运行配置的身份

原始972训练参数维持上述历史身份。600题Pre-final使用当时公开current服务；新GRPO13/25评测使用独立原生采样入口与FP32 Process输出头，不能把current/serve_policy.py冒充新GRPO入口。两者提示词、检索、预算设计相同，具体差异见[运行配置对照](experiments/process_grpo_20261026/runtime.md)。

### Loss公式

设effective batch为 $B$，样本 $i$ 的合法当前target位置为 $T_i$，token交叉熵为 $\ell_{it}$。原始972各阶段权重均为1：

```math
\ell_{it}=-\log\pi_{\theta_P}(y_{it}\mid x_i,y_{i,\lt t}).
```

批次损失为：

```math
\mathcal L_P(B)=\frac{\sum_{i\in B}\sum_{t\in T_i}\ell_{it}}{\sum_{i\in B}|T_i|}.
```

每个microbatch先计算当前target的CE总和，再除以该effective batch全部有效target-token数并累积梯度；最后不足8条按实际token数归一。不是各样本先取均值后等权，也不是按整个epoch的token量固定缩放。输入、工具、错误尝试和程序Pre-Final全部mask。模型可从历史中学习Search→Browse→State→再决策，但正确语法与历史可见不保证学到稳定策略。

### Process开发对照评测方法

在同题、matched seed、相同检索与工具预算下比较Base与Process-SFT，检查实际证据覆盖、State事实与绑定、缺口更新和停止行为。更新次数与链路长度单列，不作为质量扣分；不以Final写作差异代替Process质量。已完成结果见[统一评测结果](evaluation.md#process-development-results)，逐题记录见[记录索引](records.md)。


## 从SFT到后续训练

当前合法completion在输入历史条件下预测，mask输入并不切断模型对输入的利用。Process训练学习具体query、来源选择、State判断和Stop；工具正文是观察，不是要求模型复述的训练目标。

SFT在历史50题中交付覆盖55.83%，Base48.42%，Process模型失败2/50对6/50，因此成为后续采集起点。综合质量R没有超过Base，不能把它概括为所有能力都更好；State误读与证据利用是后续GRPO关注的问题。

**沿参数主线继续：[Process-GRPO](grpo.md)。** 答案模块的独立对照见[Final-SFT](final_sft.md)。
