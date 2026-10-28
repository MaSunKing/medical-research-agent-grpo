# Process-GRPO：轨迹目标、概率校正与确定性更新源码

[算法与公式](../../grpo.md) · [当前源码](../README.md) · [逐文件身份](SOURCE_MANIFEST.json)

本目录公开版本三Process更新器的源码和配置。目标是将同题四条真实轨迹的相对质量，转成Checklist、Decision和State的策略更新；Final保持冻结。源码按明确allowlist导出，原始与公开文件SHA分别登记，UTF-8/LF规范化单独标明。

## 从奖励到参数更新

| 模块 | 可核验内容 |
|---|---|
| [CONFIG.json](training/process_grpo_update/CONFIG.json) | 单pass、4题/批、学习率1e-6、clip 0.2、校正权重上限2、ESS与数值门控 |
| [contract.py](training/process_grpo_update/contract.py) | 题组与调用身份、同题优势、数据/配置约束、失败与留置 |
| [objective.py](training/process_grpo_update/objective.py) | 生成/训练路径的截帽重要性校正、PPO式裁剪、轨迹token归一化及反向累计 |
| [runtime.py](training/process_grpo_update/runtime.py) | 严格确定性设置、模型加载、初始训练路径reference、梯度范围、完整断点保存和恢复 |
| [train.py](training/process_grpo_update/train.py) | CPU预检、8题32条GPU验收、25批更新循环、运行锁、暂停与续跑 |
| [verify_package.py](training/process_grpo_update/verify_package.py) | 真实私有部署包的数据与模型身份校验；不以公开源码检查替代 |
| [test_update.py](training/process_grpo_update/test_update.py) | 合成CPU测试：符号、clip、校正代数、归一化、梯度与保存恢复 |

三种概率分别登记：实际生成行为概率不改写；更新前的完整训练前向概率单独冻结；当前训练概率随LoRA变化。校正权重处理生成/训练路径差异，PPO比率处理策略变化。确定性执行与保存恢复检查处理另一类梯度重复性问题；完整推导和实际验收边界见[算法说明](../../grpo.md#4-三种概率各自负责什么)。

## 离线检查

从本目录执行源码检查：

```bash
python -B verify_public.py
```

安装PyTorch和safetensors后，可显式运行合成CPU测试：

```bash
python -B verify_public.py --tests
```

合成测试会对微型测试网络执行反向及toy optimizer步骤，用于检查数学和恢复合同；不加载Qwen、不更新真实adapter、不访问检索或Judge API。GPU恢复验收与真实数据验证仍有独立条件。

## 部署输入与适用范围

公开目录不含`data/`、题目、奖励表、真实token/logprob、reference、模型身份文件、adapter或optimizer断点。`train.py`和`verify_package.py`要求这些私有材料，不会因源码检查通过而自动生成或跳过它们。保留`training/process_grpo_update/`布局，使部署根目录与代码的路径合同一致；实际运行还需自己的完整数据清单、模型路径、计算节点和验收记录。

这是一轮固定行为数据上的GRPO风格离线更新：同题中心化而不除标准差，轨迹等权，单次遍历。截帽校正控制方差并引入近似，不保证任意策略偏移都可靠；新的策略轮次需要重新采集并重新验收。本目录没有自动提交任务、下载模型或继续训练的动作。
