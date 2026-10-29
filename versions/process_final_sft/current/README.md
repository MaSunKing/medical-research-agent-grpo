# 版本三源码入口：运行组件、训练器与实验快照

[版本三说明](../README.md) · [首页](../../../README.md) · [源码身份清单](SOURCE_MANIFEST.json)

`current`是版本三维护源码的位置，不指定正在使用哪一个Process checkpoint。部署时仍需显式绑定Base、Process-SFT、GRPO13或GRPO25及对应配置；特定实验按各自冻结执行快照复核。

本目录覆盖LangGraph节点、MCP桥接、上下文、重复Search缓存、候选可用性、正文保护、worker来源恢复及相关测试。完整运行说明见[系统架构与执行机制](../engineering.md)，训练方法见[Process-SFT](../process_sft.md)与[Process-GRPO](../grpo.md)。**不附私有训练题、完整轨迹、权重、凭据、私有配置或运行日志。**

<a id="目录"></a>

## 先从哪个文件读

理解Agent怎样运行，先读`engine.py`；理解采集流程，先读`run_one.py`，再看批量入口。Engine位于runtime子目录，采集、服务和导出脚本直接位于current目录。

| 想看什么 | 直接打开 |
|---|---|
| 完整节点流程与条件路由 | [runtime/src/trainable_agent/engine.py](runtime/src/trainable_agent/engine.py) |
| 单题Process轨迹采集 | [run_one.py](run_one.py) |
| 保存实际Final请求的固定输入快照 | [freeze_final_input.py](freeze_final_input.py) |
| 模型服务与角色参数控制 | [serve_policy.py](serve_policy.py) |
| 批量采集与续跑队列 | [collect.py](collect.py) |
| 准备题集、配置与工作目录 | [prepare.py](prepare.py) |
| 一致性检查与结果导出 | [export_results.py](export_results.py) |

## 按用途选择入口

### 运行组件与检索

| 路径 | 内容 |
|---|---|
| [runtime/src/trainable_agent](runtime/src/trainable_agent) | Engine节点、动作/State合同、阶段Prompt、缓存、JSON、上下文、存储与导出 |
| [runtime/retrieval_snapshot_v2](runtime/retrieval_snapshot_v2) | Search、Browse、清洗、排序、PDF、证据坐标与来源恢复；99文件清单及共享模块绑定 |
| [runtime/tests](runtime/tests) | 当前生产合同回归；不包含旧拒绝式训练数据构建器的全仓库验收承诺 |
| [runtime/scripts](runtime/scripts) | 源码核验、运行环境预检和显式恢复工具 |

### Process参数训练主线

| 入口 | 初始化与职责 |
|---|---|
| [original972_training](original972_training) | 新建Process LoRA训练原始972阶段样本；包括effective-batch token loss、GPU验收与7项CPU/toy测试 |
| [process_grpo_update](process_grpo_update/README.md) | 从原972 Process adapter继续更新；轨迹奖励、截帽重要性校正、PPO式裁剪、确定性更新及断点恢复 |

Final-SFT是独立答案模块对照，不是上述两项之间的训练步骤。其数据、参数与结论见[Final-SFT章节](../final_sft.md)。

<a id="跨基准验证评测对应的源码"></a>

### 特定实验的冻结执行源码

跨基准四组benchmark的执行源码、评测/Judge适配及合成测试独立保存在[评测快照](../experiments/newbench_20261028/code/README.md)，结果见[跨基准验证评测 · HealthBench50＋MedBrowseComp50](../experiments/newbench_20261028/README.md)。复现使用对应源码清单、模型和运行配置；维护目录名不能替代实验身份。

### 历史实现与全仓库导航

根目录既有实现及版本一、二保留各自适用身份，不作为当前默认入口。需要定位它们时见[代码导航](../../../docs/code_navigation.md)；版本三的数据与运行身份见[数据与实验设置](../design.md)。

## 离线检查

要求Python3.11+，仅安装源代码和解析测试依赖；不自动下载模型：

```bash
python -m pip install -e './runtime[retrieval_html]'
python -B check_public_source.py --tests
```

从仓库根目录执行时，将runtime和脚本路径加上本目录前缀。原始972 CPU/toy测试另需PyTorch：

```bash
cd original972_training
python -B test_training.py
```

以上不加载Qwen、调用外部检索API或启动正式训练；本地/CI fixture成功不代替GPU最长样本、断点一致性、真实检索和医学审核。

## 部署数据仍须自行提供

实际采集和训练需要自行准备：

- 题目与划分文件`QUESTIONS`、`SPLIT`，以及与它们对应的数据身份；
- 自有基座、adapter路径和模型身份，以及实际使用的检索凭据；
- 与所选源码一致的运行配置、数据／部署manifest及工作目录。

这些私有材料未随源码附带，因此本目录不是可直接运行的完整600题服务器包。部署时需显式构建并核验自己的包，不能将发布源码与旧私有FILES清单混合使用或绕过身份校验。

**采集入口的特定绑定：** 本目录`prepare.py`要求显式传入`--process-adapter`，并校验原972 checkpoint1322及对应身份；这是该Process-only采集入口的约束，不表示所有GRPO运行入口都固定加载checkpoint1322。原972训练的`run_phase.sh`仍要求显式设置`PROCESS972_BASE/PYTHON`。

## 配置、身份与机制分别在哪里

| 需要核对什么 | 说明位置 |
|---|---|
| 模型、量化、题集、工具与上下文预算 | [数据与实验设置](../design.md)；具体实验以自身冻结配置为准 |
| 缓存、正文清洗、reading_focus、来源恢复和交接模式 | [系统架构与执行机制](../engineering.md) |
| 文件SHA、UTF-8/LF规范化、路径适配及共享manifest绑定 | [源码身份清单](SOURCE_MANIFEST.json)；部署需同时绑定完整模型、数据、配置和源码清单 |
| GRPO更新包所需的轨迹、奖励、权重与恢复材料 | [更新器部署输入与适用范围](process_grpo_update/README.md#部署输入与适用范围) |

维护源码身份与实验执行身份分别保留；Teacher完整采集源码不在这里补造，原训练保持真实messages身份。源码与合成测试用于工程审阅，不表示已附私有数据或完成医学质量认证。
