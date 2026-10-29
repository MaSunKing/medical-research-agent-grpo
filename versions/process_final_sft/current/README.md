# 版本三源码入口：运行组件、训练器与实验快照

[版本三说明](../README.md) · [首页](../../../README.md) · [源码身份清单](SOURCE_MANIFEST.json)

`current`是版本三维护源码的位置，不指定正在使用哪一个Process checkpoint。部署时仍需显式绑定Base、Process-SFT、GRPO13或GRPO25及对应配置；特定实验按各自冻结执行快照复核。

本目录覆盖LangGraph节点、MCP桥接、上下文、重复Search缓存、候选可用性、正文保护、worker来源恢复及相关测试。完整运行说明见[系统架构与执行机制](../engineering.md)，训练方法见[Process-SFT](../process_sft.md)与[Process-GRPO](../grpo.md)。**不附私有训练题、完整轨迹、权重、凭据、私有配置或运行日志。**

<a id="目录"></a>

## 按用途选择入口

### 运行组件与检索

| 路径 | 内容 |
|---|---|
| [runtime/src/trainable_agent](runtime/src/trainable_agent) | Engine节点、动作/State合同、阶段Prompt、缓存、JSON、上下文、存储与导出 |
| [runtime/retrieval_snapshot_v2](runtime/retrieval_snapshot_v2) | Search、Browse、清洗、排序、PDF、证据坐标与来源恢复；99文件清单及共享模块绑定 |
| [runtime/tests](runtime/tests) | 当前生产合同回归；不包含旧拒绝式训练数据构建器的全仓库验收承诺 |
| [runtime/scripts](runtime/scripts) | 源码核验、运行环境预检和显式恢复工具 |
| 本目录Python/Shell入口 | Process-only队列、Final输入冻结、配置准备、状态查看、SQLite一致性导出与Slurm入口 |

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

collect/prepare/训练入口依赖私有QUESTIONS、SPLIT、模型身份、数据/部署manifest和自有模型路径，本目录故意不附这些文件。公开源码不是可直接运行的完整600题服务器包；不能用缺失或自造数据伪装成功，也不能混合发布源码和旧私有FILES清单后绕过校验。需显式重新构建并核验部署包。

部署路径适配：prepare.py要求显式传入--process-adapter，仍校验checkpoint1322及对应身份；原972训练run_phase.sh要求显式设置PROCESS972_BASE/PYTHON。源码统一UTF-8/LF，检索共享manifest绑定公开字节；逐文件原始/公开SHA及适配说明见SOURCE_MANIFEST。当前源码身份与实验执行身份分别保存，部署应绑定自己的完整模型、数据、配置和源码清单。

当前合同：Search原题/query为60:40；重复Search读本题缓存并计正常动作；未读优先、部分已读后置，失败/耗尽按作用域隐藏；6+5工具预算；软12,288/硬15,000；模型抽取最后兜底。Final输入冻结为独立步骤，Process-only不生成Final/Judge；原972训练与600题采集不共享新训练目标或权重更新。

正文清洗保留弱标题规则，在排除模板噪声时保护临床短句、剂量、否定比较、中文条目和表格。worker启动及Search回执重放时恢复已经校验的来源映射，不增加Search、不改canonical ID；只能恢复回执实际保存的字段。reading_focus的检索投影每次独立按320个UTF-8字节处理，原题锚点为640字节，完整原题和证据正文另有预算。

Teacher完整采集源码不在本目录补造；原训练保持真实messages身份。源码与合成测试不等于私有数据已附带或医学质量已获批准。运行模式、更新对象和输入边界见[完整设计](../engineering.md)。
