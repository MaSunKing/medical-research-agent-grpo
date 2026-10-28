# 版本三当前实现：节点式Process与最新检索

[版本三说明](../README.md) · [首页](../../../README.md) · [源码身份清单](SOURCE_MANIFEST.json)

本目录是版本三的当前生产源码入口：LangGraph节点、MCP桥接、上下文、重复Search缓存、候选可用性、正文保护、worker来源恢复及相关测试。各步骤的设计理由和适用限制见[逐步工程设计](../engineering.md)，奖励、概率校正与更新公式见[Process-GRPO](../grpo.md)。**不附私有训练题、完整轨迹、权重、凭据、私有配置或运行日志。**

## 目录

| 路径 | 内容 |
|---|---|
| [runtime/src/trainable_agent](runtime/src/trainable_agent) | Engine节点、动作/State合同、阶段Prompt、缓存、JSON、上下文、存储与导出 |
| [runtime/retrieval_snapshot_v2](runtime/retrieval_snapshot_v2) | Search、Browse、清洗、排序、PDF、证据坐标与来源恢复；99文件清单及共享模块绑定 |
| [runtime/tests](runtime/tests) | 当前生产合同回归；不包含旧拒绝式训练数据构建器的全仓库验收承诺 |
| [runtime/scripts](runtime/scripts) | 源码核验、运行环境预检和显式恢复工具 |
| [original972_training](original972_training) | 原始972训练器、effective-batch token loss、GPU验收与7项CPU/toy测试 |
| [process_grpo_update](process_grpo_update/README.md) | Process轨迹奖励、截帽重要性校正、PPO式裁剪、确定性更新及断点恢复源码 |
| 本目录Python/Shell入口 | Process-only队列、Final输入冻结、配置准备、状态查看、SQLite一致性导出与Slurm入口 |

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

## 新100题评测对应的源码

当前实现见本目录；四组benchmark的执行源码、评测/Judge适配及合成测试独立保存在[评测源码](../experiments/newbench_20261028/code/README.md)，结果见[新增100题](../experiments/newbench_20261028/README.md)。实验复现使用对应源码清单、模型和运行配置，不以“current”目录名替代实验身份。
