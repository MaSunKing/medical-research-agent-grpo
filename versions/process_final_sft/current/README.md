# 版本三当前实现：节点式Process与最新检索

[版本三说明](../README.md) · [首页](../../../README.md) · [源码身份清单](SOURCE_MANIFEST.json)

本目录公开当前本地实现中与600题服务器冻结包一致的生产源码：LangGraph节点、MCP桥接、上下文、重复Search缓存、候选可用性、保守JSON修复、最新检索快照和相关测试。**没有题目、轨迹、权重、凭据、私有配置或运行日志。** 不覆盖版本一、二，也不改后续计划。

## 目录

| 路径 | 内容 |
|---|---|
| [runtime/src/trainable_agent](runtime/src/trainable_agent) | Engine节点、动作/State合同、阶段Prompt、缓存、JSON、上下文、存储与导出 |
| [runtime/retrieval_snapshot_v2](runtime/retrieval_snapshot_v2) | 最新完整工具快照：Search、Browse、清洗、排序、PDF与证据坐标；98文件清单及共享模块绑定 |
| [runtime/tests](runtime/tests) | 当前生产合同回归；不包含旧拒绝式训练数据构建器的全仓库验收承诺 |
| [runtime/scripts](runtime/scripts) | 源码核验、运行环境预检和显式恢复工具 |
| [original972_training](original972_training) | 原始972训练器、effective-batch token loss、GPU验收与7项CPU/toy测试 |
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

部署路径适配：prepare.py移除个人服务器硬编码地址，要求显式传入--process-adapter，仍校验checkpoint1322及对应身份；原972训练run_phase.sh要求显式设置PROCESS972_BASE/PYTHON，不再提供个人路径默认值。源码统一UTF-8/LF，检索共享manifest据此重绑。其他生产Python保持相同语法树，不改变Prompt、工具、排序、预算或模型行为；逐文件原始/公开SHA及变更说明见SOURCE_MANIFEST。

当前合同：Search原题/query为60:40；重复Search读本题缓存并计正常动作；未读优先、部分已读后置，失败/耗尽按作用域隐藏；6+5工具预算；软12,288/硬15,000；模型抽取最后兜底。Final输入冻结为独立步骤，Process-only不生成Final/Judge；原972训练与600题采集不共享新训练目标或权重更新。

Teacher旧完整源码未被本目录补造；原训练保持真实messages身份。已公开训练包的7项测试不等于原数据已附带或当前金标已合格。后续Final训练与GRPO参数本次不调整。

## 新100题评测对应的源码

本目录仍保留原600题时期的实现身份。最新四组benchmark运行使用单独冻结原生服务和FP32 Process头，源码、评测/Judge适配及合成测试见[新增评测源码](../experiments/newbench_20261028/code/README.md)，结果见[新增100题](../experiments/newbench_20261028/README.md)。不把本目录旧入口称作全部最新实验的执行入口。
