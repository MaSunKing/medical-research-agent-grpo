> 历史单 LoRA 说明。新版项目介绍见 [首页](../README.md)，当前检索见 [检索设计](retrieval.md)，独立 Process / Final 见 [版本三](../versions/process_final_sft/README.md)。

# 本地友好的稠密奖励 GRPO Agent

这个项目把检索 Agent 的学习信号从“最终答案总分”拆成 Checklist、Search、Browse、State、Final 等局部通道。可信代码根据证据回执计算 evidence gain，绑定真实生成 token，再进行同题组内相对优势和共享 LoRA 更新。

保留原来的完整 chunk ID 和句后 `<cite>` 格式；代码检查格式、绑定和证据 ID，Judge 判断语义支持。代码不能单独证明引用正确。

公开版包含核心算法、工具后端、协议、训练器、合成例子和离线测试。不包含私有题集、密钥、模型权重、服务器路径、运行日志或集群专用采集服务。

```bash
python -B run_pipeline.py demo
python -B run_pipeline.py check
```

以上不会调用 API 或加载模型。真实训练需要自己的模型、数据、capture 和可信 authority。

后续计划以 50 题、每题 4 条轨迹为一个采集/评分周期，再独立审核、编译和训练。采集 50 题不等于一次 optimizer.step；目前训练器按题组更新，跨题累积需要另外实现和验证。

已完成的一题四轨迹一次真实更新属于工程闭环验证，不代表已经证明整体效果提升。本项目用于研究，不用于医疗决策。
