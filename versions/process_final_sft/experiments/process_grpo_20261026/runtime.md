# 历史Base/SFT与本次GRPO：设计和运行配置

[结果](README.md)

## 核心设计相同

核对两份实际冻结源码：prompts.py字节一致；Search缓存、候选可用性、Browse和检索worker源码一致。检索快照MANIFEST的变化仅跟随共享ChatCodec文件哈希，不表示更换了检索算法。

| 项目 | 历史Base/SFT | 新GRPO13/25 |
|---|---|---|
| 基座 | Qwen3-8B，NF4双重量化、BF16 compute | 相同 |
| 节点与提示词 | Checklist→Decision→工具→State→Pre-final→Final | 相同 |
| embedding / reranker | bge-small-en-v1.5 / MedCPT-Cross-Encoder | 相同 |
| 工具预算 | 正常6、失败豁免5、actual最多11 | 相同 |
| 上下文 | soft12288、hard15000、repair384、safety128 | 相同 |
| 压缩 | T=0.1，最多12次 | 相同 |
| 标称采样 | T=1、top_p=1、top_k=0、repetition_penalty=1 | 相同 |
| Final | 同一基座禁用Process adapter | 相同，保留原BF16输出头 |
| Process权重 | Base / 原972 SFT | GRPO13 / GRPO25：要比较的实验变量 |

## 实现上的变化

- 新Process使用未合并FP32线性输出头；旧服务沿原输出头计算。这是数值策略差异，不是新模型或新节点。
- 旧服务显式传T/top_p/top_k；新入口进一步禁用模型默认值回填，并检查实际生效参数。两批标称T均为1，不能把旧运行直接说成T=0.6；旧缺少与新入口同级的生效配置核验。
- 新评测按题和调用输入哈希派生种子；旧按冻结题目种子。旧Q01 Process seed=2026102101，新Q01组seed=2021059079，不是相同随机实现。
- 新runtime增加保守JSON语法修正，例如尾逗号和JSON围栏；原字段、ID和医学语义校验不放宽。它可能影响失败率，不能将所有失败减少归于训练。
- ChatCodec明确return_dict=False并验证整数token；训练导出跳过程序修复target。这些兼容/导出保护不改变Search/Browse设计。
- 新检索分配8个逻辑CPU、passage线程4，冻结历史排序线程2；线程调整主要解决资源争用，不是改变排序权重或检索模型。
- 三组不同时刻联网，页面/API响应未全局冻结，即使配置相同也存在环境波动。

重复Search政策没有新增拦截规则，因此新两组完全相同查询额外动作0值得报告；仍只代表本次运行，不证明每类语义冗余都消失。历史差值作描述，GRPO13/25之间运行配置相同，但也没有重复运行误差估计。

本次仅更新文档与脱敏结果。current源码目录仍保持此前发布身份；不要将其历史模型服务当作新的GRPO原生采样入口，完整执行包、训练原始数据和权重留在私有归档。
