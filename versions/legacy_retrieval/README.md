# 历史检索快照

单 LoRA 与双 LoRA 历史路线共用此冻结源码。它保留首页重组前的 `retrieval/` 内容，文件身份见 [SOURCE_HASHES.json](SOURCE_HASHES.json)。当前检索在 [../../retrieval/](../../retrieval/README.md)。

历史方法与实验入口见 [单 LoRA](../single_lora/README.md)。旧实验使用旧上下文、引用及排序合同，不能自动归到当前检索实现。

此目录是历史复现时的工具源码根，不会被默认导入。需要旧工具时把本目录置于 Python 导入路径的最前面；不要在同一进程中混用两套 `dr_agent`。历史共享 LoRA 的训练、Judge 和 replay 入口继续保留在仓库原位置。
