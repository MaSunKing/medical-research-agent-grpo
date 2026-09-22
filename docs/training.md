# SFT 与 RL 训练安排

## 运行环境与预算

服务器训练与集成实验使用新加坡 NSCC GPU 集群，同时保留单 GPU 兼容路径；目标本地推理环境为 RTX 4080 SUPER 4-bit 加载。

共享 V4.2 合同采用 8192 tokens，计算方式为每次请求的输入加预留输出。Checklist/State 输出预算 1200，工具决策 240，证据卡提取 600，Final 2400 tokens。Final 因而最多保留 5792 tokens 的完整输入，包含模板、题目、任务状态与证据；没有固定的 Final 字符上限。8K 是部署与训练合同的预算选择，不是硬件或 backbone 能力上限的声明。

GPU 训练需要 Linux/WSL、兼容 CUDA PyTorch、自有模型与数据、真实 capture 及可信 authority。安装依赖不会自动生成可训练 batch；公开静态案例的摘录不能代替 token-exact 的训练记录。

## SFT

sft/train_tc2.py 实现 completion-only、样本加权的单卡 QLoRA。Prompt labels 被 mask，completion Token 参与 loss，样本权重独立施加。参考配置为 NF4 double quantization、bfloat16、attention/MLP LoRA（r32、alpha64、dropout0.05）。

自行提供 stage 数据与 fast tokenizer；编码契约见 sft/prepare_dataset.py。Manifest 用于完整性检查，不代表私有训练数据的公开授权。

```bash
python -B sft/train_tc2.py --help
```

## RL

从原始 backbone 加固定、完整的 SFT LoRA 起步，更新同一个共享 adapter。SFT/RL 对照使用两套替代 adapter，不将它们叠加合并。

```text
冻结 behavior policy
→ 每题采集四条 rollout
→ 分阶段 Judge 与增量 coverage
→ 独立语义审核
→ authority / token / group 验证
→ 编译 reward 与 advantage
→ clipped policy optimization
→ 原子 checkpoint 与完整性检查
→ 留出集配对评测
```

training/train.py 需要绑定 batch、模型路径、起始 adapter 和独立可信 authority。先查看 CLI，不手工拼造评分记录。--preflight 不加载模型，真实训练另做 behavior replay parity 检查。

参考默认参数：lr 1e-6、epsilon 0.2、target KL 0.02、parity tolerance 0.05。它们不是已证明最优的超参数。GPU trainer 使用 Linux 锁机制，应采用 Linux/WSL 并安装兼容 CUDA PyTorch；依赖见 pyproject.toml，部署版本需单独冻结验证。

## 批次安排

先验收完整小批次，再以 **50 题 × 4 rollout = 200 轨迹**为一个采集/评分周期。逐轨迹保留 capture 与断点状态，同一个 behavior batch 不混 policy identity；不完整题组暂不放行。

采集周期与 optimizer step 分开记录：50 题用于形成可审核的采集/评分批次，trainer 再按冻结的题组、token 绑定与训练配置执行更新。

审核可在采集和评分后执行。纠正必须保留原始回执、版本和可信审核 authority，并重新编译/验证；直接改 batch 分数不是合法训练信号。Prompt、adapter 或 decoding processor 变化也会改变身份与缓存/replay 复用资格。
