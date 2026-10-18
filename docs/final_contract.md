# Final输入、冻结与引用合同

## 两层输入，不可混用

| 产物 | 用途 |
|---|---|
| run/PRE_FINAL_PACKAGE.json | 完整原题、Checklist、State、证据、来源映射、历史与未解决项，供审计 |
| teacher_input/FINAL_REQUEST.json | 同版Final构建器准备的真实messages、token ID、预算及投影审计 |
| teacher_input/MESSAGES.json | 两位老师共用、后续Final训练绑定的冻结输入 |
| teacher_input/TEACHER_INPUT.txt | 同一messages的网页粘贴版，不额外加全文 |
| teacher_input/INPUT_IDENTITY.json | 原轨迹/原包/消息/文件哈希、token预算、quality_hold |

原始包并非实际Final消息的逐字副本。构建器先精简重复包装、已知坐标展示和重复State引用，必要时才原文抽取；不改原轨迹、State、原证据或引用关系。不能换packing后沿用旧金标。

## 当前600题只冻结，不生成

使用原始972 Process checkpoint1322，600题全部采Process→Pre-Final，预分500 train/100 dev。process_only的final=null、final_status=not_requested；冻结请求不等于调用Final。超限时可能调用禁用LoRA的同基座压缩兜底，须单独留回执，不能声称完全没有辅助模型调用。

三题服务端试采正常完成并通过输入完整性；仍有乱码及State内容问题，尚非医学合格金标。无证据、降级、失败、无法安全冻结或语义可疑记录留置，不把文件存在当作训练批准。

## 预算与引用

实际chat模板/tokenizer计量：软线12,288、硬线15,000；Final预留2,400、repair384、安全余量128均计总量。tokenizer显式返回扁平整数ID并校验，不把BatchEncoding键误算成tokens；保留实际环境与输入哈希。

完整问答模式输出单个&lt;answer&gt;...&lt;/answer&gt;，使用当前可引用[E1]等证据ID。E绑定真实已打开片段及source_id/chunk_id，不是Browse来源字母。没有E不能造E1；Search标题/预览、State、历史和预算回执均不能替代正文。

程序检查answer边界和引用ID存在，不保证逐句语义，也不把Final幻觉修成正确答案。数字、单位、人群、比较、结局、时效与研究不确定性按可见证据审核。未见摘录不等于原研究未报告或研究不存在。Final可纠正State，不能编造缺失证据。

## 金标边界

后续网页ChatGPT与Codex读取同一冻结输入各出候选，按证据支持、问题覆盖和引用审核，不只选文风。不能额外联网、引用隐藏全文或从题目来源补结论。两份不合格则留置或有审计地修改；不属于本次服务器采集，不表示Final已训练。

实际模块为ContextBuilder.prepare('final')、Engine.prefinal/final_node及[当前freeze_final_input.py](../versions/process_final_sft/current/freeze_final_input.py)。既有[Final引用组件](../versions/process_final_sft/final/final_citation_alias_v1.py)保留原Prompt/数据合同，不冒充当前冻结构建器源码。见[代码导航](code_navigation.md)。后续超参数和评测计划本次不调整。
