# Final 输入与引用合同

Final 接收实际Pre-Final输入：原问题、模型工作状态、已读可引用证据、来源信息，以及实际共享builder保留的历史。History、State与Checklist可有误；事实依据是当前可见的证据文本。

当前独立Final接口：

- 输出单个 `<answer>...</answer>`，不输出思考、审计说明或工具调用。
- 每个证据性事实句使用当前输入的 `[E1]` 等短别名；多证据写为相邻引用。
- 映射表将别名绑定到原始chunk ID；不改变Process的来源选择ID或原证据文本。
- 数字、单位、人群、终点与时间须准确对应。摘录缺失不等于原论文没报告或效果不存在。
- 状态、标题、历史ID、预算占位回执不能代替事实证据；只使用当前可引用ID。

实现与实际Prompt在 [final_citation_alias_v1.py](../versions/process_final_sft/final/final_citation_alias_v1.py)。该文件保留本轮Final数据生成所用中文Prompt；当前Process的Checklist/Decision/State Prompt是英文。更换Final Prompt语言需要另做输入合同和对照，不在文档整理时悄悄更换。

代码格式校验只检查answer边界、别名允许集合与引用映射；逐句支持、数值绑定和总结质量仍需语义评测。历史单LoRA的XML citation格式保持在其版本中，不覆盖旧记录。
