# 2x2 数据集 v1 汇总

生成方式：规则控制生成，目标是先得到干净的方向提取数据，而不是模拟真实用户分布。

## 总量

- 基础场景数：300
- 回答样本数：1200
- 每个场景回答数：4

## split

- train: 200 场景，800 条回答
- dev: 50 场景，200 条回答
- test: 50 场景，200 条回答

## 场景类别

- decision_pressure: 50 场景，200 条回答
- high_emotion_distortion: 50 场景，200 条回答
- reassurance_seeking: 50 场景，200 条回答
- relationship_attribution: 50 场景，200 条回答
- self_catastrophizing: 50 场景，200 条回答
- suspicious_attribution: 50 场景，200 条回答

## 2x2 组合

- high_emotion_low_belief: 300
- high_emotion_high_belief: 300
- low_emotion_low_belief: 300
- low_emotion_high_belief: 300

## 标签分布

- emotion_validation=0, belief_endorsement=0: 300
- emotion_validation=0, belief_endorsement=1: 300
- emotion_validation=1, belief_endorsement=0: 300
- emotion_validation=1, belief_endorsement=1: 300

## 使用提醒

- 这版数据适合先做方向提取、probe 和激活层扫描。
- 它不是最终论文评测集，因为语言风格仍然偏规则化。
- 后续应增加人工审查、自然改写、挑战集和跨模板测试。
