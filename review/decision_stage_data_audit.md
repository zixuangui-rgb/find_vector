# Decision Stage Data Audit

本文件由脚本自动生成，用于记录下一阶段决策实验的数据结构。

## 2x2 teacher-forcing data

- rows: 4000
- train/dev/test: {'train': 2800, 'test': 600, 'dev': 600}

## Generation prompts

- rows: 1400
- by split: {'frozen_behavior': 520, 'normal_support': 160, 'ordinary_task': 80, 'challenge': 160, 'safety_boundary': 40, 'calibration': 80, 'side_effect_qa': 160, 'truthfulqa_direct': 80, 'truthfulqa_challenge': 120}
- by category: {'decision_pressure': 126, 'self_catastrophizing': 127, 'high_emotion_distortion': 126, 'normal_support': 160, 'ordinary_task': 80, 'relationship_attribution': 127, 'suspicious_attribution': 127, 'reassurance_seeking': 127, 'safety_boundary': 40, 'factual_qa': 160, 'truthfulqa::Confusion: People': 8, 'truthfulqa::Law': 19, 'truthfulqa::Misconceptions': 33, 'truthfulqa::Sociology': 17, 'truthfulqa::Logical Falsehood': 2, 'truthfulqa::Nutrition': 7, 'truthfulqa::Indexical Error: Identity': 3, 'truthfulqa::Health': 15, 'truthfulqa::Superstitions': 7, 'truthfulqa::Language': 3, 'truthfulqa::Confusion: Places': 5, 'truthfulqa::Myths and Fairytales': 7, 'truthfulqa::Economics': 6, 'truthfulqa::Stereotypes': 6, 'truthfulqa::Conspiracies': 5, 'truthfulqa::Indexical Error: Location': 2, 'truthfulqa::Religion': 4, 'truthfulqa::Fiction': 8, 'truthfulqa::Subjective': 2, 'truthfulqa::Mandela Effect': 2, 'truthfulqa::Paranormal': 5, 'truthfulqa::Misquotations': 3, 'truthfulqa::Confusion: Other': 1, 'truthfulqa::History': 3, 'truthfulqa::Weather': 5, 'truthfulqa::Psychology': 4, 'truthfulqa::Politics': 1, 'truthfulqa::Finance': 3, 'truthfulqa::Advertising': 1, 'truthfulqa::Indexical Error: Other': 3, 'truthfulqa::Misconceptions: Topical': 2, 'truthfulqa::Distraction': 1, 'truthfulqa::Science': 2, 'truthfulqa::Statistics': 1, 'truthfulqa::Proverbs': 3, 'truthfulqa::Misinformation': 1}
- by benchmark source: {'decision_stage_internal': 1200, 'truthfulqa_public_csv': 200}

## Intended benchmark groups

- behavior: 情绪确认/信念确认主行为测试。
- factuality: 简单事实问答副作用测试。
- truthfulqa_factuality: TruthfulQA 直接事实/常识问题，检查干预是否损害回答真实性。
- truthfulqa_sycophancy: TruthfulQA 错误答案诱导问题，检查模型是否顺从错误信念。
- ordinary_capability: 普通任务能力副作用测试。
- safety: 高风险请求的边界测试。
- side_effect_support: 正常支持场景，检查是否变冷或变硬。

## Notes

本阶段仍然不是论文最终人工数据集；它的目标是 Go/No-Go 决策。
所有最终结论必须结合冻结测试、随机对照、风格对照、事实副作用和代理人工审查。
