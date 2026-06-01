# 后训练前实验计划：寻找并验证“情绪确认”和“信念确认”向量

生成日期：2026-06-01  
实验目录：`/Users/hell/Documents/interpretability/emotion_belief_direction_prevalidation`  
目标模型：Qwen3.5-7B 系列 instruction 模型  
实验边界：本文只设计 **后训练之前** 的实验，即寻找、筛选、验证两个候选内部向量；不包含 SFT、RLHF、DPO、ReFT 训练或 preventative steering 后训练。

## 1. 核心问题

我们要验证的问题不是：

> 模型内部是否天然存在两个完全独立、完美干净的心理模块？

这个说法太强，也不符合当前可解释性研究的谨慎表达。

更科学的问题是：

> 在 Qwen3.5-7B 的某些中间表示中，是否可以找到两个稳定、可读出、可操控、可部分分离的线性方向：一个主要对应“情绪确认”，一个主要对应“信念确认”？

其中：

- **情绪确认**：承认、理解、接住用户的感受。
- **信念确认**：在证据不足或判断过强时，顺着用户确认其判断、归因或结论。

本文中“信念确认”比“确认事实”更准确。我们不希望模型少确认真实事实，而是希望模型少确认用户未经证据支持的判断。

## 2. 文献依据

本计划主要参考以下方法线索：

- **2x2 控制数据**：借鉴 sycophancy 测评和 social sycophancy 工作，核心是把“用户情绪”和“用户信念”分开操控。
- **Activation Addition / CAA**：用正负样本的激活差值寻找方向。
- **Representation Engineering**：方向不能只被读出来，还要能因果操控输出。
- **Inference-Time Intervention**：按层、按模块寻找最有效干预位置。
- **Emotion Concepts**：情绪概念可能在中后层更清晰，应做全层扫描。
- **Sycophancy Hides Linearly in Attention Heads**：信念确认/迎合可能不只在 residual stream，也可能藏在 attention heads。
- **Linear Probe Penalties / Persona Coordinates**：probe 要做跨场景泛化，避免只学到模板词。
- **INLP / LEACE**：用概念擦除检验两个方向是否可分。
- **AxBench**：简单方法常常很强，必须包括 DiffMean、probe、prompt/random baseline，不能直接假设复杂方法更好。

因此，本实验采用“由浅到深”的路线：

1. 先在 residual stream 找方向。
2. 再用 probe 和 steering 验证方向。
3. 如果 residual stream 不够清楚，再扩展到 attention head / MLP 输出。
4. 后训练前只做向量验证，不做训练改权重。

## 3. 实验目标与假设

### 3.1 目标

本阶段的最终产物不是一个训练好的模型，而是：

1. 一个候选情绪确认向量 `v_emotion`。
2. 一个候选信念确认向量 `v_belief`。
3. 每个候选向量对应的最佳层或最佳层组。
4. 一套证据说明二者是否可分离。
5. 一个明确的 Go / No-Go 结论：是否值得进入后训练阶段。

### 3.2 假设

H1：在 Qwen3.5-7B 的部分层中，情绪确认和信念确认可以被线性 probe 读出。  
H2：情绪确认方向和信念确认方向不是同一个方向，至少存在部分可分性。  
H3：对模型激活加减对应方向，会对输出行为产生可预测影响。  
H4：如果擦除或正交化信念确认方向后，情绪确认仍能保留，则说明二者有可利用的分离空间。  
H5：如果 H1-H4 不成立，则不应进入后训练；应该先重新设计数据或改用更细概念。

## 4. 数据设计

### 4.1 已完成的数据 v1

当前已有 v1 数据：

- 路径：`data/2x2_items_all_v1.jsonl`
- 基础场景：300 个
- 回答样本：1200 条
- train/dev/test：800 / 200 / 200 条
- 6 类场景：
  - relationship_attribution
  - self_catastrophizing
  - suspicious_attribution
  - reassurance_seeking
  - high_emotion_distortion
  - decision_pressure

每个场景有四种回答：

| 条件 | emotion_validation | belief_endorsement | 用途 |
|---|---:|---:|---|
| high_emotion_low_belief | 1 | 0 | 情绪正例、信念负例 |
| high_emotion_high_belief | 1 | 1 | 双高混合样本 |
| low_emotion_low_belief | 0 | 0 | 双低基线 |
| low_emotion_high_belief | 0 | 1 | 情绪负例、信念正例 |

### 4.2 为什么必须是 2x2

如果只比较“温暖回答”和“冷淡回答”，模型可能学到的是：

- 礼貌程度
- 长短差异
- 语气强弱
- 是否有建议

而不是“情绪确认”。

如果只比较“迎合回答”和“不迎合回答”，模型可能学到的是：

- 是否说“你是对的”
- 是否反驳用户
- 是否有事实纠错

而不是“信念确认”。

2x2 的意义是：在同一用户场景里，让情绪确认和信念确认独立变化，这样才可能分离两个方向。

### 4.3 v1 数据的已知风险

v1 是规则控制数据，优点是干净，缺点是模板痕迹较强。正式推理前需要承认以下风险：

- 高信念确认回答故意较强，可能让方向偏向“极端同意”。
- 低情绪回答偏短，可能让模型把“长度”当成特征。
- 部分表达存在模板重复，probe 可能读到表面句式。

因此，v1 只作为第一阶段方向提取数据。后续如果 v1 结果过强，必须做 v2 自然改写和挑战集验证。

## 5. 模型与服务器配置

### 5.1 模型

主模型：

- Qwen3.5-7B-Instruct 或本地等价 7B 指令模型。

可选对照：

- Qwen3.5-4B：快速调试。
- Qwen3.5-14B：少量关键复核，不做全量扫描。

### 5.2 推荐服务器

推荐：

- 4 × 24GB GPU
- 128GB CPU 内存
- 约 13.20 元/小时

理由：

- 7B 推理和激活提取单卡可行，但全层扫描、steering 和 probe 会耗时。
- 4 卡可以并行不同层、不同方法、不同 split。
- 128GB 内存足够存中间激活、向量、probe 特征和日志。

### 5.3 运行原则

- 模型权重保持冻结。
- 不做梯度更新。
- 所有结果可恢复、可重跑。
- 所有中间文件写盘。
- 每个阶段必须写日志。

## 6. 表示位置选择

### 6.1 第一优先级：residual stream

先在每层 residual stream 上提取表示。

建议 token pooling：

1. response tokens mean pooling：主结果。
2. final response token：辅助结果。
3. first assistant token：辅助结果。

主结果使用 response tokens mean pooling，因为我们要找的是“回答风格/策略方向”，不是单个词预测。

### 6.2 第二优先级：attention head output

如果 residual stream 的方向不稳定，扩展到 attention head 输出：

- 每层每个 head 的输出。
- 训练 head-level probe。
- 找最能区分信念确认的 heads。

原因：已有 sycophancy 机制工作提示，迎合信号可能藏在 attention heads 中。

### 6.3 第三优先级：MLP output

如果 attention heads 仍不清晰，再看 MLP output：

- 每层 MLP 输出平均。
- 比较 MLP 与 attention 的 probe AUC 和 steering 效果。

这一步不是主线，只有 residual/head 证据不足时才做。

## 7. 实验阶段

### 阶段 0：服务器预检

目的：避免全量任务开跑后才发现显存、模型路径或 hook 有问题。

输入：

- 10 个场景，40 条样本。
- Qwen3.5-7B。

操作：

- 加载模型。
- 运行 40 条样本。
- 抽取 3 个层的 residual stream。
- 保存激活。
- 运行一次简单 direction 计算。

输出：

- `logs/00_smoke_test.log`
- `activations/smoke_residual_layers.npz`
- `analysis/smoke_direction_check.json`

通过标准：

- 模型成功加载。
- 无 OOM。
- 激活 shape 正确。
- 每条样本都有对应激活。
- 单次运行可在合理时间内完成。

失败处理：

- OOM：降低 batch size 或改为分层分批保存。
- hook 错误：先只保存 hidden_states，再接 forward hook。
- 模型路径错误：修正本地缓存或联网下载。

### 阶段 1：全量冻结推理与激活提取

目的：在不改模型的情况下，收集 2x2 样本的内部表示。

输入：

- `data/2x2_items_train_v1.jsonl`
- `data/2x2_items_dev_v1.jsonl`
- `data/2x2_items_test_v1.jsonl`

操作：

- 对每条样本构造对话：
  - user prompt
  - assistant response
- 采用 teacher-forcing 方式送入模型。
- 提取每层 assistant response tokens 的 residual stream。
- 保存 train/dev/test 的激活。

说明：

这里优先用 teacher-forcing，而不是让模型自己生成回答。原因是我们要研究模型对“给定回答风格”的内部表征，而不是先受模型生成偏好影响。

输出：

- `activations/residual_train_layers.npz`
- `activations/residual_dev_layers.npz`
- `activations/residual_test_layers.npz`
- `analysis/activation_shapes.json`

检查项：

- 每个样本都有激活。
- 每层维度一致。
- train/dev/test 不混。
- 没有把 user tokens 混进 response pooling 主结果。

### 阶段 2：候选方向提取

目的：为每层提取两个候选方向。

#### 2.1 DiffMean / CAA-style 方向

对每一层 `l`：

```text
v_emotion_l = mean(h | emotion_validation=1) - mean(h | emotion_validation=0)
v_belief_l  = mean(h | belief_endorsement=1) - mean(h | belief_endorsement=0)
```

其中 `h` 是该层 response token pooled activation。

输出：

- `vectors/diffmean_emotion_by_layer.npz`
- `vectors/diffmean_belief_by_layer.npz`

#### 2.2 条件化方向

为了避免混淆，再计算条件化版本：

```text
v_emotion_given_low_belief =
  mean(high_emotion_low_belief) - mean(low_emotion_low_belief)

v_emotion_given_high_belief =
  mean(high_emotion_high_belief) - mean(low_emotion_high_belief)

v_belief_given_low_emotion =
  mean(low_emotion_high_belief) - mean(low_emotion_low_belief)

v_belief_given_high_emotion =
  mean(high_emotion_high_belief) - mean(high_emotion_low_belief)
```

如果同一方向在不同条件下差异很大，说明概念纠缠严重。

#### 2.3 正交化方向

计算：

```text
v_emotion_orth = v_emotion - projection(v_emotion on v_belief)
v_belief_orth  = v_belief  - projection(v_belief on v_emotion)
```

这不是最终答案，只是用于测试“去掉另一个方向后是否仍有效”。

输出：

- `vectors/orthogonalized_vectors_by_layer.npz`
- `analysis/vector_geometry_by_layer.csv`

### 阶段 3：几何与统计初筛

目的：先判断方向在几何上是否有基本可分性。

指标：

- `cos(v_emotion_l, v_belief_l)`
- 两个方向在样本上的 projection correlation
- 条件化方向之间的 cosine
- train/dev/test 上 projection 分布是否稳定
- 每层 effect size

输出：

- `analysis/vector_geometry_by_layer.csv`
- `figures/layer_cosine_curve.png`
- `figures/projection_distribution_by_layer.png`

初筛标准：

- 至少存在若干层使 `|cos(v_emotion, v_belief)| < 0.35`。
- 情绪方向在 dev/test 上能区分 emotion label。
- 信念方向在 dev/test 上能区分 belief label。
- 条件化方向不能完全崩坏。

注意：

几何初筛不能单独证明方向有效。它只决定哪些层进入后续 probe 和 steering。

### 阶段 4：linear probe 验证

目的：检验两个概念是否可被线性读出，并测试是否只是模板特征。

训练：

- 在 train 上训练 emotion probe。
- 在 train 上训练 belief probe。
- 模型：logistic regression / linear SVM。
- 输入：每层 pooled residual activation。
- 正则：L2。
- 类别平衡。

评估：

- dev/test AUC。
- macro F1。
- calibration。
- cross-condition AUC。

关键交叉测试：

1. emotion probe 在低信念样本上是否仍能识别情绪确认。
2. emotion probe 在高信念样本上是否仍能识别情绪确认。
3. belief probe 在低情绪样本上是否仍能识别信念确认。
4. belief probe 在高情绪样本上是否仍能识别信念确认。

输出：

- `probes/emotion_probe_by_layer.pkl`
- `probes/belief_probe_by_layer.pkl`
- `analysis/probe_metrics_by_layer.csv`
- `analysis/probe_cross_condition_metrics.csv`

通过标准：

- emotion probe test AUC ≥ 0.80。
- belief probe test AUC ≥ 0.80。
- cross-condition AUC ≥ 0.70。
- 两个 probe 的错误模式不同。

如果 probe AUC 很高但 cross-condition 低，说明 probe 可能学到了模板或长度，不算通过。

### 阶段 5：方向因果验证，也就是 steering

目的：证明方向不只是“可读”，还要“可控”。

#### 5.1 生成式测试集

不能只在 teacher-forcing 样本上做。需要准备一批 held-out user prompts，让模型自由生成回答。

建议规模：

- 100 个单轮 held-out prompts。
- 60 个多轮压力 prompts。
- 60 个普通情绪支持 prompts。
- 60 个事实/常识 side-effect prompts。

这些 prompt 不应和 v1 场景完全重复。

#### 5.2 干预条件

对每个候选层和 alpha：

- base：不加向量。
- `+v_emotion`。
- `-v_emotion`。
- `+v_belief`。
- `-v_belief`。
- `+v_emotion - v_belief`。
- random vector control。
- norm-matched random vector control。

#### 5.3 alpha 扫描

建议：

```text
alpha ∈ {0.25, 0.5, 1.0, 1.5, 2.0} × std_projection
```

不要直接用固定绝对值，因为不同层激活尺度不同。

#### 5.4 评分维度

每个输出独立评分：

- emotion_validation
- belief_endorsement
- warmth
- gentle_correction
- epistemic_calibration
- over_reassurance
- naturalness
- refusal_or_harshness
- factual_side_effect_correctness

评分方式：

1. 规则指标：是否出现明显确认词、保证词、绝对化词。
2. LLM judge：严格分项 rubric，不给总分。
3. 人工抽样复核：至少每条件 20 条。

输出：

- `responses/steering_generations.jsonl`
- `scores/steering_scores.jsonl`
- `analysis/steering_effects.csv`
- `analysis/steering_paired_tests.csv`

因果通过标准：

- `+v_emotion`：emotion_validation 上升，belief_endorsement 不显著上升。
- `+v_belief`：belief_endorsement 上升。
- `-v_belief`：belief_endorsement 下降，emotion_validation 保留至少 80%。
- random vector 不应产生同方向稳定效果。
- side-effect factual correctness 不显著下降。

### 阶段 6：擦除与可分性验证

目的：检验两个方向是否真的可以拆开。

#### 6.1 projection ablation

在激活中移除某个方向：

```text
h_without_belief = h - projection(h on v_belief)
h_without_emotion = h - projection(h on v_emotion)
```

测试：

- 移除 `v_belief` 后，belief probe 是否下降。
- 移除 `v_belief` 后，emotion probe 是否保留。
- 移除 `v_emotion` 后，emotion probe 是否下降。
- 移除 `v_emotion` 后，belief probe 是否过度受损。

#### 6.2 INLP / LEACE

如果单向量擦除不足，再用：

- INLP：反复训练线性分类器并投影掉概念子空间。
- LEACE：闭式线性概念擦除。

输出：

- `analysis/erasure_probe_metrics.csv`
- `analysis/erasure_behavior_metrics.csv`

通过标准：

- belief erasure 后，belief probe AUC 明显下降。
- belief erasure 后，emotion probe AUC 保留 ≥ 80%。
- 行为上，信念确认下降，情绪确认不显著下降。

如果擦除 belief 后 emotion 也大幅下降，说明二者在当前表示里高度纠缠。

### 阶段 7：泛化与稳健性测试

目的：防止只在 v1 模板上成功。

需要额外构造三个 held-out 集合：

1. **自然改写集**：同一语义，但更口语化、少模板。
2. **挑战集**：更模糊、更接近真实对话。
3. **跨任务集**：非情绪事实 QA、普通建议、知识问答。

测试内容：

- probe 是否仍有效。
- steering 是否仍有效。
- `-v_belief` 是否会让模型变冷或拒答。
- `+v_emotion` 是否会增加过度安慰。

输出：

- `analysis/generalization_probe_metrics.csv`
- `analysis/generalization_steering_metrics.csv`

通过标准：

- 自然改写集上 probe AUC 不低于原 test 过多。
- challenge 集上 steering 效果方向一致。
- side-effect 集无明显退化。

## 8. 层选择规则

不要人工凭感觉选层。每层都计算综合分：

```text
score_layer =
  0.25 * emotion_probe_auc
  + 0.25 * belief_probe_auc
  + 0.20 * steering_selectivity
  + 0.15 * erasure_selectivity
  + 0.10 * cross_condition_generalization
  - 0.05 * side_effect_penalty
```

最终保留：

- 最佳情绪层。
- 最佳信念层。
- 最佳共同层。
- 最佳多层组合。

如果单层不稳定，尝试多层组合：

- 中后层平均。
- top-k 层加权。
- 每层独立 alpha。

但第一版报告必须先呈现单层结果，避免一上来过拟合。

## 9. 决策门槛

### 9.1 Go：可以进入后训练

满足以下条件：

1. 两个方向在 test 上都能被稳定读出。
2. 两个方向 cosine 不高，或正交化后仍有效。
3. steering 能分别改变对应行为。
4. `-v_belief` 能降低信念确认，同时情绪确认保留 ≥ 80%。
5. random vector control 不产生同样效果。
6. 自然改写/挑战集上仍有同方向结果。

### 9.2 Conditional Go：需要先改数据或方法

出现以下情况：

- probe 很强，但 steering 很弱。
- residual stream 不行，但 head-level probe 有信号。
- v1 成功，自然改写集失败。
- 情绪方向和信念方向高度相关，但某些层可分。

处理：

- 做 v2 数据。
- 增加 head-level / MLP-level 分析。
- 拆更细概念，例如“情绪承认”“过度保证”“用户判断确认”“压力顺从”。

### 9.3 No-Go：不进入后训练

出现以下情况：

- probe 无法稳定读出。
- steering 不产生行为变化。
- 擦除 belief 后 emotion 大幅消失。
- random vector 与目标向量效果相近。
- 方向只在模板数据有效。

此时不应继续烧算力做后训练。更好的结果表达是：当前模型/数据下，二者尚未显示可稳定分离。

## 10. 统计分析

### 10.1 probe 指标

- AUC
- F1
- accuracy
- calibration error
- bootstrap 95% CI

### 10.2 steering 指标

使用 paired design：

- 同一个 prompt 在不同条件下比较。
- 报告平均差异。
- bootstrap 95% CI。
- Wilcoxon signed-rank test。
- 多重比较用 Benjamini-Hochberg FDR。

### 10.3 effect size

所有主结果必须报告：

- mean delta
- standardized effect size
- confidence interval
- n prompts
- n generations

不要只报告 p-value。

## 11. 预期文件结构

建议服务器上实验目录：

```text
emotion_belief_direction_prevalidation_server/
  config/
    experiment_config.yaml
  data/
    2x2_items_train_v1.jsonl
    2x2_items_dev_v1.jsonl
    2x2_items_test_v1.jsonl
    heldout_generation_prompts.jsonl
    challenge_prompts.jsonl
  scripts/
    00_smoke_test.py
    01_extract_activations.py
    02_build_vectors.py
    03_probe_validation.py
    04_steering_validation.py
    05_erasure_validation.py
    06_generalization_tests.py
    07_analyze_and_report.py
  activations/
  vectors/
  probes/
  responses/
  scores/
  analysis/
  figures/
  logs/
```

## 12. 预计运行时间

在 4 × 24GB GPU / 128GB 内存上，Qwen3.5-7B：

- 服务器环境与 smoke test：1-2 小时。
- 全量 teacher-forcing 激活提取：2-5 小时。
- 方向提取与几何分析：0.5-1 小时。
- probe 训练与交叉验证：1-2 小时。
- steering 生成测试：6-14 小时。
- erasure 与泛化测试：4-10 小时。
- 分析、图表、报告：1-3 小时。

总计：

- 顺利情况：约 15-25 小时。
- 稳妥预算：约 1.5-2 天。
- 如果加入 head-level 全扫描和挑战集重跑：约 2-3 天。

## 13. 主要风险与应对

### 风险 1：模板方向污染

表现：

- probe 很强，但自然改写集失败。

应对：

- 做 v2 自然改写。
- 控制回答长度。
- 增加 paraphrase split。

### 风险 2：方向可读但不可控

表现：

- probe AUC 高，但 steering 没有效果。

应对：

- 改用 CAA 多样本方向。
- 搜索更多层。
- 看 head/MLP 输出。
- 用 ReFT-r1 作为后续候选，但不在本阶段训练主模型。

### 风险 3：情绪确认与信念确认高度纠缠

表现：

- `-v_belief` 同时显著降低情绪确认。

应对：

- 进一步拆概念。
- 把情绪确认拆成“情绪命名/感受合理化/支持下一步”。
- 把信念确认拆成“事实确认/意图归因确认/过度保证/压力顺从”。

### 风险 4：向量副作用大

表现：

- `-v_belief` 让模型变冷、拒答、啰嗦或事实能力下降。

应对：

- 降低 alpha。
- 使用多层小 alpha。
- 加入 side-effect penalty 选层。

### 风险 5：7B 模型信号弱

表现：

- 方向不稳定，steering 效果弱。

应对：

- 先报告 7B 负结果。
- 用 14B 做少量关键复核。
- 不直接把失败归因于方法无效。

## 14. 最终报告应包含

最终 pretraining-vector-discovery 报告至少包含：

1. 数据说明。
2. 模型与硬件说明。
3. 激活提取方法。
4. 每层几何结果。
5. 每层 probe 结果。
6. steering 因果结果。
7. erasure 可分性结果。
8. 泛化测试。
9. 失败案例。
10. Go / Conditional Go / No-Go 结论。

## 15. 本阶段最重要的科学表达

如果成功，结论应写成：

> 在 Qwen3.5-7B 的若干中后层表示中，我们发现情绪确认与信念确认可以被部分线性分离。两个方向不仅能被 probe 读出，而且在生成时具有一定的因果操控效果。尤其是降低信念确认方向可以减少错误信念确认，同时保留大部分情绪确认。

如果失败，结论应写成：

> 在当前模型、数据和线性表示设定下，我们没有找到足够稳定、可操控、可泛化的分离方向。结果提示情绪确认和信念确认可能在该模型中高度纠缠，或当前数据仍含模板污染，需要更细粒度概念和更自然的数据。

这两种结果都有科学价值。真正不严谨的是：只凭 probe AUC 或少量 steering 成功，就直接宣称找到了两个独立内部模块。

