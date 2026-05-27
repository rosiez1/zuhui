# qkvcomm 项目周报：KV Cache 自适应量化与 SQuAD 小规模评测

日期：2026-05-27  
模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
数据：MRQA 2019 Shared Task 中的 SQuAD dev 子集  
主要目标：验证在当前 KV cache 量化框架中引入 hidden states 层重要性信息的可行性，并建立 MRQA/SQuAD 小规模 EM/F1 测评路径。

## 一、本周完成工作概览

本周围绕 ComprExIT 论文中的跨层信息传输思想，在现有 qkvcomm 项目中完成了两条低风险自适应量化策略的 demo 接入，并补充了基于 MRQA/SQuAD 的小规模 QA 评测链路。

主要完成内容如下：

1. 新增 MRQA/SQuAD 小规模 EM/F1 测评脚本。
2. 新增 `KV+Hidden mix` 自适应量化策略。
3. 新增 `hidden only` 自适应量化策略。
4. 使用 SQuAD dev 小样本进行 200 题评测，并分析当前指标反映的问题。
5. 明确下一步需要改进的方向：答案抽取后处理、随机采样、以及小规模 SFT。

## 二、SQuAD 小规模测评

### 2.1 测评脚本与指标

新增文件：

- `eval_mrqa_small.py`
- `evaluation/qa_metrics.py`

其中 `evaluation/qa_metrics.py` 实现了 SQuAD/MRQA 风格的 QA 指标：

- EM：Exact Match，标准化后预测答案与任一 gold answer 完全一致。
- F1：预测答案与 gold answer 的 token-level overlap F1。

`eval_mrqa_small.py` 完成了如下流程：

1. 读取 MRQA `.jsonl.gz` 文件，并跳过第一行 header。
2. 从每个 context 中抽取若干 QA。
3. 对 context 生成 KV cache。
4. 分别对 baseline、uniform8、adaptive864、depth_aware864、hidden_only864 进行答案生成。
5. 对每个 mode 的预测答案计算 EM/F1。
6. 输出 summary、逐题结果、context 存储开销等信息。

本轮正式小样本评测输出文件：

- `outputs_mrqa_eval/squad_dev_200q.json`

### 2.2 测评配置

本轮评测配置如下：

```json
{
  "max_contexts": 50,
  "max_questions": 200,
  "max_questions_per_context": 4,
  "max_context_tokens": 512,
  "max_new_tokens": 16,
  "include_hidden_states": true
}
```

含义：

- 最多读取 50 个 SQuAD context。
- 总计评测 200 个问题。
- 每个 context 最多取 4 个问题，避免样本集中在少数 context。
- 每个 context 最多保留 512 tokens。
- 每个问题最多生成 16 个 token 作为答案。
- 因为需要测试 `depth_aware864` 和 `hidden_only864`，所以开启 hidden states 提取。

### 2.3 测评结果

| mode | EM | F1 | Delta EM vs baseline | Delta F1 vs baseline | 平均压缩率 | 平均 bit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 12.00 | 35.08 | 0.00 | 0.00 | 1.00x | - |
| uniform8 | 12.50 | 35.75 | +0.50 | +0.67 | 2.00x | - |
| adaptive864 | 15.50 | 35.12 | +3.50 | +0.04 | 2.63x | 6.09 |
| depth_aware864 | 15.50 | 35.41 | +3.50 | +0.33 | 2.63x | 6.09 |
| hidden_only864 | 16.00 | 32.92 | +4.00 | -2.16 | 2.63x | 6.09 |

### 2.4 结果解读

从 200 题结果看，当前不能简单认为 adaptive 或 hidden-only 已经显著提升了 QA 能力。

更合理的解读如下：

1. `uniform8` 与 baseline 非常接近。
   - 200 题中，`uniform8` 有 174 题的规范化 prediction 与 baseline 完全一致。
   - 说明 8bit uniform 量化基本在复现 baseline 行为，保真度正常。

2. `adaptive864` 和 `depth_aware864` 在 2.63x 压缩率下 F1 基本不掉。
   - `adaptive864` 的 F1 为 35.12，几乎等于 baseline 的 35.08。
   - `depth_aware864` 的 F1 为 35.41，略高于 baseline。
   - 这说明 8/6/4 分层量化在当前小样本下没有明显破坏 QA 输出。

3. `hidden_only864` 的 EM 最高，但 F1 明显下降。
   - EM 为 16.00，高于 baseline 的 12.00。
   - F1 为 32.92，低于 baseline 2.16。
   - 这说明 hidden-only 会产生更多“短答案直接命中”的情况，但也带来更多完全跑偏或低重叠答案，稳定性不足。

4. baseline 指标偏低的重要原因是答案格式。
   - baseline 有 136/200 题的 prediction 中包含 gold answer 子串。
   - 但其中 112 题由于预测答案过长，EM 仍然为 0。
   - 例如 gold 为 `Denver Broncos` 时，baseline 可能输出完整句子 `The American Football Conference (AFC) champion Denver Broncos.`，语义上包含正确答案，但 EM 会失败，F1 也会被多余 token 稀释。

### 2.5 当前测评暴露的问题

本轮 SQuAD 小样本评测主要暴露出以下问题：

1. TinyLlama-Chat 不是抽取式 QA 模型。
   - 模型倾向生成完整解释句，而不是短答案 span。
   - MRQA/SQuAD 的 EM/F1 对短答案格式非常敏感。

2. 当前 prompt 与后处理仍不够稳定。
   - 即使 prompt 中要求 short span，模型仍会输出解释性句子。
   - `prediction` 清洗规则能截断换行和明显 marker，但无法可靠抽取句子中的答案 span。

3. 当前样本仍属于小规模顺序抽样。
   - 虽然已经从 20 题扩展到 200 题，但样本仍来自 SQuAD dev 的前 50 个 context。
   - 后续应加入随机采样和固定 seed，降低顺序样本偏差。

4. hidden-only 策略缺少 KV 误差约束。
   - 它完全依赖 hidden states 计算层重要性。
   - 当前结果显示它会改变生成风格，但对 QA 正确性的稳定保留不如 KV+Hidden mix。

### 2.6 后续改进计划

短期计划：

1. 增强答案后处理。
   - 针对 `The answer is ...`、完整句子、冒号后答案等情况做更稳的 prediction 抽取。
   - 目标是减少 EM/F1 被输出格式干扰。

2. 增加随机采样。
   - 为 `eval_mrqa_small.py` 增加随机 context/QA 抽样和 seed。
   - 避免只评测 SQuAD dev 文件前部样本。

3. 扩大评测规模。
   - 从 200 题扩展到 500 到 1000 题。
   - 先在 SQuAD dev 上稳定，再扩展到 NewsQA、HotpotQA、NaturalQuestionsShort 等 MRQA 子集。

中期计划：

1. 做小规模 LoRA SFT。
   - 目标不是让量化策略“看起来更好”，而是让 TinyLlama 学会 MRQA/SQuAD 的短答案格式。
   - 建议先用 MRQA train/SQuAD 取 5k 到 20k QA 进行 LoRA SFT。
   - SFT 后在 dev 集上重新比较 baseline、uniform8、adaptive864、depth_aware864、hidden_only864。

2. 对 hidden-only 做权重消融。
   - 当前 hidden-only 的 F1 不稳定。
   - 后续可降低 final-alignment 权重，增加层间 delta 约束，或与 KV quant error 做混合。

## 三、新增自适应量化策略

### 3.1 KV+Hidden mix 策略

代码入口：

- `compression/depth_aware_importance.py`
- `compression/adaptive_quantizer.py`
- `configs/depth_aware_default.json`

对应 mode：

- `importance_mode = "depth_aware"`
- 评测名称：`depth_aware864`

该策略保留原有 KV cache 自适应量化框架，在层重要性评分中加入 hidden states 的跨层变化信息。

当前组合评分大致包括：

1. KV norm score。
2. KV probe quantization error。
3. middle-layer prior。
4. hidden-state depth score。

默认权重：

```json
{
  "norm_weight": 0.30,
  "error_weight": 0.35,
  "prior_weight": 0.15,
  "depth_weight": 0.20
}
```

量化 bit 分配仍沿用 8/6/4：

- top 30% 重要层使用 8bit。
- middle 40% 使用 6bit。
- bottom 30% 使用 4bit。

在本轮 SQuAD 200 题结果中：

- 压缩率约 2.63x。
- F1 为 35.41，略高于 baseline 的 35.08。
- EM 为 15.50，高于 baseline 的 12.00。

当前判断：

- KV+Hidden mix 是目前更稳的新增策略。
- 它在保持压缩率的同时，没有明显损害 QA F1。
- 后续可以作为主线策略继续扩展评测。

### 3.2 Hidden only 策略

代码入口：

- `compression/depth_aware_importance.py`
- `compression/adaptive_quantizer.py`
- `configs/hidden_only_default.json`

对应 mode：

- `importance_mode = "hidden_only"`
- 评测名称：`hidden_only864`

该策略只使用 hidden states 计算层重要性，不使用 KV norm、KV quantization error 或 middle prior。它要求 KV dump 中必须包含 hidden states；如果没有 hidden states，会直接报错，避免偷偷退化为 KV fallback。

当前评分项：

1. `hidden_delta_score`
   - 衡量每层输入 hidden state 与输出 hidden state 的变化。

2. `final_alignment_score`
   - 衡量当前层 hidden state 与最终层 hidden state 的相似程度。

3. `hidden_norm_score`
   - 衡量 hidden state 激活幅度。

默认权重：

```json
{
  "hidden_delta_weight": 0.55,
  "hidden_final_weight": 0.35,
  "hidden_norm_weight": 0.10
}
```

量化 bit 分配同样使用 8/6/4。

在本轮 SQuAD 200 题结果中：

- 压缩率约 2.63x。
- EM 为 16.00，是所有 mode 中最高。
- F1 为 32.92，低于 baseline 的 35.08。

当前判断：

- hidden-only 可以改变模型生成风格，并在部分题目上直接给出短答案。
- 但它也更容易在部分题目上完全跑偏，导致 F1 下滑。
- 单独使用 hidden states 作为层重要性依据目前不够稳，更适合作为消融对照，而不是主线策略。

## 四、本周结论

1. MRQA/SQuAD 小规模 EM/F1 评测链路已经跑通。
   - 已完成 200 题 SQuAD dev 小样本评测。
   - 输出结果保存在 `outputs_mrqa_eval/squad_dev_200q.json`。

2. 当前评测的核心问题不是量化本身，而是 TinyLlama 的 QA 输出格式。
   - baseline 经常输出完整句子，EM/F1 因短答案格式不匹配而偏低。
   - 后续需要先改进答案抽取和 prompt，或者进行小规模 SFT。

3. `KV+Hidden mix` 是目前更稳的新增策略。
   - 在约 2.63x 压缩率下，F1 基本保持 baseline 水平。
   - 比 hidden-only 更适合作为后续主线实验。

4. `hidden only` 验证了只用 hidden states 进行层重要性评分的可行性，但当前稳定性不足。
   - EM 略高，但 F1 明显下降。
   - 后续应作为消融项继续观察，或与 KV error 重新混合。

## 五、下周计划

1. 优化 MRQA 答案后处理。
   - 增加对完整句子中短答案 span 的抽取。
   - 降低 EM/F1 对模型输出格式的敏感性。

2. 为 `eval_mrqa_small.py` 增加随机采样与 seed。
   - 支持从 SQuAD dev 中随机抽取 context 和 QA。
   - 支持固定 seed 复现实验。

3. 进行更大规模评测。
   - SQuAD dev 500 到 1000 题。
   - 对 NewsQA、HotpotQA、NaturalQuestionsShort 做小样本横向验证。

4. 准备小规模 LoRA SFT。
   - 训练 TinyLlama 输出 MRQA 风格短答案。
   - 先使用 SQuAD train 的 5k 到 20k QA。
   - SFT 后重新比较 baseline 与各量化模式。

5. 继续改进量化策略。
   - 以 `KV+Hidden mix` 为主线。
   - 对 hidden-only 进行权重消融。
   - 后续再考虑 layer + token block 级别的 bit 分配。
