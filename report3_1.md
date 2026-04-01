# 上周遗留工作
量化的精度是从float16降低到int8

# Q-KVComm 论文复现汇报

## 1. 当前复现进度概述

目前项目已完成前两阶段，并已进入第三阶段 `Hybrid Information Extraction` 的工程实现与实验验证。

已完成内容如下：

1. `KV cache` 的提取、保存与注入。
2. `uniform 8bit` 基线实现与验证。
3. 自适应分层量化 `adaptive layer-wise quantization`。
4. 真实 `bit-packing` 存储。
5. 单样本对比、多策略消融、多样本批量评测。
6. 第三阶段 `hybrid information extraction` 的完整工程接入。

当前实验环境如下：

- 模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Transformer 层数：`22`
- 典型 KV 形状：`(1, 4, seq_len, 64)`

---

## 2. 前两阶段的结论回顾

前两阶段已经比较稳定，当前结论明确：

- `uniform 8bit` 是稳定基线。
- 多样本评测中，`uniform` 的 `exact_match_rate = 1.0`。
- `adaptive_876_default` 是当前最优自适应量化方案。
- `adaptive_864_default` 的压缩更强，但质量下降明显。

最新多样本批量结果中：

| 方法 | exact_match_rate | avg_similarity_ratio | avg_total_payload_bytes |
|---|---:|---:|---:|
| uniform | 1.0 | 1.0000 | 238972.8 |
| adaptive_876_default | 0.8 | 0.9093 | 210480.0 |
| adaptive_865_default | 0.2 | 0.7825 | 190128.0 |
| adaptive_864_default | 0.0 | 0.6726 | 181987.2 |

因此第三阶段的目标不是继续优化第二阶段，而是验证：

> 在更强压缩的 KV 传输条件下，额外发送少量显式关键信息，能否恢复语义质量，同时保持总通信成本较低。

---

## 3. 第三阶段目前做了什么

### 3.1 总体思路

第三阶段采用“主干 KV 压缩 + 旁路显式信息”的混合传输思路：

- 主干仍然使用已有的 `adaptive quantized KV`
- 额外提取少量 `side information`
- 将 `side information` 压成极短文本
- 与接收端的 `next_text` 一起输入模型，继续走现有的 KV 注入续写流程

也就是说，目前第三阶段没有修改 Transformer 内部结构，而是先以最小工程风险验证 hybrid 思路是否有效。

### 3.2 实现形式

当前新增了一个独立的 `extraction/` 模块，负责从 `extract_text` 中抽取显式信息。核心文件包括：

- `extraction/content_router.py`
- `extraction/yake_extractor.py`
- `extraction/pattern_extractor.py`
- `extraction/entity_extractor.py`
- `extraction/fact_schema.py`
- `extraction/fact_serializer.py`
- `extraction/hybrid_pipeline.py`
- `extraction/fusion_prompt.py`

另外新增了第三阶段实验脚本：

- `compare_hybrid.py`
- `ablate_hybrid.py`

并扩展了原有评测和统计模块：

- `batch_quantization_eval.py`
- `evaluation/communication_cost.py`
- `evaluation/report_utils.py`
- `evaluation/batch_summary.py`
- `core/kv_injector.py`
- `core/model_loader.py`

---

## 4. Hybrid 混合信息的策略

### 4.1 整体流程

当前 hybrid 处理链路如下：

1. 对 `extract_text` 做内容类型判断。
2. 根据内容类型，分别执行关键词抽取、模式抽取、实体抽取。
3. 将抽取结果统一组织成 `fact`。
4. 对 `fact` 去重、排序、截断，生成短摘要。
5. 将该摘要作为 `side information`，与量化后的 KV 一起传输。
6. 接收端将 `side information` 以“注释”的形式放在 `next_text` 之前，再进行 KV 注入和续写。

### 4.2 内容路由策略

在 `content_router.py` 中，当前将输入内容粗分为三类：

- `general`
- `structured`
- `entity_heavy`

判别依据主要是：

- 是否包含大量数字、URL、参数、字段、版本号
- 是否包含 API / endpoint / rate limit / header / json 等结构化提示词
- 是否包含较多大写专有名词或实体短语

若结构化线索足够强，则判为 `structured`；若实体线索更强，则判为 `entity_heavy`；否则判为 `general`。

### 4.3 信息抽取策略

当前实现了三类抽取器：

#### 1. 关键词抽取

- 文件：`extraction/yake_extractor.py`
- 优先尝试 `YAKE`
- 若环境中未安装 `yake`，则退化为轻量 fallback 关键词抽取

#### 2. 模式抽取

- 文件：`extraction/pattern_extractor.py`
- 主要通过正则表达式抽取：
  - URL
  - API 路径
  - 版本号
  - 数值参数
  - 约束句
  - 结构化字段

#### 3. 实体抽取

- 文件：`extraction/entity_extractor.py`
- 优先尝试 `spaCy`
- 若不可用，则退化为启发式实体抽取

### 4.4 统一 fact 表示

不同抽取器的结果最终统一成如下结构：

```python
{
    "type": "keyword | entity | pattern | number | concept | constraint | field | url | api_path | version",
    "content": "...",
    "confidence": 0.0,
    "metadata": {...}
}
```

随后会统一执行：

- 去重
- 排序
- 截断
- 序列化

### 4.5 fact 排序策略

在 `fact_serializer.py` 中，目前对 fact 的优先级定义为：

- `constraint`、`field` 优先级最高
- `url`、`api_path`、`version` 次高
- `entity`、`concept` 其后
- `keyword` 再后
- `number` 最低

排序时综合考虑：

1. fact 类型优先级
2. 置信度
3. 内容长度

之后在最大字符数限制下拼接成一个非常短的 `facts_text`。

---

## 5. Hybrid 信息是通过什么形式实现的

### 5.1 当前默认配置

当前默认 hybrid 配置在 `configs/hybrid_default.json` 中，关键参数为：

- `adaptive_variant = adaptive_864_default`
- `keyword_top_k = 6`
- `pattern_top_k = 6`
- `entity_top_k = 6`
- `final_fact_top_k = 8`
- `max_summary_chars = 240`
- `min_fact_confidence = 0.7`
- `fusion_mode = comment_slashline`

这意味着当前默认实验是：

- 主干 KV 使用 `adaptive_864_default`
- side info 最多保留 8 条高置信 fact
- 生成的摘要最多 240 字符
- 接收端采用“单行注释风格”融合

### 5.2 side information 的生成方式

在 `hybrid_pipeline.py` 中，流程是：

1. `detect_content_type(text)`
2. 按配置执行 `extract_keywords / extract_patterns / extract_entities`
3. `deduplicate_facts`
4. `rank_facts`
5. 按 `max_summary_chars` 生成 `facts_text`
6. 调用 `build_comment_summary()` 变成真正发送的 `summary_text`
7. 统计 `side_info_bytes = len(summary_text.encode("utf-8"))`

### 5.3 发送内容的组成

当前 hybrid payload 由两部分组成：

1. `quantized KV`
2. `summary_text`

评测时统计：

- `quantized_payload_bytes`
- `side_info_bytes`
- `total_payload_bytes`
- `total_compression_ratio`

这样可以避免只看 KV 压缩率而忽略 side info 带来的额外开销。

---

## 6. 拼接格式具体长什么样

这是第三阶段目前最关键的实现点之一。

### 6.1 设计原则

经过多轮实验后，当前拼接遵循两个原则：

1. `summary` 必须像注释，而不能像新指令
2. `next_text` 必须是最后一段

这样做的原因是：

- 如果 side info 写得像指令，模型会明显偏向“解释模式”
- 如果 `next_text` 不在最后一段，会扰乱 continuation 的局部分布

### 6.2 当前默认拼接格式

默认 `fusion_mode = comment_slashline`，格式为：

```text
// fact_1; fact_2; fact_3

<next_text>
```

也就是：

1. 先把抽取出的关键信息压成一行或多行注释
2. 然后空一行
3. 最后一段保留原始 `next_text`

### 6.3 代码中的拼接逻辑

当前 `fusion_prompt.py` 中的实际逻辑是：

```python
return f"{comment_summary}\n\n{base_next_text}"
```

也就是说，注释摘要始终放在前面，而真实 continuation 前缀 `next_text` 始终放在最后一段。

### 6.4 当前支持的注释样式

目前支持三种注释风格：

#### 1. 单行斜杠注释

```text
// transformer model; key-value cache

In one sentence, it helps
```

#### 2. 单行井号注释

```text
# transformer model; key-value cache

In one sentence, it helps
```

#### 3. 块注释

```text
/* transformer model; key-value cache */

In one sentence, it helps
```

当前默认使用的是第 1 种，即 `comment_slashline`。

### 6.5 一个真实实验样例

在 `compare_hybrid.py` 的默认样例中：

- `extract_text`：`Explain what a key-value cache is in a transformer model.`
- `next_text`：` In one sentence, it helps`

抽取得到的 side info 为：

```text
// transformer model; key-value cache
```

最终送入接收端的融合文本为：

```text
// transformer model; key-value cache

In one sentence, it helps
```

对应的传输统计为：

- `quantized_payload_bytes = 137392`
- `side_info_bytes = 37`
- `total_payload_bytes = 137429`

---

## 7. 当前第三阶段已经实现了什么效果

### 7.1 工程上已经实现的效果

目前第三阶段已经不是概念设计，而是完成了可运行的工程接入：

- hybrid 信息抽取模块已接入
- side info 已纳入统一 payload
- continuation 评测口径已修正为只比较生成续写部分，而不是把 prompt 一起算进去
- 支持单样本对比
- 支持 hybrid 消融
- 支持多样本批量评测
- 支持排行榜汇总

### 7.2 当前多样本效果

最新 batch 结果显示：

| 方法 | exact_match_rate | avg_similarity_ratio | avg_total_payload_bytes |
|---|---:|---:|---:|
| uniform | 1.0 | 1.0000 | 238972.8 |
| adaptive_876_default | 0.8 | 0.9093 | 210480.0 |
| adaptive_864_default | 0.0 | 0.6726 | 181987.2 |
| hybrid_864_patterns_only | 0.0 | 0.6726 | 181987.2 |
| hybrid_864_keywords_patterns_hash | 0.0 | 0.5603 | 182040.8 |
| hybrid_864_keywords_patterns_block | 0.0 | 0.5399 | 182044.8 |
| hybrid_864_long_summary | 0.0 | 0.5018 | 182088.0 |
| hybrid_864_keywords_only | 0.0 | 0.4631 | 182041.8 |
| hybrid_864_keywords_patterns_slash | 0.0 | 0.4631 | 182041.8 |
| hybrid_876_keywords_patterns_slash | 0.2 | 0.7216 | 210534.6 |

### 7.3 结果展示
```json
baseline_unquantized
      "generated_text": " In one sentence, it helps to store and retrieve data in a way that is efficient and fast.\n\nExplain how a",
      "generated_token_count": 28,
      "generated_continuation_text": "to store and retrieve data in a way that is efficient and fast.\n\nExplain how a",

uniform:
      "generated_text": " In one sentence, it helps to store and retrieve data in a way that is efficient and fast.\n\nExplain how a",
      "generated_token_count": 28,
      "generated_continuation_text": "to store and retrieve data in a way that is efficient and fast.\n\nExplain how a",

hybrid_864_keywords_only
        "generated_text": "// transformer model; key-value cache\n\nIn one sentence, it helps to explain what a key-value cache is in a transformer model.\n\n:prepenses",
        "generated_token_count": 38,
        "generated_continuation_text": "to explain what a key-value cache is in a transformer model.\n\n:prepenses",

hybrid_864_patterns_only
        "generated_text": " In one sentence, it helps to store and retrieve data from a database.\n- 1. Explain the concept of a",
        "generated_token_count": 28,
        "generated_continuation_text": "to store and retrieve data from a database.\n- 1. Explain the concept of a",

hybrid_864_keywords_patterns_slash
        "generated_text": "// transformer model; key-value cache\n\nIn one sentence, it helps to explain what a key-value cache is in a transformer model.\n\n:prepenses",
        "generated_token_count": 38,
        "generated_continuation_text": "to explain what a key-value cache is in a transformer model.\n\n:prepenses",

hybrid_864_keywords_patterns_hash
        "generated_text": "# transformer model; key-value cache\n\nIn one sentence, it helps to store and retrieve data in a transformer model. A key-value cache is a data structure",
        "generated_token_count": 38,
        "generated_continuation_text": "to store and retrieve data in a transformer model. A key-value cache is a data structure",

hybrid_864_keywords_patterns_block
        "generated_text": "/* transformer model; key-value cache */\n\nIn one sentence, it helps to store and retrieve data in a transformer model. A key-value cache is a data structure",
        "generated_token_count": 39,
        "generated_continuation_text": "to store and retrieve data in a transformer model. A key-value cache is a data structure",

hybrid_876_keywords_patterns_slash
        "generated_text": "// transformer model; key-value cache\n\nIn one sentence, it helps to store and retrieve data in a way that is easy to access and manipulate. A key-value",
        "generated_token_count": 38,
        "generated_continuation_text": "to store and retrieve data in a way that is easy to access and manipulate. A key-value",

hybrid_864_short_summary
        "generated_text": "// transformer model; key-value cache\n\nIn one sentence, it helps to explain what a key-value cache is in a transformer model.\n\n:prepenses",
        "generated_token_count": 38,
        "generated_continuation_text": "to explain what a key-value cache is in a transformer model.\n\n:prepenses",

hybrid_864_long_summary
        "generated_text": "// transformer model; key-value cache\n\nIn one sentence, it helps to explain what a key-value cache is in a transformer model.\n\n:prepenses",
        "generated_token_count": 38,
        "generated_continuation_text": "to explain what a key-value cache is in a transformer model.\n\n:prepenses",
```
---

## 8. 当前结论

### 8.1 已经得到的明确结论

1. 第三阶段 hybrid 框架已经搭建完成，并已进入系统评测。
2. 从工程角度看，`side information + quantized KV` 的联合传输已实现。
3. 当前 hybrid 的通信统计口径已经完整，能够同时比较质量与总 payload。

### 8.2 当前尚未达到的目标

目前 hybrid 还没有超过纯 adaptive 方案。

尤其是：

- `adaptive_876_default` 仍然是当前综合效果最好的压缩方案
- `hybrid_864_*` 尚未实现“用更少总比特恢复更多质量”的目标

### 8.3 目前最重要的现象

实验表明：

> 只要显式 side information 进入同一段输入流，即使写成注释，也仍然可能扰乱 continuation 的生成分布。

换句话说，当前瓶颈已经不是“有没有抽取出信息”，而是：

> 如何让 side information 被模型利用，同时又不破坏原有续写分布。

这也是当前第三阶段尚未取得明显收益的主要原因。

---

## 9. 当前问题与下一步方向

从现有结果看，第三阶段下一步更值得做的是：

修改测评方法，现有的 `extract_text` 有点短了，上下文信息不足，导致抽取的 side information 可能并不关键，因此可以考虑：
- 设计更长的 `extract_text`，让其中包含更多关键信息
- 设计更复杂的 `extract_text`，让其中包含一些容易被扰乱的信息

---

## 10. 代码与实验产物位置

当前与第三阶段最相关的文件如下：

### 代码

- `extraction/hybrid_pipeline.py`
- `extraction/fusion_prompt.py`
- `extraction/fact_serializer.py`
- `extraction/content_router.py`
- `compare_hybrid.py`
- `ablate_hybrid.py`
- `batch_quantization_eval.py`

### 配置

- `configs/hybrid_default.json`
- `configs/hybrid_ablation.json`

### 实验报告

- `quantized_payloads/hybrid_compare_report.json`
- `quantized_payloads/hybrid_ablation_report.json`
- `outputs_batch/batch_quantization_report.json`

---

## 11. 总结

目前 Q-KVComm 的第三阶段 hybrid information extraction 已经实现了完整工程链路和批量评测，但当前结果表明：显式 side information 虽然已经能被提取、压缩和传输，但在 continuation 任务中容易扰乱生成分布。因此下一步更重要的不是继续优化抽取策略，而是设计更合理的 `extract_text` 和 `fusion_mode`，让 side information 能真正成为“有用的提示”，而不是“干扰项”。