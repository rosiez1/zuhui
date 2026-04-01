# KV Cache Compression 研究现状综述

KV cache compression 的研究目标，是在尽量不损伤长上下文建模能力的前提下，降低大模型推理阶段的 **显存占用、带宽压力与时延开销**。从现有工作的研究脉络来看，这一方向大致经历了三个阶段：**量化压缩阶段**、**token 级选择/丢弃阶段**，以及进一步面向 **head / layer 的细粒度结构化压缩阶段**。[1][2]

## 1. 第一阶段：以量化为核心的 KV 数值压缩

这一阶段的基本思想是：**尽量保留完整的 KV 结构，不删除 token，只降低 K/V 张量的存储精度**。这类方法对原始推理流程改动较小，工程部署相对直接，因此成为 KV cache compression 的一条主线。[1][3][4]

### 1.1 代表性方法

- **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**  
  该工作系统分析了 Key 与 Value 在分布特性上的差异，提出 **Key 按 channel 量化、Value 按 token 量化** 的非对称 2bit 方案，是 KV cache 超低比特量化中的代表性方法。[1]

- **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization**  
  该工作进一步推动了超长上下文场景下的 KV 量化，提出 **pre-RoPE quantization、per-channel key quantization、non-uniform quantization、dense-and-sparse quantization** 等设计，强调在极长上下文下兼顾量化精度与可扩展性。[3]

- **QAQ: Quality Adaptive Quantization for LLM KV Cache**  
  该方法从 **质量自适应量化** 出发，结合 Key/Value 的不同敏感性与 outlier 处理机制，在高压缩率下保持较小性能退化，属于实用性较强的 KV 量化方案。[4]

- **GEAR: An Efficient KV Cache Compression Recipe for LLMs**  
  GEAR 不仅使用超低比特量化，还结合误差恢复思想来减少自回归过程中误差累积，体现出“**量化 + 误差补偿**”这一更稳健的路线。[5]

### 1.2 阶段特点

这一阶段的优点是：**不改变 token 数量与注意力拓扑，兼容性较强**；缺点则在于：当压缩比继续提高时，量化误差会逐步成为瓶颈，且仅靠数值压缩往往难以从根本上降低长序列 attention 的访存负担。[1][3]

---

## 2. 第二阶段：以 SnapKV 为代表的 token 级选择/丢弃

随着研究深入，学界逐渐意识到：KV 冗余不仅体现在“每个向量存得太精细”，也体现在“**并非所有历史 token 都值得长期保留**”。因此，第二阶段的核心转向为：**在 token 维度上做选择性保留，只缓存更重要的上下文 token**。[6][7]

### 2.1 早期代表：重要 token 保留 / eviction

- **Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time**  
  该工作提出“**重要性持续性**”假设，认为早期被高度关注的 token 在后续生成中往往仍然重要，因此可以在固定 budget 下保留“关键 token”，对其他 token 做概率性丢弃。[6]

- **H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**  
  H₂O 提出 **heavy hitters** 概念，认为少数 token 对注意力贡献显著，应优先保留；其核心是构造兼顾 **近期 token** 与 **高贡献 token** 的动态 eviction 策略。[7]

### 2.2 代表性成熟方法：SnapKV

- **SnapKV: LLM Knows What You are Looking for Before Generation**  
  SnapKV 是 token 级 KV 压缩的代表性工作。其核心思想是：利用生成前 observation window 中的注意力模式，对历史 token 进行打分与聚合，从而筛出更值得保留的 token。它之所以影响较大，是因为它把“**基于注意力的重要 token 选择**”做成了较强、较通用、且无需微调的方案。[8]

### 2.3 阶段特点

相较于量化方法，这一阶段开始直接减少 **缓存长度**，因此在显存压缩之外，也更有机会降低后续 attention 的访存与计算成本。但其局限也很明显：**一旦 token 被丢弃，其信息往往不可恢复**，因此压缩策略对重要性判断的准确度高度敏感。[6][7][8]

---

## 3. 第三阶段：面向 head / layer 的更细粒度结构化压缩

进一步的研究发现，KV cache 的冗余并不只在 token 维度上存在，还存在于 **不同注意力头、不同网络层之间**。也就是说，“某个 token 是否重要”并不是一个全局统一判断，而可能取决于 **它位于哪一层、被哪个 head 使用**。因此，近期工作开始从单纯 token 级裁剪，转向 **head-wise / layer-wise / structured compression**。[9][10][11]

### 3.1 代表性方法

- **Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs**  
  该工作通过 profiling 分析注意力头的行为模式，区分更偏局部上下文的 head 与更依赖长程信息的 head，并据此做 **自适应 eviction**。这类方法的重要意义在于：它不再把所有 head 视为同质结构，而是开始显式利用 **head 间功能差异**。[9]

- **RazorAttention: Efficient KV Cache Compression Through Retrieval Heads**  
  RazorAttention 明确提出：大多数 attention heads 主要关注局部上下文，只有少数 **retrieval heads** 才真正承担全局检索功能。因此，它对不同 head 采用不同缓存策略：**对 retrieval heads 保留完整 cache，对非 retrieval heads 更激进地压缩远程 token**。这是 head-wise 压缩的代表性工作。[10]

- **PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling**  
  PyramidKV 进一步把结构差异扩展到 **layer 维度**。该工作观察到信息在层间传播具有“金字塔式汇聚”现象：低层注意力更分散，高层更集中，因此提出 **低层保留更多 cache、高层保留更少 cache** 的动态层间预算分配方式，是 layer-wise KV compression 的代表性方法。[11]

- **FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation**  
  FastKV 更进一步，不仅关注显存压缩，也显式优化 **prefill 阶段的时延与吞吐**。它通过 **Token-Selective Propagation** 在不同层传播不同规模的上下文信息，并结合 GQA-aware 压缩，使 KV compression 从“省显存”走向“同时省计算”。[12]

- **CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences**  
  CAKE 将 KV cache eviction 建模为“cake-slicing problem”，强调不同层由于注意力模式不同，应当进行 **layer-specific preference** 驱动的动态预算分配。该方法同时考虑空间与时间维度上的注意力变化，并以级联式方式管理缓存预算，是动态 layer budget 分配方向的重要代表方法。[13]

- **Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference**  
  Ada-KV 指出既有 Top-k eviction 方法往往在各 attention heads 上采用**均匀预算分配**，忽略了不同 head 的注意力集中程度差异。为此，Ada-KV 提出 **head-wise adaptive budget allocation**，通过将预算从稀疏 head 重新分配给注意力更分散的 head，以降低 eviction loss，并可插拔地增强 SnapKV、PyramidKV 等方法。[14]

- **DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs**  
  DynamicKV 观察到不同任务在不同层上的 token retention pattern 存在显著差异，因此提出 **task-aware 的 layer-wise adaptive retention**。与固定 retention pattern 或固定金字塔结构不同，DynamicKV 会根据任务特征动态调整各层的 KV cache 保留规模，突出 layer-wise 自适应压缩思路。[15]

- **LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation**  
  LAVa 从 Transformer residual stream 中的信息流损失出发，构建统一的 cache eviction 框架，并同时实现 **dynamic head budgets** 与 **dynamic layer budgets**。相较于仅在 head 或 layer 单一维度做自适应分配的方法，LAVa 进一步实现了更完整的 **head-layer 联合动态预算分配**，是第三阶段中较具代表性的统一化方法。[16]

### 3.2 阶段特点

这一阶段的核心思想是：**压缩决策的粒度从 token 扩展到了 token-head-layer 的多维结构**。相比早期“一整个 token 留或删”的粗粒度方法，这类结构化方法更符合 Transformer 内部不同层、不同头的功能异质性，因此往往能在相同压缩预算下获得更好的性能。与此同时，这类方法的系统实现也更复杂，对 cache layout、kernel 设计与部署友好性提出了更高要求。[10][13][14][16]

---

## 4. 研究脉络总结

总体来看，KV cache compression 的演化路径可以概括为：

1. **量化阶段**：重点解决“每个 KV 向量如何存得更省”；  
2. **token 级裁剪阶段**：重点解决“哪些 token 的 KV 值得保留”；  
3. **结构化细粒度阶段**：进一步解决“同一个 token 在不同 layer / head 中是否应被区别对待”。[1][8][10][11]

在第三阶段内部，研究又进一步分化为几个更细的方向：一类方法强调 **head-wise adaptive allocation**，如 Ada-KV；一类方法强调 **dynamic layer budgets**，如 CAKE 与 DynamicKV；还有一类方法尝试同时在 **head 与 layer 两个维度** 上实现联合动态预算分配，如 LAVa。[13][14][15][16]

因此，当前研究现状已经从单纯的“低比特存储”发展到“**面向 Transformer 内部结构异质性的选择性压缩**”。这也意味着，未来更有潜力的方向，往往不是单一量化或单一裁剪，而是 **量化、token 选择、head/layer 结构建模以及系统实现优化的联合设计**。[5][10][12][16]

---

## 参考文献

[1] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.  
https://arxiv.org/abs/2402.02750

[2] A Survey of Context Compression for Large Language Models.  
https://arxiv.org/abs/2410.06251

[3] KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.  
https://arxiv.org/abs/2401.18079

[4] QAQ: Quality Adaptive Quantization for LLM KV Cache.  
https://arxiv.org/abs/2403.04643

[5] GEAR: An Efficient KV Cache Compression Recipe for LLMs.  
https://arxiv.org/abs/2403.05527

[6] Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.  
https://arxiv.org/abs/2305.17118

[7] H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.  
https://arxiv.org/abs/2306.14048

[8] SnapKV: LLM Knows What You are Looking for Before Generation.  
https://arxiv.org/abs/2404.14469

[9] Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs.  
https://arxiv.org/abs/2310.01801

[10] RazorAttention: Efficient KV Cache Compression Through Retrieval Heads.  
https://arxiv.org/abs/2407.15891

[11] PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.  
https://arxiv.org/abs/2406.02069

[12] FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation.  
https://arxiv.org/abs/2502.01068

[13] CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences.  
https://arxiv.org/abs/2503.12491

[14] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference.  
https://arxiv.org/abs/2407.11550

[15] DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs.  
https://aclanthology.org/2025.findings-emnlp.737/

[16] LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation.  
https://arxiv.org/abs/2509.09754

## 附：代表性方法是否为 Training-Free

> 说明：此处“Training-Free”指**无需重新训练原始大语言模型参数，可直接在推理阶段部署**。  
> “明确是”表示论文正文/摘要中直接强调 tuning-free、fine-tuning-free、training-free 或 without re-training；  
> “基本是”表示论文未必显式使用该标签，但从方法机制看属于推理时压缩/驱逐策略，不涉及重训原模型参数。

| 方法 | 是否 Training-Free | 说明 |
|---|---|---|
| KIVI | 明确是 | 论文标题和摘要直接强调 **tuning-free** 2bit KV cache quantization |
| KVQuant | 基本是 | 属于 KV quantization 方法，论文主要讨论量化设计与推理部署，通常按无需重训原模型理解 |
| QAQ | 基本是 | 属于质量自适应量化方案，未明显强调 training-free 标签，但方法形态上不依赖重训原模型 |
| GEAR | 基本是 | 属于量化 + 误差补偿框架，主要是推理阶段的压缩与恢复设计，通常按无需重训原模型理解 |
| Scissorhands | 基本是 | 属于测试时 KV cache compression / eviction 方法，核心是基于重要性持续性做推理时保留与丢弃 |
| H₂O | 基本是 | 属于 heavy-hitter eviction 策略，基于推理时 attention/缓存贡献进行驱逐，不依赖重训原模型 |
| SnapKV | 明确是 | 论文明确写为 **fine-tuning-free** 方法 |
| FastGen / Model Tells You What to Discard | 明确是 | 论文明确写为 plug-and-play，且 **without resource-intensive fine-tuning or re-training** |
| RazorAttention | 明确是 | 论文摘要明确写为 **training-free KV cache compression algorithm** |
| PyramidKV | 基本是 | 属于动态 layer-wise KV cache compression，方法本身是推理期预算/缓存分配，并非重训原模型 |
| FastKV | 基本是 | 论文核心是 Token-Selective Propagation 与 GQA-aware compression，按方法形态属于推理时压缩 |
| CAKE | 明确是 | LAVa 论文将 CAKE描述为此前 **the only training-free method with dynamic layer budgets** |
| Ada-KV | 基本是（接近明确） | 论文强调 **plug-and-play** 的 head-wise adaptive budget allocation，可无缝集成现有 eviction 方法，通常按无需重训原模型理解 |
| DynamicKV | 基本是 | 方法是在 prefill / inference 阶段动态调整各层 token retention；论文未像 LAVa 那样直接反复强调 training-free 标签 |
| LAVa | 明确是 | 论文明确写为 **the first training-free method to achieve dynamic budget allocation** |

总体而言，现有 KV cache compression 方法大多属于 **training-free / inference-time optimization** 范式，即不重新训练原始 LLM，而是在推理阶段通过量化、token eviction、head-wise 或 layer-wise 动态预算分配等策略压缩 KV cache。相比之下，这一方向目前较少依赖额外训练一个独立压缩器或重新微调主模型参数。