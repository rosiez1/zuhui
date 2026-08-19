# 大模型 Hidden State、跨模型潜空间通信与相关工作梳理

## 1. Hidden State 在大模型中是什么

在 Transformer 中，hidden state 并不特指某一层，而是指某个 token 在某一层经过计算后的内部表示。

若模型共有 \(L\) 个 Transformer Block，则可写为：

\[
h_t^{(0)} \rightarrow h_t^{(1)} \rightarrow \cdots \rightarrow h_t^{(L)}
\]

其中：

- \(h_t^{(0)}\)：进入第一个 Transformer Block 前的表示；
- \(h_t^{(l)}\)：第 \(l\) 个 Transformer Block 输出的表示；
- \(h_t^{(L)}\)：最后一个 Transformer Block 的输出；
- 对现代 decoder-only LLM，还常有一个 final RMSNorm，其输出再送入 LM Head。

对于一整个长度为 \(T\) 的序列，第 \(l\) 层的 hidden states 为：

\[
H^{(l)}=[h_1^{(l)},h_2^{(l)},\dots,h_T^{(l)}]\in\mathbb{R}^{T\times d}
\]

其中 \(d\) 是 hidden dimension。

---

## 2. \(h^{(0)}\) 和位置编码

### 经典 Transformer

在采用绝对位置编码的模型中：

\[
h_t^{(0)} = E_{\text{token}}(x_t)+E_{\text{pos}}(t)
\]

因此 \(h^{(0)}\) 可以理解为 token embedding 加位置编码。

### Qwen / LLaMA 等现代 LLM

Qwen、LLaMA 等模型常使用 RoPE，因此更接近：

\[
h_t^{(0)} = E_{\text{token}}(x_t)
\]

位置相关信息随后通过 Attention 内部的 \(Q,K\) 注入：

\[
Q'=\operatorname{RoPE}(Q), \qquad K'=\operatorname{RoPE}(K)
\]

因此在这些模型中，不能简单写成：

\[
h^{(0)}=\text{Embedding}+\text{Position Embedding}
\]

---

## 3. Qwen Block 中 hidden state 从哪里出来

Qwen 类模型通常采用 decoder-only、Pre-Norm、RoPE 架构。

一个 Qwen Block 可概括为：

\[
x_l
\rightarrow \operatorname{RMSNorm}
\rightarrow \operatorname{Attention}
\rightarrow +x_l
\rightarrow \operatorname{RMSNorm}
\rightarrow \operatorname{MLP}
\rightarrow +\text{residual}
\rightarrow x_{l+1}
\]

具体可写为：

\[
\bar{x}_l=\operatorname{RMSNorm}(x_l)
\]

\[
a_l=\operatorname{Attention}(\bar{x}_l)
\]

\[
x'_l=x_l+a_l
\]

\[
\tilde{x}_l=\operatorname{RMSNorm}(x'_l)
\]

\[
m_l=\operatorname{MLP}(\tilde{x}_l)
\]

\[
x_{l+1}=x'_l+m_l
\]

通常：

\[
\boxed{x_{l+1}}
\]

即整个 Qwen Block 最上方残差加法后的输出，被视为该层的 block-level hidden state。

整个模型可以看成：

```text
Token IDs
   ↓
Embedding
   ↓
H^(0)
   ↓
QwenBlock_0
   ↓
H^(1)
   ↓
QwenBlock_1
   ↓
H^(2)
   ↓
...
   ↓
QwenBlock_N
   ↓
H^(N+1)
   ↓
Final RMSNorm
   ↓
H_final
   ↓
LM Head / Linear
   ↓
Logits
```

需要区分：

\[
H^{(N+1)}
\]

是最后一个 Block 的输出，而：

\[
H_{\text{final}}=\operatorname{RMSNorm}(H^{(N+1)})
\]

才是真正送进 LM Head 的表示。

---

## 4. Hidden State、Logits 和概率的区别

三者关系：

\[
\boxed{
h_t^{(L)}
\rightarrow
\text{Final Norm}
\rightarrow
W_{\rm LM}
\rightarrow
z_t
\rightarrow
\operatorname{Softmax}
\rightarrow
p_t
}
\]

其中：

- \(h_t^{(L)}\)：hidden state；
- \(z_t\)：logits；
- \(p_t\)：token probability distribution。

它们不是同一个东西。

---

## 5. 一个模型的 Hidden State 能否直接给另一个模型

关键结论：

\[
\boxed{\text{Hidden State 不是天然通用的“跨模型语言”}}
\]

是否能直接工作取决于 Sender 和 Receiver 的表示接口是否兼容。

| Sender → Receiver | 是否可能直接工作 | 原因 |
|---|---:|---|
| 同一模型，第 \(l\) 层 → 第 \(l+1\) 层 | 可以 | 本来就是模型正常 forward 的接口 |
| 同一模型，最后层 → 第一层 | 通常不合理 | layer-space mismatch |
| 不同模型，同层 → 同层 | 通常不可靠 | latent coordinate system 不一致 |
| 不同模型 + adapter | 更合理 | 学习 A→B 的表示映射 |

需要区分三种兼容性：

\[
\boxed{
\text{shape compatible}
\neq
\text{distribution compatible}
\neq
\text{functionally compatible}
}
\]

即使两个模型都有：

\[
h_A,h_B\in\mathbb R^{4096}
\]

也只说明 shape 一样，并不说明两个表示空间等价。

---

## 6. 为什么需要 Adapter

不同模型可能以不同坐标系编码相似语义：

\[
h_A \neq h_B
\]

但可能存在一个简单映射：

\[
h_B \approx Wh_A+b
\]

因此可以设计：

\[
h_A^{(l)}
\rightarrow
g_{A\rightarrow B}
\rightarrow
\hat h_B^{(k)}
\]

再交给 Receiver。

Adapter 可以是：

- Linear；
- MLP；
- Attention；
- Transformer adapter；
- Cross-attention 模块。

从语义通信角度，比“让表示长得一样”更重要的是：

\[
\boxed{\text{Receiver 能不能正确利用这些信息}}
\]

因此 functional alignment 通常比单纯 representation matching 更关键。

---

## 7. Model Stitching 的核心思想

《Revisiting Model Stitching to Compare Neural Representations》的基本思想是：

将模型 A 的前半部分和模型 B 的后半部分接起来，中间仅训练一个低容量 stitcher：

\[
x
\rightarrow
A_{\le l}
\rightarrow
\boxed{s}
\rightarrow
B_{>l}
\rightarrow
y
\]

如果一个简单映射就能让拼接后的模型保持性能，则说明 A 和 B 的中间表示在功能上是兼容的。

它关注的是：

\[
\boxed{\text{Functional Compatibility}}
\]

而不只是：

\[
\boxed{\text{Geometric Similarity}}
\]

因此：

\[
H_A\neq H_B
\]

并不意味着两者不能被映射到相互可用的表示空间。

这个思想与跨模型 latent communication 非常相关。

---

## 8. PD 分离中为什么主要传 KV Cache

在 Prefill–Decode Disaggregation 中，Prefill 节点处理 prompt 并构造各层 KV Cache：

\[
\{K_l,V_l\}_{l=1}^{L}
\]

然后 Decode 节点继续逐 token 生成。

原因是生成第 \(T+1\) 个 token 时，每层 Attention 都需要历史 token 的：

\[
K_{1:T}^{(l)},V_{1:T}^{(l)}
\]

因此：

\[
\boxed{\text{KV Cache 是标准 autoregressive decoding 所需的计算状态}}
\]

而普通最后层 hidden state：

\[
H^{(L)}
\]

无法直接替代所有层的历史 KV。

---

## 9. 为什么只传最后一层 Hidden State 不能直接持续 Decode

如果电脑 A 拿到：

\[
h_T^{(L)}
\]

电脑 B 可以用 LM Head 算出一次 next-token logits：

\[
\text{logits}_{T+1}
=
W_{\text{LM}}\operatorname{Norm}(h_T^{(L)})
\]

因此：

\[
h_T^{(L)}
\rightarrow
\text{LM Head}
\rightarrow
x_{T+1}
\]

这一小步可以。

但生成下一个 token 后，B 的第 1 层 Attention 就需要：

\[
K_{1:T}^{(1)},V_{1:T}^{(1)}
\]

后续每层也需要对应 KV。

仅有：

\[
h_T^{(L)}
\]

无法恢复：

\[
\{K_l,V_l\}_{l=1}^{L}
\]

因此：

\[
\boxed{
\text{Last Hidden State}
\neq
\text{完整 Decoding State}
}
\]

所以：

- 标准 PD 分离：传 KV Cache；
- Pipeline Parallel：常传 hidden activation；
- Agent latent communication：可传 hidden-derived latent。

---

## 10. Interlat：直接用 Hidden State 做 Agent 间通信

《Enabling Agents to Communicate Entirely in Latent Space》提出 Interlat。

核心流程：

\[
\boxed{
\text{Reasoning Agent}
\rightarrow
\text{last-layer hidden-state trajectory}
\rightarrow
\text{communication adapter}
\rightarrow
\text{Actor Agent}
}
\]

Sender 在生成一段 reasoning/message 时，对每个生成步骤收集最后层 hidden state：

\[
H=[h_1,h_2,\dots,h_L]
\in\mathbb R^{L\times d}
\]

这些 hidden states 与原本的 token trajectory 时间对齐：

\[
h_1,h_2,\dots,h_L
\]

对应：

\[
y_1,y_2,\dots,y_L
\]

Receiver 输入形式：

\[
E=
[
e(x_1),\dots,e(x_m),
e(\langle bop\rangle),
g(H),
e(\langle eop\rangle)
]
\]

其中 \(g\) 是 communication adapter，包含轻量 self-attention / projection，用来完成 rescaling 和 interpretation。

### 关键点

Interlat 不是：

\[
H_A\rightarrow B\text{ 中间层}
\]

而是：

\[
\boxed{
g(H_A)
\rightarrow
B\text{ 的 embedding-level 输入接口}
}
\]

### 为什么 Receiver 不会直接忽略 Latent

它额外使用：

\[
\mathcal L_{\text{sep}}
\]

对 matched latent 和 mismatched latent 的输出分布做分离，以迫使 Receiver 真正利用 task-relevant latent information。

### 压缩阶段

Interlat 进一步训练 reasoning model 在 latent space 中直接生成压缩后的：

\[
H_K\in\mathbb R^{K\times d},
\qquad K\ll L
\]

通过把上一时刻 hidden state 经 projection 后直接反馈成下一步输入：

\[
h_i
\rightarrow
\operatorname{Proj}(h_i)
\rightarrow
E_{i+1}
\]

从而避免：

\[
\text{hidden}
\rightarrow
\text{LM Head}
\rightarrow
\text{token}
\rightarrow
\text{embedding}
\]

这一离散化过程。

---

## 11. THOUGHTCOMM：从多个 Agent 的 Hidden State 中提取“潜在思想”

《Thought Communication in Multiagent Collaboration》采用了另一条路线。

它不是直接把 hidden state 当作 thought，而是假设：

\[
\boxed{
H_t=f(Z_t)
}
\]

其中：

- \(H_t\)：多个 Agent 的 model states；
- \(Z_t\)：更底层的 latent thoughts；
- \(f\)：未知生成映射。

因此目标是：

\[
H_t
\rightarrow
\hat Z_t
\]

即从 hidden states 中恢复更紧凑、更结构化的 latent thoughts。

---

## 12. THOUGHTCOMM 的完整输入输出流程

整体流程：

```text
原始问题 / dialogue context
        ↓
多个 Agent 分别生成回答
        ↓
每个 Agent 取最后生成 token 的 model state
        ↓
H1, H2, ..., Hn
        ↓
拼接
        ↓
H = [H1; H2; ...; Hn]
        ↓
Sparse Autoencoder
        ↓
Z = [z1, z2, ..., zk]
        ↓
Jacobian dependency structure
        ↓
区分 shared / private thoughts
        ↓
为不同 Agent 做 selective routing
        ↓
Z~_i
        ↓
Adapter
        ↓
Prefix P_i
        ↓
P_i + textual context
        ↓
Agent i 下一轮生成
```

### 12.1 每个 Agent 提取什么

论文中每个 Agent 取：

\[
H_t^{(i)}
\]

即 communication round 前，最后生成 token 对应的 model state。

与 Interlat 不同：

- Interlat：取整条 hidden trajectory；
- THOUGHTCOMM：每个 Agent 每轮主要取一个 state vector。

然后：

\[
H_t=
[
H_t^{(1)};
H_t^{(2)};
\dots;
H_t^{(n)}
]
\]

---

## 13. Sparse Autoencoder 的作用

Encoder：

\[
\hat Z_t=\hat f^{-1}(H_t)
\]

Decoder：

\[
\hat H_t=\hat f(\hat Z_t)
\]

loss：

\[
\mathcal L_{\mathrm{rec}}
=
\|H_t-\hat f(\hat Z_t)\|_2^2
+
\|J_{\hat f}\|_1
\]

第一项保证 latent 能重构原 hidden states。

第二项对 decoder Jacobian 做稀疏约束：

\[
J_{\hat f}
=
\frac{\partial H}{\partial Z}
\]

通过 Jacobian 的非零模式判断：

> 某个 latent dimension \(z_j\) 会影响哪些 Agent 的 hidden states。

例如：

| latent | Agent 1 | Agent 2 | Agent 3 |
|---|---:|---:|---:|
| \(z_1\) | ✓ |  |  |
| \(z_2\) | ✓ | ✓ | ✓ |
| \(z_3\) |  | ✓ |  |
| \(z_4\) | ✓ | ✓ |  |

那么：

- \(z_2\)：shared thought；
- \(z_1\)：Agent 1 private thought；
- \(z_3\)：Agent 2 private thought。

---

## 14. THOUGHTCOMM 的选择性路由

对每个 latent thought：

\[
z_j
\]

定义其 agreement level：

\[
\alpha_j
=
\sum_{k=1}^{n_a}
I(z_j \text{ 与 Agent }k\text{相关})
\]

然后根据：

\[
\alpha_j
\]

将不同类型 latent 重新加权并组合，形成：

\[
\tilde Z_t^{(i)}
\]

即每个 Agent 自己的 personalized latent message。

这一步实际上类似：

\[
\boxed{\text{semantic-aware routing}}
\]

而不是无差别 broadcast。

---

## 15. THOUGHTCOMM 如何把 Latent 注入 Receiver

得到：

\[
\tilde Z_t^{(i)}
\]

后，通过 adapter：

\[
P_t^{(i)}=g(\tilde Z_t^{(i)})
\]

得到：

\[
P_t^{(i)}\in\mathbb R^{m\times d}
\]

再作为 continuous prefix 加到 Agent 输入 embedding 前：

\[
[
P_t^{(i)};
e(x_1);
e(x_2);
\dots;
e(x_T)
]
\]

然后从 Transformer 第一层开始正常 forward。

因此：

\[
\boxed{
\text{THOUGHTCOMM 不是 hidden state 直接接着 decode}
}
\]

而是：

\[
\boxed{
\text{latent}
\rightarrow
\text{prefix}
\rightarrow
\text{完整 forward}
}
\]

---

## 16. THOUGHTCOMM 中的文本消息到底有什么作用

文本并不是单纯给人看的。

原文明确保留 surface-level messages，并指出这些 messages 会 broadcast。

因此其系统更接近：

\[
\boxed{
\text{Text Communication}
+
\text{Latent Side Channel}
}
\]

文本至少有四个作用：

1. Agent 需要先生成 response，才能得到用于抽取的 model state；
2. surface-level textual messages 仍然参与 multi-agent dialogue；
3. Adapter 训练时使用短文本 continuation 和语义/流畅性约束；
4. 最终任务答案仍通过文本输出并评价。

所以 THOUGHTCOMM 更准确地说是：

\[
\boxed{\text{latent-augmented multi-agent communication}}
\]

而不是“完全取消文本通道”。

---

## 17. Interlat 和 THOUGHTCOMM 的主要区别

| 方法 | Sender 提取什么 | 中间表示 | Receiver 如何使用 |
|---|---|---|---|
| Interlat | 整条 CoT 对应的 last-layer hidden trajectory | \(H\in\mathbb R^{L\times d}\)，可继续压缩 | Adapter + latent prefix |
| THOUGHTCOMM | 每个 Agent 最后生成 token 的 model state | 多 Agent states 经 SAE 提炼出的 \(Z\) | Selective routing + Adapter + prefix |

Interlat 更直接：

\[
\boxed{
H_A
\rightarrow
B
}
\]

THOUGHTCOMM 更强调：

\[
\boxed{
(H_A,H_B,\dots)
\rightarrow
Z_{\text{shared/private}}
\rightarrow
A,B,\dots
}
\]

---

## 18. 对潜空间语义通信研究的启发

综合这些工作，可以把跨 Agent latent communication 拆成三级结构：

\[
\boxed{
\text{LLM Raw Hidden State}
\rightarrow
\text{Communicable Semantic Latent}
\rightarrow
\text{Receiver-Compatible Representation}
}
\]

进一步写成：

\[
H_A
\xrightarrow{C_A}
Z
\xrightarrow{\text{Channel}}
\hat Z
\xrightarrow{G_B}
\tilde H_B
\]

其中：

- \(C_A\)：semantic encoder / compressor；
- \(Z\)：semantic codeword；
- Channel：通信链路；
- \(G_B\)：semantic decoder / receiver adapter。

从传统语义通信角度，还可以进一步引入：

\[
R=\operatorname{bits}(Z)
\]

以及任务失真：

\[
D_{\text{task}}
=
\mathcal L(B(\hat Z),y)
\]

从而研究：

\[
\boxed{
\min R
\quad
\text{s.t.}
\quad
D_{\text{task}}\le\epsilon
}
\]

这比简单追求 hidden-state reconstruction 更接近 task-oriented semantic communication。

---

## 19. 当前几个核心研究问题

### 19.1 选哪一层 hidden state

需要决定：

\[
l\in\{1,\dots,L\}
\]

不同层可能包含不同程度的：

- 词法信息；
- 上下文信息；
- 抽象语义；
- 任务决策信息；
- 输出预测偏置。

最后一层并不一定是最佳通信层。

### 19.2 选哪些 token

可以比较：

\[
H^{(l)}\in\mathbb R^{T\times d}
\]

全部传输，或者只取：

\[
h_T^{(l)}
\]

也可以进一步学习：

- pooling；
- token selection；
- attention compression；
- latent bottleneck。

### 19.3 如何跨模型兼容

需要区分：

\[
\text{representation alignment}
\]

与：

\[
\text{functional alignment}
\]

最终更重要的是：

\[
\boxed{\text{Receiver 是否能利用 Sender 信息完成任务}}
\]

### 19.4 如何证明 Receiver 真的在用 Latent

必须做因果式对照，例如：

\[
H_{\text{correct}}
\]

vs.

\[
H_{\text{wrong}}
\]

vs.

\[
H_{\text{random}}
\]

vs.

\[
H_{\text{zero}}
\]

若：

\[
P_{\text{correct latent}}
\gg
P_{\text{wrong/random/zero}}
\]

才更能说明 Receiver 真正在利用 Sender latent content，而不是把 latent 当作 soft prompt 或噪声。

### 19.5 如何真正体现“通信效率”

Raw hidden state 通常 bit 数很大，因此不能只说 latent 比 token “信息丰富”。

需要真正比较：

\[
\text{bit rate}
\]

\[
\text{latency}
\]

\[
\text{task performance}
\]

\[
\text{robustness}
\]

并研究：

\[
\boxed{
\text{Rate} \leftrightarrow \text{Task Utility}
}
\]

之间的 trade-off。

---

## 20. 一句话总结

整个讨论可以归结为：

\[
\boxed{
\text{KV Cache 适合“继续计算”，
Hidden State 适合探索“传递语义信息”，
而真正可用的跨 Agent 通信表示，很可能还需要进一步的压缩、路由和 Receiver 适配。}
}
