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