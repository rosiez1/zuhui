![alt text](image.png)
## 1. 论文要解决的问题

现有多智能体 / 多模型系统通常通过**文本**进行通信：

\[
\text{Agent A hidden state}
\rightarrow
\text{token}
\rightarrow
\text{text}
\rightarrow
\text{token}
\rightarrow
\text{Agent B hidden state}
\]

这种方式会把模型内部本来连续、高维的表示压缩到离散 token 序列中，并且通信通常是**轮次式、串行式**的。

本文提出的问题是：

> 两个预训练语言模型能否不通过文本，而是直接利用各自的中间 hidden state，在生成过程中持续、双向地通信？

作者提出 **Bicameral Model**，核心结构为：

\[
M_p
\leftrightarrow
\phi
\leftrightarrow
M_a
\]

其中：

- \(M_p\)：Primary Model，负责读取用户问题并产生最终回答；
- \(M_a\)：Auxiliary Model，负责调用工具、执行计算、求解约束、运行代码；
- \(\phi\)：可训练 Neural Interface，用于在两个模型的 hidden-state space 之间传递和融合信息。

两个 LLM 的参数全部冻结，只训练接口 \(\phi\)。

---

## 2. 系统输入与输出

### 2.1 Primary Model 的输入与输出

Primary Model \(M_p\) 是用户直接交互的模型。

输入：

\[
\text{System Prompt} + \text{User Query}
\]

例如：

```text
What is 4 * 3?
```

输出：

```text
4 * 3 equals 12
```

Primary **不直接访问 calculator、Z3 或 Python 工具**，也不会看到 Auxiliary 使用的工具调用语法。

---

### 2.2 Auxiliary Model 的输入与输出

Auxiliary Model \(M_a\) 的输入与 Primary 不同。

它首先只读取自己的 tool instruction，例如：

```text
You may call a calculator using calc(expr)
```

随后它**不直接看到用户问题文本**，而是通过 Primary 的 hidden state 获取任务信息。

Auxiliary 输出的是工具调用，例如：

```text
calc(4*3)
```

工具执行后返回：

```text
=12;
```

这里的 `=12;` 不是 Auxiliary 自己预测的，而是由外部工具**强制写回** Auxiliary token stream。

因此整个信息流是：

\[
\boxed{
\text{User Query}
\rightarrow
M_p
\xrightarrow{\text{hidden state}}
M_a
\rightarrow
\text{Tool}
\rightarrow
M_a
\xrightarrow{\text{hidden state}}
M_p
\rightarrow
\text{Final Answer}
}
\]

---

## 3. 两个模型之间交互的媒介是什么？

两个模型之间传递的不是文本，也不是 token ID，而是 Transformer 中间层的：

\[
\boxed{\text{residual-stream hidden state}}
\]

作者定义：

\[
h_m^{(\ell)}(t)\in\mathbb R^{d_m}
\]

表示模型 \(m\in\{p,a\}\) 在 token 位置 \(t\)、第 \(\ell\) 个 Transformer block 执行之前的 residual-stream state。

其中：

- \(\ell=0\)：input embedding；
- \(\ell=L\)：最后一个 Transformer block 之后、final norm / LM head 之前的状态。

可以把 residual stream 理解为 Transformer 中从 embedding 一直流到最后一层的“主干 hidden state”：

\[
h^{(0)}
\rightarrow
\text{Block 1}
\rightarrow
h^{(1)}
\rightarrow
\text{Block 2}
\rightarrow
\dots
\rightarrow
h^{(L)}
\]

Bicameral 就是在这条主干上读写 hidden state。

---

## 4. 整体架构

### 4.1 原论文 Figure 1：整体结构

![alt text](image-1.png)
---

## 5. 一个 token 内，两个模型如何交互？

作者一共设置四个 layer index：

\[
\ell_r^{p\rightarrow a}
\]

Primary read layer；

\[
\ell_w^{p\rightarrow a}
\]

Auxiliary write layer；

\[
\ell_r^{a\rightarrow p}
\]

Auxiliary read layer；

\[
\ell_w^{a\rightarrow p}
\]

Primary write layer。

一次 token step 内的流程：

### Step 1：Primary 先运行到 read layer

Primary 当前 token 的 hidden state：

\[
h_p^{(\ell_r^{p\rightarrow a})}(t)
\]

被接口读取。

### Step 2：Primary → Auxiliary

接口将 Primary hidden state 翻译到 Auxiliary representation space：

\[
h_p
\rightarrow
f^{p\rightarrow a}(h_p)
\]

然后写入 Auxiliary residual stream：

\[
h_a^{(\ell_w^{p\rightarrow a})}
\]

### Step 3：Auxiliary 继续执行自身 Transformer

Auxiliary 接收到 Primary 信息之后继续经过若干 Transformer layers：

\[
h_a'
\rightarrow
M_a
\rightarrow
h_a^{new}
\]

因此返回给 Primary 的不是简单的 projector 输出，而是：

\[
\boxed{
\text{Primary signal}
+
\text{Auxiliary 自身计算}
}
\]

得到的结果。

### Step 4：Auxiliary → Primary

Auxiliary hidden state：

\[
h_a^{(\ell_r^{a\rightarrow p})}(t)
\]

再经过 reverse translator：

\[
f^{a\rightarrow p}
\]

注入 Primary：

\[
h_p^{(\ell_w^{a\rightarrow p})}(t)
\]

### Step 5：Primary 完成剩余层并预测下一个 token

最终：

\[
h_p'
\rightarrow
\text{LM Head}
\rightarrow
\text{next-token distribution}
\]

因此一个 token 内形成：

\[
\boxed{
M_p
\rightarrow
M_a
\rightarrow
M_p
}
\]

作者称这种有效的信息往返为：

\[
\text{within-token round trip}
\]

---

## 6. Figure 2：完整交互时序

![alt text](image-2.png)
系统分为三个阶段。

### Phase 1：Auxiliary Initialization

Auxiliary 先读取自己的 tool instruction：

```text
<s> Sys prompt ... <s> Assistant
```

此阶段：

- Primary 不运行；
- coupling 关闭；
- 目的主要是建立 Auxiliary 的 KV cache。

---

### Phase 2：Primary Input Processing

Primary 开始处理用户问题：

```text
What is 4 * 3?
```

这些 token 是 **input-forced**，不是 Primary 自己生成的。

与此同时 Auxiliary 并没有看到这句话，而是在自己的 stream 上生成 / 保持 wait tokens：

```text
• • • • •
```

但从这一阶段开始双向 hidden-state coupling 已开启。

Primary 每处理一个 token：

\[
h_p(t)
\rightarrow
\phi^{p\rightarrow a}
\rightarrow
h_a(t)
\]

因此 Auxiliary 可以通过 latent signal 逐渐“理解”：

```text
4 * 3
```

并开始生成：

```text
calc(4*3)
```

---

### Phase 3：Joint Generation

Primary 和 Auxiliary 都进入自由生成。

Auxiliary 完成：

```text
calc(4*3)
```

calculator 执行：

\[
4\times 3=12
\]

然后系统把：

```text
=12;
```

作为 **tool-forced tokens** 写回 Auxiliary。

这些 token 进入 Auxiliary 的 embedding 和后续 hidden state，随后通过 reverse coupling：

\[
M_a
\rightarrow
\phi^{a\rightarrow p}
\rightarrow
M_p
\]

Primary 最终生成：

```text
4 * 3 equals 12
```

注意：

\[
\boxed{
M_p \text{ 从未看到 } calc(4*3)=12 \text{ 这段文本}
}
\]

它只通过 hidden-state communication 接收到工具结果带来的影响。

---

## 8. Neural Interface 的结构

论文中最核心的 coupling operator 为：

\[
\phi^{p\rightarrow a}(h_p,h_a)
=
(1-\sigma^{p\rightarrow a})h_a
+
\sigma^{p\rightarrow a}f^{p\rightarrow a}(h_p)
\]

其中：

\[
\sigma^{p\rightarrow a}
=
\operatorname{Sigmoid}
\left(
g^{p\rightarrow a}(h_a)
\right)
\]

reverse direction 完全对称。

这里有两个核心模块：

### 8.1 Translation Network \(f\)

作用：

\[
\mathbb R^{d_p}
\rightarrow
\mathbb R^{d_a}
\]

即：

> 将 sender hidden state 翻译到 receiver 的 representation space。

---

### 8.2 Suppression Gate \(g\)

作用：

\[
h_a
\rightarrow
\sigma
\]

决定 receiver 接收多少 sender 信息。

若：

\[
\sigma\approx0
\]

则：

\[
h_a'\approx h_a
\]

表示 receiver 几乎不接受外部信息。

若：

\[
\sigma\approx1
\]

则：

\[
h_a'\approx f(h_p)
\]

表示大量接收 sender 信息。

由于 gate 的输入是 receiver 自己的 hidden state：

\[
\sigma=g(h_{\text{receiver}})
\]

因此作者称之为 **Pull design**：

> receiver 根据自身状态决定“我要不要拉取对方的信息”。

---

## 9. 两种 Neural Interface

### 9.1 PULLSTANDARD

Translation network：

\[
f=\text{MLP}
\]

Suppression gate：

\[
g=\text{MLP}
\]

Arithmetic 配置中，translation network 为：

\[
3L,\quad h=2048
\]

即可以近似理解为：

\[
896
\rightarrow
2048
\rightarrow
2048
\rightarrow
2048
\rightarrow
896
\]

gate network 使用较小的 MLP：

\[
3L,\quad h=64
\]

例如：

\[
896
\rightarrow
64
\rightarrow
64
\rightarrow
64
\rightarrow
1
\]

PULLSTANDARD 同时学习：

\[
\boxed{\text{怎么翻译}}
\]

和：

\[
\boxed{\text{什么时候通信}}
\]

---

### 9.2 PULLIDENTITY / Gated Identity

Identity 版本直接设：

\[
f(h)=h
\]

所以：

\[
h'_a
=
(1-\sigma)h_a+\sigma h_p
\]

不再训练 translator，只训练 gate。

这要求两个模型 hidden size 相同，并假设 representation space 足够兼容。

Figure 4 中的：

```text
Gated Identity
(SCALARIDENTITY)
```

就是这个版本。

它可以简单理解为：

\[
\boxed{
\text{不学习“怎么翻译”，只学习“门开多大”}
}
\]

---

## 10. Scalar Gate 与 Element-wise Gate

### Scalar Gate

输出一个标量：

\[
\sigma\in[0,1]
\]

整个 hidden vector 统一使用同一个通信强度：

\[
h'_i
=
(1-\sigma)h_i
+
\sigma s_i
\]

可以理解为“整个频道一个总音量旋钮”。

---

### Element-wise Gate

输出一个向量：

\[
\boldsymbol{\sigma}
=
[\sigma_1,\sigma_2,\dots,\sigma_d]
\]

融合变成：

\[
h'_i
=
(1-\sigma_i)h_i
+
\sigma_i s_i
\]

也就是每一个 hidden dimension 都有独立的 gate。

可以理解为：

\[
\boxed{\text{每一维拥有独立的信息接收强度}}
\]

注意：论文并没有证明 hidden state 的每一个坐标都有明确、单独可解释的语义。

---

## 11. Identity Coupling 中的 Layer Adapter

Identity coupling 有时需要额外 adapter，使后续 Transformer layers 适应混合后的 hidden state。

论文使用：

\[
h
\leftarrow
h+
\frac{\alpha}{r}
W_{up}
\left(
\operatorname{ReLU}(W_{down}h)
\right)
\]

其中：

- \(r\)：bottleneck rank；
- \(\alpha\)：scaling factor；
- \(W_{down},W_{up}\)：可训练参数。

例如：

\[
896
\rightarrow
64
\rightarrow
896
\]

其中：

\[
r=64
\]

控制 adapter capacity。

---

## 12. 训练时到底更新哪些参数？

### PULLSTANDARD

更新四个网络：

\[
f^{p\rightarrow a}
\]

\[
g^{p\rightarrow a}
\]

\[
f^{a\rightarrow p}
\]

\[
g^{a\rightarrow p}
\]

也就是其中所有 MLP 的：

\[
W,b
\]

而两个 LLM：

\[
M_p,M_a
\]

全部 frozen。

---

### PULLIDENTITY

因为：

\[
f(h)=h
\]

没有 translation network 参数。

主要训练：

\[
g^{p\rightarrow a},
\quad
g^{a\rightarrow p}
\]

以及可选的 layer adapters。

---

## 13. 损失函数设计

这是本文非常关键的一部分。

作者的总损失：

\[
\boxed{
L(\theta_\phi)
=
L_p(\theta_\phi)+L_a(\theta_\phi)
}
\]

其中：

- \(L_p\)：Primary next-token prediction loss；
- \(L_a\)：Auxiliary next-token prediction loss。

两个 LLM 冻结，真正被优化的是：

\[
\theta_\phi
\]

---

## 15. Primary Loss \(L_p\)

以：

```text
What is 564 * 848?
```

为例，Primary target 可能是：

```text
564 * 848 equals 478272.
```

Primary prompt token 不计 loss：

```text
What is 564 * 848 ? | 564 * 848 equals 478272 .
0    0   0  0   0 0 | 1   1   1    1      1
```

所以 \(L_p\) 主要监督：

> Auxiliary 的结果最终能否通过 reverse coupling 帮助 Primary 生成正确回答。

---

## 16. Auxiliary Loss \(L_a\)

Auxiliary target：

```text
calc(564*848)
```

因此它同样执行标准 NTP：

```text
calc → (
(    → 564
564  → *
*    → 848
848  → )
```

\(L_a\) 监督：

> Primary hidden state 经过 forward coupling 后，是否足以让 Auxiliary 正确理解任务并生成工具调用。

因此：

\[
\boxed{
L_a
\text{重点训练 }
M_p\rightarrow M_a
}
\]

---

## 17. Tool Output 不计算 Auxiliary Loss

Auxiliary 自己生成：

```text
calc(564*848)
```

calculator 强制返回：

```text
=478272;
```

其中：

```text
calc(564*848)
```

计算 Auxiliary CE；

而：

```text
=478272;
```

设置：

\[
m_t^{(a)}=0
\]

即 masked from loss。

原因是：

\[
\boxed{
\text{作者希望 Auxiliary 学会“调用计算器”，
而不是学会“自己模拟计算器”}
}
\]

虽然 tool output 不计算 Auxiliary loss，但它仍然会作为 token 输入 Auxiliary：

\[
=478272
\rightarrow
Embedding
\rightarrow
M_a
\rightarrow
h_a
\]

然后通过 reverse coupling 帮助 Primary。

---

## 18. Masked Cross Entropy

论文最终使用：

\[
L_p=
\frac{
\sum_t
m_t^{(p)}
\ell_t^{(p)}
}{
\sum_t m_t^{(p)}
}
\]

\[
L_a=
\frac{
\sum_t
m_t^{(a)}
\ell_t^{(a)}
}{
\sum_t m_t^{(a)}
}
\]

其中 \(m_t\) 为 loss mask。

主要 mask：

- Primary prompt：0；
- Auxiliary system prompt：0；
- tool-forced output：0；
- dropped auxiliary content：0；
- mirrored prompt：0；
- 正常 target token：1；
- wait token：可以设置为 \(w\in[0,1]\)。

两条 stream 分别做平均，再：

\[
L=L_p+L_a
\]

因此 Primary / Auxiliary 的 loss 不会因为有效 token 数量不同而天然失衡。

---

# 22. 实验结果

## 22.1 Arithmetic + Calculator

模型：

\[
M_p=M_a=\text{Qwen2.5-0.5B-Instruct}
\]

Auxiliary 独享 calculator。

![alt text](image-3.png)

核心结果：

| Configuration | Arithmetic | GSM8K | GSM8K-IRL |
|---|---:|---:|---:|
| Primary-only | 36.2 | 49.6 | 12.8 |
| Adapter-equivalent | 48.0 | 40.9 | 9.2 |
| Bicameral PULLSTANDARD | **96.5** | 39.1 | 17.1 |
| ELEMIDENTITY + adapters | 84.2 | 40.8 | **22.2** |
| SCALARIDENTITY + adapters | 68.1 ± 11.8 | 36.8 ± 1.5 | 17.5 ± 3.0 |

主要结论：

### 1. Calculator domain 上提升非常明显

\[
36.2\%
\rightarrow
96.5\%
\]

说明 hidden-state channel 可以有效传输任务信息和工具结果。

### 2. Adapter-equivalent 远弱于完整 Bicameral

\[
48.0\%
\ll
96.5\%
\]

说明收益不是来自额外参数量，而主要来自 Auxiliary + tool reasoning。

### 3. Identity 在 OOD 上反而更好

GSM8K-IRL：

\[
PULLSTANDARD=17.1\%
\]

\[
ELEMIDENTITY=22.2\%
\]

作者认为可能存在：

\[
\boxed{
\text{capacity-generalization tradeoff}
}
\]

自由度更大的 MLP translator 更容易拟合训练分布，而 identity coupling 的 inductive bias 可能更利于 OOD 泛化。

### 4. GSM8K 上反而下降

Primary-only：

\[
49.6\%
\]

Bicameral：

\[
39.1\%
\]

说明当 base model 本身已经较擅长任务时：

\[
\boxed{
\text{hidden-state perturbation 带来的噪声}
>
\text{tool capability 带来的收益}
}
\]


---

# 23. Figure 4：通信能力是如何形成的？
![alt text](image-4.png)

作者跟踪：

- forward coupling strength；
- reverse coupling strength；
- tool recall；
- task accuracy。

对比：

### a) Gated Identity / SCALARIDENTITY

\[
f(h)=h
\]

只学习 scalar gate。

### b) Gated MLP / PULLSTANDARD

\[
f=\text{MLP}
\]

既学习 translation，也学习 gate。

作者观察到总体顺序：

\[
\boxed{
\text{Forward coupling}
\rightarrow
\text{Tool recall}
\rightarrow
\text{Accuracy}
}
\]

其中 Gated Identity 初始 tool recall 就较高，因为相同模型的 raw hidden states 已经具有一定兼容性；

Gated MLP 初始 tool recall 接近 0，需要先学会 translator。

作者将 accuracy 的突然跃升描述为一种 phase-transition-like behavior。

---

# 24. Figure 5：Gate 学会了选择性通信
![alt text](image-5.png)
作者发现：

Forward coupling：

\[
M_p\rightarrow M_a
\]

会在类似：

```text
sum
total
all items
```

等任务相关 token 附近增强。

Reverse coupling：

\[
M_a\rightarrow M_p
\]

则在 calculator 返回：

```text
=2300.87
```

等工具结果时明显增强。

也就是说 suppression gate 没有人为规定通信 protocol，但通过 task loss 学会了：

\[
\boxed{\text{什么时候需要通信}}
\]

而不是全程持续强耦合。

---

# 25. Layer 选择

作者进行了 890 组 layer-wiring sweep。

![alt text](image-6.png)

结论：

\[
\boxed{\text{mid-to-late layers 最优}}
\]

大致集中在：

\[
10\sim20
\]

层。

浅层 coupling 明显较差。

作者认为较中后层已经形成较充分的 contextual / task representation，但尚未完全进入最终 token prediction 阶段，因此更适合作为通信位置。

另一个关键发现：

\[
\boxed{\text{within-token round trip 更优}}
\]

即：

\[
\text{Primary read}
\rightarrow
\text{Aux write}
\rightarrow
\text{Aux read}
\rightarrow
\text{Primary write}
\]

最好在同一个 token step 内完成。
