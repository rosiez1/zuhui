# Q-KVComm 跨模型通信实验周报

时间：2026-05-27  
项目：`qkvcomm`  
主题：显式跨模型通信封装与 TinyLlama -> Qwen2.5 异构 KV 通信实验

## 一、本轮工作概述

本轮主要围绕论文中的跨模型通信部分推进实验实现，重点完成了两件事：

1. 将原先隐式的 KV 提取、量化、注入流程，封装为显式的 `sender -> payload -> receiver` 通信结构。
2. 接入 `Qwen/Qwen2.5-1.5B-Instruct`，并实现 `TinyLlama/TinyLlama-1.1B-Chat-v1.0 -> Qwen/Qwen2.5-1.5B-Instruct` 的异构模型 KV cache 通信路径。

本轮暂时跳过了论文中的 `hybrid information extraction`，直接推进异构模型 calibration。

## 二、已完成代码实现

### 1. 显式通信封装

新增文件：

```text
core/communication.py
```

主要类：

```python
QKVSender
QKVCommPayload
QKVReceiver
```

当前通信流程为：

```text
Sender model 处理上下文
-> 提取 past_key_values
-> adaptive / uniform 量化
-> bit packing
-> 构造 QKVCommPayload
-> Receiver 接收 payload
-> 反量化 KV
-> 注入 receiver model
-> 继续生成
```

### 2. Qwen2.5 模型接入

已下载并验证模型：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

本地缓存位置：

```text
C:\Users\Administrator\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct
```

实测 Qwen2.5 KV 结构：

```text
num_hidden_layers = 28
num_key_value_heads = 2
head_dim = 128
```

TinyLlama KV 结构：

```text
num_hidden_layers = 22
num_key_value_heads = 4
head_dim = 64
```

因此 TinyLlama 的 KV cache 不能直接注入 Qwen，需要异构 calibration 和 shape adapter。

### 3. 异构模型 Calibration

新增文件：

```text
core/calibration.py
```

当前实现的跨模型 calibration 包括：

```text
TinyLlama KV
-> scalar mean/std 对齐
-> layer mapping
-> head/head_dim shape adapter
-> Qwen KV-compatible cache
```

#### 3.1 标量均值/方差对齐

对每层 K/V 计算：

```text
mean, std
```

然后使用论文中的统计对齐思路：

```text
KV_calibrated = (KV_source - mean_source) / std_source * std_target + mean_target
```

这里目前采用的是 scalar 级别统计，而不是 per-dim 或 learned projection。

#### 3.2 层映射

TinyLlama 有 22 层，Qwen2.5 有 28 层。

当前采用线性 layer mapping：

```text
TinyLlama 22 layers -> Qwen 28 layers
```

示例映射：

```text
[0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, ...]
```

#### 3.3 Shape Adapter

TinyLlama KV shape：

```text
[batch, 4, seq_len, 64]
```

Qwen2.5 KV shape：

```text
[batch, 2, seq_len, 128]
```

当前使用插值方式做 shape adapter：

```text
KV heads: 4 -> 2
head_dim: 64 -> 128
```

这一步主要保证张量形状能被 Qwen 接收。

## 三、实验结果

### 1. 跨模型 smoke test

运行命令：

```powershell
C:\Coder\anaconda3\envs\qkvcomm\python.exe qkvcomm_demo.py --max_new_tokens 2 --out_dir ./outputs_cross_model_smoke
```

实验链路：

```text
TinyLlama sender
-> adaptive quantized payload
-> Qwen receiver
-> calibration
-> KV injection
-> generation
```

结果：

```text
Qwen 成功接收 TinyLlama 的 KV payload
Qwen forward 成功
Qwen generation 成功
```

短生成输出：

```text
A key advantage is used to
```

### 2. 压缩结果

本次样例中：

```text
original_past_key_values_bytes = 653312
quantized_payload_bytes = 248880
compression_ratio = 2.625x
```

说明 adaptive quantization 仍然有效减少了 KV payload 大小。

### 3. 长一点生成时的问题

当 `max_new_tokens = 20` 时，输出出现明显重复：

```text
A key advantage is used to is used to is used to is used to is used in is used in is used in
```

这说明当前异构 KV 注入虽然能跑通，但生成质量较差。

## 四、当前跨模型通信方法总结

当前 TinyLlama -> Qwen2.5 通信本质上采用的是：

```text
zero-shot statistical calibration + shape adapter
```

也就是：

1. 不训练额外参数。
2. 不学习 TinyLlama 到 Qwen 的真实语义映射。
3. 只通过统计分布对齐和张量形状变换，让 TinyLlama KV 能被 Qwen 接收。

当前方法可以概括为：

```text
让 KV “形状像 Qwen”
让 KV “数值分布像 Qwen”
但不能保证 KV “语义空间像 Qwen”
```

## 五、效果较差的本质原因

### 1. 均值/方差对齐能力太弱

mean/std 只能调整整体数值范围，无法对齐高维语义方向。

Qwen 的 query 会和传入的 key 做 attention dot product。如果 key 的语义方向没有对齐，即使均值和方差接近，attention score 也不一定有意义。

### 2. Shape adapter 只是工程适配，不是语义映射

当前把 TinyLlama 的：

```text
[4 heads, 64 dim]
```

插值成 Qwen 的：

```text
[2 heads, 128 dim]
```

这只是让张量 shape 对上，不能保证 TinyLlama 的 attention head 和 Qwen 的 attention head 有正确语义对应关系。

### 3. 层语义不一定对应

TinyLlama 第 10 层和 Qwen 第 13 层不一定处在相同抽象阶段。当前线性 layer mapping 只是一个简化假设。

### 4. Tokenizer 和 RoPE 差异会破坏 KV 兼容性

KV cache 不是普通 embedding，它绑定了：

```text
token 序列
position_ids
RoPE 相位
attention head 结构
```

TinyLlama 和 Qwen 的 tokenizer、position 编码细节、训练分布不同，都会导致 KV cache 难以直接迁移。

### 5. Qwen 接收到的是 out-of-distribution KV

对 Qwen 来说，校准后的 TinyLlama KV 并不是它训练过程中自然产生的内部状态。

因此 Qwen 后续生成时容易出现：

```text
重复
退化
语义不稳定
```

### 6. Greedy decoding 会放大重复问题

当前生成主要使用 argmax greedy decoding。一旦模型进入高概率重复片段，例如：

```text
is used to
```

它会继续沿着重复路径滚下去。

不过这只是表面问题。即使加入 repetition penalty 或 sampling，也只能缓解重复，不能解决跨模型 KV 语义错位的根因。

## 六、阶段性结论

### 可信部分

同构模型 KV 通信是比较可信的：

```text
TinyLlama -> TinyLlama
```

原因是：

```text
模型结构一致
tokenizer 一致
层数一致
head 语义一致
RoPE 设置一致
KV 空间一致
```

因此同构情况下，KV cache 量化、传输、反量化、注入是合理的。

### 存疑部分

异构模型 KV 通信目前效果较弱：

```text
TinyLlama -> Qwen2.5
```

当前实验说明：

```text
路径可以跑通
但语义质量不好
```

因此，论文中如果声称仅靠 zero-shot calibration 就能稳定实现异构模型 KV 通信，需要谨慎看待。

当前更合理的判断是：

```text
zero-shot mean/std calibration 可以作为工程 baseline
但不足以实现高质量跨模型通信
```

## 七、后续建议

### 1. 做 ablation 实验

建议比较：

```text
Qwen only
Qwen + random calibrated KV
Qwen + TinyLlama calibrated KV
Qwen + true Qwen KV
TinyLlama -> TinyLlama KV
```

如果：

```text
TinyLlama calibrated KV ~= random calibrated KV
```

说明当前异构 KV 主要是噪声，而不是有效通信。

### 2. 升级 calibration 方法

后续可以尝试：

```text
scalar mean/std
-> per-layer mean/std
-> per-head mean/std
-> per-dim mean/std
-> learned linear projection
-> trained adapter
```

### 3. 引入 learned projection

更可靠的方向是训练一个小投影器：

```text
TinyLlama KV -> Qwen KV
```

训练目标可以是同一批文本下：

```text
TinyLlama KV as input
Qwen KV as target
```

学习每层或每组层的映射关系。

### 4. 检查同源模型通信

相比 TinyLlama -> Qwen，后续可以尝试：

```text
Qwen2.5 -> Qwen3
Qwen2.5-1.5B -> Qwen2.5-3B
Qwen2.5-Instruct -> Qwen2.5-Base
```

同源模型的 tokenizer、架构和训练分布更接近，成功概率会高于 TinyLlama -> Qwen。

## 八、本轮结论一句话

本轮已经实现了 TinyLlama -> Qwen2.5 的跨模型 KV 通信路径，并成功跑通 forward 与 generation；但当前方法本质上只是 zero-shot 统计校准加 shape adapter，能解决“能不能注入”的问题，不能解决“语义是否真正对齐”的问题，因此生成质量较差，出现重复是符合预期的。

