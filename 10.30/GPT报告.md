## 一、研究动机与文献依据

审稿人：*“The STE itself introduces training gradient bias. The paper conducts no ablation experiments for STE and fails to analyze the performance loss brought by STE.”*

模型中存在两处关键的不可微操作：**STE①（发送端语义量化）** 将连续特征转换为\(\{0,A\}\)（\(A=0.01\)）；**STE②（接收端解调硬判决）** 将Softmax得到的软比特转换为\(\{0,1\}\)。硬判决的真实导数几乎处处为0，因此直接使用硬判决会阻断上游网络的梯度传播。STE在前向保持硬判决，反向使用人为构造的替代梯度，但该梯度并不等于离散操作的真实导数，可能影响优化效果。

已有研究为本次实验提供了依据：

- **Yin等（ICLR 2019）[1]：** 比较Identity、ReLU与Clipped ReLU等不同STE，指出替代梯度的选择会显著影响量化网络的训练表现，支持“固定前向量化、改变反向梯度”的消融设计。
- **Guo等（ICML 2021）[2]：** 在神经图像压缩中比较STE与其他量化训练策略，说明编码器和解码器之间的量化瓶颈需要专门分析其训练近似方法。
- **Stan与Rhodes（Scientific Reports 2024）[3]：** 在二值脉冲网络中直接开展替代梯度函数消融，发现不同替代梯度的效果具有模型依赖性。

回答三个问题：
**1. 为什么可以通过替换STE的反向梯度进行消融？**
**2. 为什么有必要分析STE带来的性能影响？**
**3. 为什么STE①和STE②可能需要不同的替代梯度？**
1. 参考已有量化神经网络对STE梯度估计的研究，我们在保持前向硬量化不变的情况下，对不同反向梯度近似方法进行比较
2. 尽管STE能够使梯度穿过硬量化节点，但它的替代梯度可能引入优化偏差并影响学习到的特征表示，这一点在神经压缩研究中已经得到讨论
3. 替代梯度的效果依赖于具体网络结构和优化环境，因此不能假设某一种替代梯度在所有场景下都具有最佳性能

基于此，本次实验分为**两类**：① w/ STE与w/o STE，验证梯度传播的必要性；② Identity与Sigmoid的四组合消融，分析梯度近似方式的性能影响。两者回答的问题不同，不能将它们的性能差异都解释为“STE梯度偏差”。

## 二、实验机制与设计

### 1. Identity STE与Sigmoid替代梯度

对于输入\(x\)，设硬判决结果为\(Q(x)\)。两种方法**前向输出完全一致**，均为\(Q(x)\)；区别只在反向传播。

**Identity STE：** 前向硬判决，反向近似让梯度原样通过：

\[
y=x+\operatorname{sg}(Q(x)-x),\qquad \frac{\partial y}{\partial x}\approx1.
\]

**Sigmoid Surrogate：** 前向仍是同一个硬判决，反向在判决阈值附近传递较大的梯度、远离阈值时减小梯度：

\[
\frac{\partial y}{\partial x}\approx4\sigma\!\left(\frac{x-\theta}{T}\right)\left[1-\sigma\!\left(\frac{x-\theta}{T}\right)\right].
\]

这里\(\theta\)是判决阈值，\(T\)控制梯度集中程度，系数4使阈值处的梯度峰值为1。**STE①使用\(\theta=0,T=0.1\)；STE②使用\(\theta=0.5,T=0.2\)**。调制前的标准比特转换（原STE②所在位置）不作为本次消融变量。

### 2. 消融设置

- **四组合实验：** STE①、STE②分别取Identity或Sigmoid，共4组；每组在0/10 dB下使用10个随机种子（33–42），比较实体及三元组恢复指标，并观察Loss曲线。
- **w/o STE实验：** STE①与STE②同时改成直接返回硬判决值，不再构造替代梯度；分别在0/10 dB下使用3个随机种子（40–42）。
- **梯度检查：** 使用`tf.gradients(loss, ...)`检查发送端编码器参数和接收端解码器输入的计算图连接状态。

## 三、实验结果

### 1. 四种替代梯度组合（10个种子）

下表为**平均值±样本标准差**，单位均为百分比（%）。

| SNR | STE① / STE② | 实体恢复准确率 | 去重实体准确率 | 三元组精确恢复率 | 频率平衡三元组恢复率 |
|---|---|---:|---:|---:|---:|
| 0 dB | Identity / Identity | 69.19±2.91 | 58.66±3.20 | 25.43±4.97 | 35.44±4.68 |
| 0 dB | **Sigmoid / Identity** | **75.89±2.68** | **67.42±2.37** | **36.68±4.77** | **47.25±5.52** |
| 0 dB | Identity / Sigmoid | 65.70±3.59 | 55.41±3.33 | 23.54±4.31 | 32.61±3.66 |
| 0 dB | Sigmoid / Sigmoid | 74.96±2.30 | 65.98±1.82 | 35.05±4.42 | 46.10±4.75 |
| 10 dB | Identity / Identity | 84.19±4.22 | 82.53±1.97 | 55.21±11.70 | 71.37±6.72 |
| 10 dB | **Sigmoid / Identity** | **93.09±5.67** | **88.87±1.74** | **79.94±16.47** | **88.32±5.56** |
| 10 dB | Identity / Sigmoid | 81.97±1.79 | 78.92±1.87 | 49.40±4.27 | 64.79±6.20 |
| 10 dB | Sigmoid / Sigmoid | 90.42±5.95 | 87.78±1.80 | 72.01±17.42 | 83.70±6.57 |

**观察：**

- 固定STE②为Identity时，将STE①改为Sigmoid，三元组精确恢复率在**0 dB提升11.25个百分点**（25.43%→36.68%），在**10 dB提升24.73个百分点**（55.21%→79.94%）。
- 固定STE①为Identity时，将STE②改为Sigmoid，三元组精确恢复率反而下降：0 dB约下降1.89个百分点，10 dB约下降5.82个百分点。
- 两处同时采用Sigmoid并未超过**STE① Sigmoid + STE② Identity**；该组合在两个SNR下均取得最高的四项指标平均值。
- 训练Loss曲线表明，**STE①采用Sigmoid的配置下降更快、最终Loss更低**；改变STE②未带来同等改善。原图未明确说明曲线是否为10种子的均值，因此这里只讨论曲线趋势。

### 2. w/ STE与w/o STE（3个种子）

| SNR | 方法 | 实体恢复准确率 | 去重实体准确率 | 三元组精确恢复率 | 频率平衡三元组恢复率 |
|---|---|---:|---:|---:|---:|
| 0 dB | w/o STE①② | 28.55% | 17.18% | 2.21% | 3.01% |
| 0 dB | **w/ STE①②** | **71.76%** | **61.00%** | **29.98%** | **39.20%** |
| 10 dB | w/o STE①② | 81.38% | 73.20% | 47.15% | 59.15% |
| 10 dB | **w/ STE①②** | **82.34%** | **81.10%** | **50.10%** | **66.59%** |

**观察：** 在0 dB，移除STE后三元组精确恢复率由29.98%降至2.21%，下降27.77个百分点；在10 dB，该指标差距缩小至2.95个百分点，但去重实体准确率和频率平衡三元组恢复率仍分别低7.90和7.43个百分点。由于只有3个种子，这部分作为辅助机制验证，不作强统计显著性结论。

### 3. 梯度传播验证

| 检查位置 | w/o STE①② | w/ STE①② |
|---|---|---|
| 发送端`fcl1`与`highway1`的全部受检参数 | DISCONNECTED | CONNECTED |
| 编码器输出`highway_1` | DISCONNECTED | CONNECTED |
| 解码器输入`quantization_after_channel` | CONNECTED | CONNECTED |

验证表明，移除两处STE后，重建损失的梯度无法回传至发送端编码器，但**解码器仍然可以更新**，因为其参数梯度不需要穿过前面的硬判决节点。`CONNECTED`仅表示存在梯度路径，并不保证梯度数值始终非零。


最终损失：
$$
\mathcal L=\|X-\hat X\|^2
$$

解码器参数损失：

$$
\boxed{
\frac{\partial\mathcal L}{\partial\theta_{\mathrm{dec}}}
=
\frac{\partial\mathcal L}{\partial\hat X}
\frac{\partial\hat X}{\partial\theta_{\mathrm{dec}}}
}
$$
编码器参数损失：

$$
\boxed{
\frac{\partial\mathcal L}{\partial\theta_{\mathrm{enc}}}
=
\frac{\partial\mathcal L}{\partial\hat X}
\frac{\partial\hat X}{\partial\hat B}
\frac{\partial\hat B}{\partial H}
\frac{\partial H}{\partial\theta_{\mathrm{enc}}}
}
$$

$\frac{\partial\hat B}{\partial H}$时经过硬量化、16QAM和硬盘决，当把STE改为return binary的时候，反向传播路径就断开了。

w/ STE： 编码器和解码器能够通过重建损失联合优化。
w/o STE： 编码器的重建损失梯度路径断开，但解码器仍然能够学习如何从固定的二值表示恢复语义特征。

## 四、分析与总结

1. **STE的作用是维持端到端梯度传播。** w/o STE并非整个网络不能训练，而是发送端编码器失去通过当前重建损失学习的能力。10 dB下固定的二值编码加上可训练解码器仍可能取得一定性能；0 dB下其抗噪声能力不足，性能显著下降。这是与现象一致的解释，仍需误码率或编码分布分析进一步验证。
2. **不同替代梯度会影响优化质量。** Identity与Sigmoid保持相同的前向通信过程，因此四组合实验能够较清晰地隔离反向梯度近似策略的作用。发送端的阈值感知Sigmoid提高了平均恢复性能；接收端使用Sigmoid则没有取得相同收益。
3. **结论具有位置依赖性。** 当前最佳配置是**STE①：Sigmoid（T=0.1）；STE②：Identity**，说明不能对所有离散节点统一使用同一种替代梯度。
4. **不能将上述差距视为STE的绝对偏差。** Sigmoid本身也是有偏替代梯度；四组合消融衡量的是估计策略间的性能差异，而w/o STE衡量的是梯度路径被阻断后的影响。此外，10 dB下三元组精确恢复率的种子间标准差较大，高频实体/三元组恢复差异可能参与造成波动，尚需专门验证。

**总体结论：** 本次消融从“是否保留梯度传播”和“如何估计离散节点梯度”两个层面回应了审稿意见，既说明STE对端到端训练的必要性，也展示了原始Identity STE在发送端量化处相对于Sigmoid替代梯度的可观测性能局限。

## 五、参考文献

[1] Yin, P., Lyu, J., Zhang, S., et al. **Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets.** *ICLR*, 2019. https://arxiv.org/abs/1903.05662

[2] Guo, Z., Zhang, Z., Feng, R., and Chen, Z. **Soft then Hard: Rethinking the Quantization in Neural Image Compression.** *ICML*, 2021. https://proceedings.mlr.press/v139/guo21c.html

[3] Stan, M.-I., and Rhodes, O. **Learning long sequences in spiking neural networks.** *Scientific Reports*, 14, 21957, 2024. https://doi.org/10.1038/s41598-024-71678-8

